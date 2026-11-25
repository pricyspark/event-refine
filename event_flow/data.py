"""Sequence manifests, FRED import, and contiguous supervised clips.

Developed with Codex assistance. Missing labels never imply negative examples.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .events import load_event_map


@dataclass(frozen=True)
class Frame:
    rgb: Path
    events: Path
    timestamp: float
    boxes: tuple[tuple[float, float, float, float], ...]
    labels: tuple[int, ...]
    annotated: bool
    event_encoding: str
    event_background: tuple[int, ...] | None


@dataclass(frozen=True)
class Sequence:
    name: str
    split: str
    frames: tuple[Frame, ...]


@dataclass
class FrameSample:
    rgb: torch.Tensor
    events: torch.Tensor
    boxes: torch.Tensor
    labels: torch.Tensor
    timestamp: float


def load_manifest(path: str | Path) -> tuple[tuple[str, ...], tuple[Sequence, ...]]:
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported Event-Flow manifest version.")
    classes = tuple(manifest["classes"])
    if (
        not classes
        or len(set(classes)) != len(classes)
        or not all(isinstance(name, str) and name for name in classes)
    ):
        raise ValueError("Manifest classes must be unique, non-empty names.")
    sequences = []
    names: set[str] = set()
    for record in manifest["sequences"]:
        name = record["id"]
        if name in names:
            raise ValueError(f"Sequence ID occurs more than once: {name}")
        names.add(name)
        frames = []
        previous_time = -float("inf")
        for item in record["frames"]:
            timestamp = float(item["timestamp"])
            if not np.isfinite(timestamp) or timestamp <= previous_time:
                raise ValueError(f"Frame timestamps must strictly increase in {name}.")
            previous_time = timestamp
            annotated = item.get("annotated", True)
            if not isinstance(annotated, bool):
                raise ValueError("annotated must be a boolean.")
            if annotated and ("boxes" not in item or "labels" not in item):
                raise ValueError(
                    "Annotated frames must explicitly supply boxes and labels (possibly empty)."
                )
            boxes = tuple(tuple(float(v) for v in box) for box in item.get("boxes", []))
            labels = tuple(item.get("labels", []))
            if len(boxes) != len(labels) or any(
                not isinstance(label, int) or not 0 <= label < len(classes)
                for label in labels
            ):
                raise ValueError(f"Invalid box/label rows in {name}.")
            if any(
                len(box) != 4
                or not np.isfinite(box).all()
                or box[0] >= box[2]
                or box[1] >= box[3]
                for box in boxes
            ):
                raise ValueError(
                    "Boxes must be finite pixel xyxy with positive dimensions."
                )
            background = item.get("event_background")
            frames.append(
                Frame(
                    path.parent / item["rgb"],
                    path.parent / item["events"],
                    timestamp,
                    boxes,
                    labels,
                    annotated,
                    item.get("event_encoding", "binary"),
                    tuple(background) if background is not None else None,
                )
            )
        if not frames:
            raise ValueError(f"Empty sequence: {name}")
        sequences.append(Sequence(name, record["split"], tuple(frames)))
    if not sequences:
        raise ValueError("Manifest has no sequences.")
    return classes, tuple(sequences)


def read_frame(frame: Frame) -> FrameSample:
    with Image.open(frame.rgb) as image:
        rgb = (
            torch.from_numpy(np.asarray(image.convert("RGB")).copy())
            .permute(2, 0, 1)
            .float()
            / 255
        )
    events = load_event_map(
        frame.events, encoding=frame.event_encoding, background=frame.event_background
    )
    if rgb.shape[-2:] != events.shape[-2:]:
        raise ValueError(
            f"RGB/event dimensions differ; register/pad before loading: {frame.rgb}"
        )
    boxes = torch.tensor(frame.boxes, dtype=torch.float32).reshape(-1, 4)
    height, width = rgb.shape[-2:]
    if boxes.numel() and bool(
        ((boxes < 0) | (boxes > boxes.new_tensor([width, height, width, height]))).any()
    ):
        raise ValueError(f"Annotations lie outside the registered image: {frame.rgb}")
    return FrameSample(
        rgb,
        events,
        boxes,
        torch.tensor(frame.labels, dtype=torch.long),
        frame.timestamp,
    )


class ClipDataset(Dataset[list[FrameSample]]):
    def __init__(
        self,
        manifest: str | Path,
        *,
        split: str,
        clip_length: int = 4,
        max_gap: float = 0.05,
    ) -> None:
        if clip_length < 1 or not np.isfinite(max_gap) or max_gap <= 0:
            raise ValueError("clip_length and max_gap must be positive.")
        self.classes, sequences = load_manifest(manifest)
        self.clips: list[tuple[Frame, ...]] = []
        for sequence in sequences:
            if sequence.split != split:
                continue
            for start in range(len(sequence.frames) - clip_length + 1):
                clip = sequence.frames[start : start + clip_length]
                if all(frame.annotated for frame in clip) and all(
                    0 < right.timestamp - left.timestamp <= max_gap
                    for left, right in zip(clip, clip[1:])
                ):
                    self.clips.append(clip)
        if not self.clips:
            raise ValueError(f"No contiguous supervised clips for split {split!r}.")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> list[FrameSample]:
        return [read_frame(frame) for frame in self.clips[index]]


def _natural_key(path: Path) -> list[int | str]:
    return [
        int(piece) if piece.isdigit() else piece
        for piece in re.split(r"(\d+)", path.name)
    ]


def import_fred(
    root: Path,
    output: Path,
    *,
    event_background: tuple[int, int, int],
    frame_period: float = 0.033333,
) -> None:
    """Import extracted official FRED train/test folders into our manifest.

    Frame ordering and the default annotation timestamp cadence follow the
    official loader. Rendered event backgrounds must be chosen explicitly.
    Missing annotation timestamps remain unsupervised, not empty negatives.
    """
    if not np.isfinite(frame_period) or frame_period <= 0:
        raise ValueError("frame_period must be positive and finite.")
    sequences = []
    for split in ("train", "validation", "test"):
        for folder in sorted((root / split).glob("*/"), key=_natural_key):
            rgb_paths = sorted((folder / "RGB").glob("*.jpg"), key=_natural_key)
            event_paths = sorted(
                (folder / "Event" / "Frames").glob("*.png"), key=_natural_key
            )
            if not rgb_paths or len(rgb_paths) != len(event_paths):
                raise ValueError(f"Missing/mismatched RGB/event frames in {folder}")
            # Both official modalities are paired by natural frame order. No
            # frames are filtered before annotation timestamps are assigned.
            coordinates: dict[float, list[tuple[float, float, float, float]]] = {}
            annotation_path = folder / "coordinates.txt"
            for line_number, line in enumerate(
                annotation_path.read_text().splitlines(), 1
            ):
                if not line.strip():
                    continue
                try:
                    stamp, values = line.split(":", 1)
                    fields = [value.strip() for value in values.split(",")]
                    if len(fields) < 6:
                        raise ValueError("Expected xyxy, identity, and class")
                    box = tuple(float(value) for value in fields[:4])
                    if len(box) != 4:
                        raise ValueError("Expected four box coordinates")
                    coordinates.setdefault(round(float(stamp), 6), []).append(box)
                except ValueError as error:
                    raise ValueError(
                        f"Malformed annotation {annotation_path}:{line_number}"
                    ) from error
            used_timestamps: set[float] = set()
            frames = []
            for index, (rgb, event) in enumerate(
                zip(rgb_paths, event_paths, strict=True)
            ):
                timestamp = round((index + 1) * frame_period, 6)
                boxes = coordinates.get(timestamp)
                if boxes is not None:
                    used_timestamps.add(timestamp)
                frames.append(
                    {
                        "rgb": str(rgb.resolve()),
                        "events": str(event.resolve()),
                        "timestamp": timestamp,
                        "annotated": boxes is not None,
                        "boxes": boxes or [],
                        "labels": [0] * len(boxes or []),
                        "event_encoding": "rendered",
                        "event_background": event_background,
                    }
                )
            if coordinates.keys() - used_timestamps:
                raise ValueError(
                    f"Unmatched annotation timestamps in {folder}; verify --frame-period and frame ordering."
                )
            if not used_timestamps:
                raise ValueError(f"No supervised frames in {folder}.")
            sequences.append(
                {"id": f"{split}/{folder.name}", "split": split, "frames": frames}
            )
    if not sequences:
        raise ValueError("No extracted FRED sequence folders found.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"format_version": 1, "classes": ["drone"], "sequences": sequences},
            indent=2,
        )
        + "\n"
    )
