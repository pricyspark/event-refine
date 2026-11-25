"""Pinned, selective FRED download and deterministic import.

Developed with Codex assistance. Archives are cached, never committed to Git.
"""

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import platform
from importlib.metadata import version
import tempfile
from zipfile import ZipFile

from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image

from flowsis.artifacts import atomic_write_text
from .data import load_manifest, read_frame
from .events import save_event_map


@dataclass(frozen=True)
class VideoSelection:
    archive: str
    split: str
    sha256: str
    size_bytes: int
    frame_start: int
    frame_stop: int | None
    frame_fraction: float = 1.0


@dataclass(frozen=True)
class FredConfig:
    repo_id: str
    revision: str
    videos: tuple[VideoSelection, ...]
    event_background: tuple[int, int, int]
    event_palette: tuple[tuple[int, int, int], ...]
    missing_annotations: str
    content_hash: str
    source: dict[str, object]


def load_config(path: Path) -> FredConfig:
    raw = json.loads(path.read_text())
    if raw.get("format_version") != 1 or raw.get("repo_id") != "GabrieleMagrini/FRED":
        raise ValueError("Expected a version-1 official FRED download config.")
    revision = raw["revision"]
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Pin revision to a full 40-character commit SHA.")
    preprocess = raw["preprocess"]
    if (
        preprocess.get("boxes") != "extended"
        or preprocess.get("clip_boxes") is not True
        or preprocess.get("class_mapping") != "single-drone"
    ):
        raise ValueError(
            "Expected extended boxes, clipping, and single-drone class mapping."
        )
    if preprocess.get("timestamp_source") != "event_filename_us":
        raise ValueError("Expected event_filename_us timestamps.")
    missing = preprocess["missing_annotations"]
    if missing not in {"unlabeled", "empty"}:
        raise ValueError("missing_annotations must be unlabeled or empty.")
    background = tuple(preprocess["event_background"])
    palette = tuple(tuple(color) for color in preprocess["event_palette"])
    if (
        any(
            len(color) != 3
            or any(type(v) is not int or not 0 <= v <= 255 for v in color)
            for color in (background, *palette)
        )
        or background not in palette
    ):
        raise ValueError(
            "Expected RGB byte colors and background included in event_palette."
        )
    videos = []
    seen = set()
    for item in raw["videos"]:
        archive, split = item["archive"], item["split"]
        if re.fullmatch(r"(train|test)/[0-9]+\.zip", archive) is None:
            raise ValueError(f"Invalid sequence archive: {archive}")
        if archive in seen or split not in {"train", "validation", "test"}:
            raise ValueError("Each sequence must occur once and have a valid split.")
        if archive.startswith("test/") and split != "test":
            raise ValueError("Official test sequences must retain the test split.")
        if archive.startswith("train/") and split == "test":
            raise ValueError("Use official test sequences for the test split.")
        seen.add(archive)
        checksum, size = item["sha256"], item["size_bytes"]
        start, stop = item["frame_start"], item["frame_stop"]
        fraction = item.get("frame_fraction", 1.0)
        if (
            type(fraction) not in (int, float)
            or not np.isfinite(fraction)
            or not 0 < fraction <= 1
        ):
            raise ValueError("frame_fraction must be finite and in (0,1].")
        if fraction != 1 and (start != 0 or stop is not None):
            raise ValueError(
                "Fractional selection requires frame_start=0 and frame_stop=null."
            )
        if (
            re.fullmatch(r"[0-9a-f]{64}", checksum) is None
            or type(size) is not int
            or size <= 0
        ):
            raise ValueError("Archive SHA256 and positive size_bytes are required.")
        if (
            type(start) is not int
            or start < 0
            or stop is not None
            and (type(stop) is not int or stop <= start)
        ):
            raise ValueError("Frame bounds must define a nonempty [start, stop) range.")
        videos.append(
            VideoSelection(archive, split, checksum, size, start, stop, float(fraction))
        )
    if not videos:
        raise ValueError("Select at least one video.")
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    return FredConfig(
        raw["repo_id"],
        revision,
        tuple(videos),
        background,
        palette,
        missing,
        hashlib.sha256(encoded).hexdigest(),
        raw,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_plan(config: FredConfig) -> dict[str, object]:
    return {
        "repo_id": config.repo_id,
        "revision": config.revision,
        "config_sha256": config.content_hash,
        "video_count": len(config.videos),
        "download_bytes": sum(video.size_bytes for video in config.videos),
        "videos": [
            {
                "archive": video.archive,
                "split": video.split,
                "size_bytes": video.size_bytes,
                "frame_start": video.frame_start,
                "frame_stop": video.frame_stop,
                "frame_fraction": video.frame_fraction,
            }
            for video in config.videos
        ],
    }


def _frame_timestamp(name: str) -> int:
    match = re.search(r"_frame_(\d+)\.[^.]+$", name)
    if match is None:
        raise ValueError(f"Cannot read frame timestamp from FRED filename: {name}")
    return int(match.group(1))


def _rgb_timestamp(name: str) -> int:
    match = re.search(r"_(\d{2})_(\d{2})_(\d{2})\.(\d{6})\.jpg$", name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot read RGB clock timestamp from FRED filename: {name}")
    hours, minutes, seconds, micros = map(int, match.groups())
    if hours >= 24 or minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid RGB clock timestamp: {name}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1_000_000 + micros


def _import_video(
    archive_path: Path,
    video: VideoSelection,
    config: FredConfig,
    output: Path,
    *,
    compress_events: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    sequence_id = PurePosixPath(video.archive).stem
    source_split = PurePosixPath(video.archive).parent.name
    prefix = f"{sequence_id}/"
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate ZIP member names: {video.archive}")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError(f"Unsafe ZIP member: {name}")
        rgb_names = sorted(
            (
                name
                for name in names
                if name.startswith(prefix + "RGB/") and name.lower().endswith(".jpg")
            ),
            key=_rgb_timestamp,
        )
        event_names = sorted(
            (
                name
                for name in names
                if name.startswith(prefix + "Event/Frames/")
                and name.lower().endswith(".png")
            ),
            key=_frame_timestamp,
        )
        if not rgb_names or len(rgb_names) != len(event_names):
            raise ValueError(f"Missing/mismatched RGB/event pairs in {video.archive}")
        for ordered, stamp_function in (
            (rgb_names, _rgb_timestamp),
            (event_names, _frame_timestamp),
        ):
            stamps = [stamp_function(name) for name in ordered]
            if any(right <= left for left, right in zip(stamps, stamps[1:])):
                raise ValueError("Frame timestamps must strictly increase.")
        if (
            _rgb_timestamp(rgb_names[-1]) - _rgb_timestamp(rgb_names[0])
            > 12 * 3600 * 1_000_000
        ):
            raise ValueError(
                "RGB clock crosses midnight; import requires explicit clock unwrapping."
            )
        stop = (
            int(len(rgb_names) * video.frame_fraction)
            if video.frame_stop is None
            else video.frame_stop
        )
        if not video.frame_start < stop <= len(rgb_names):
            raise ValueError(
                f"Frame range exceeds {len(rgb_names)} frames in {video.archive}"
            )
        annotations: dict[int, list[tuple[list[float], int, str]]] = {}
        coordinate_name = prefix + "coordinates.txt"
        for number, line in enumerate(
            archive.read(coordinate_name).decode("utf-8-sig").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                timestamp, values = line.split(":", 1)
                x1, y1, x2, y2, track, label = [
                    value.strip() for value in values.split(",", 5)
                ]
                box = [float(x1), float(y1), float(x2), float(y2)]
                stamp = float(timestamp)
                if (
                    not np.isfinite([stamp, *box]).all()
                    or box[0] >= box[2]
                    or box[1] >= box[3]
                ):
                    raise ValueError("Invalid timestamp/box")
                annotations.setdefault(round(stamp * 1e6), []).append(
                    (box, int(track), label)
                )
            except ValueError as error:
                raise ValueError(f"Malformed {coordinate_name}:{number}") from error
        all_stamps = {_frame_timestamp(name) for name in event_names}
        if annotations.keys() - all_stamps:
            raise ValueError(
                f"Annotation timestamps do not match event filenames in {video.archive}"
            )
        frames = []
        clipped_count = 0
        dropped_count = 0
        sequence_path = Path("sequences") / f"{source_split}-{sequence_id}"
        for index in range(video.frame_start, stop):
            rgb_name, event_name = rgb_names[index], event_names[index]
            rgb_data = archive.read(rgb_name)
            with Image.open(io.BytesIO(rgb_data)) as rgb:
                rgb.load()
                size = rgb.size
            with Image.open(io.BytesIO(archive.read(event_name))) as event:
                event = event.convert("RGB")
                if event.size != size:
                    raise ValueError(
                        f"RGB/event size mismatch: {video.archive} frame {index}"
                    )
                colors = event.getcolors(maxcolors=len(config.event_palette))
                if colors is None or any(
                    color not in config.event_palette for _, color in colors
                ):
                    raise ValueError(f"Unexpected rendered-event palette: {event_name}")
                occupancy = (
                    np.asarray(event) != np.asarray(config.event_background)
                ).any(axis=-1)
            stamp = _frame_timestamp(event_name)
            source_boxes = annotations.get(stamp)
            boxes, track_ids, source_classes = [], [], []
            for box, track_id, label in source_boxes or []:
                clipped = np.clip(box, 0, [size[0], size[1], size[0], size[1]]).tolist()
                clipped_count += int(clipped != box)
                if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                    dropped_count += 1
                    continue
                boxes.append(clipped)
                track_ids.append(track_id)
                source_classes.append(label)
            rgb_relative = sequence_path / "RGB" / f"{index:06d}.jpg"
            event_relative = (
                sequence_path
                / "events"
                / f"{index:06d}{'.npz' if compress_events else '.npy'}"
            )
            (output / rgb_relative).parent.mkdir(parents=True, exist_ok=True)
            (output / event_relative).parent.mkdir(parents=True, exist_ok=True)
            (output / rgb_relative).write_bytes(rgb_data)
            save_event_map(output / event_relative, occupancy)
            frames.append(
                {
                    "rgb": rgb_relative.as_posix(),
                    "events": event_relative.as_posix(),
                    "timestamp": stamp / 1e6,
                    "event_timestamp_us": stamp,
                    "rgb_clock_timestamp_us": _rgb_timestamp(rgb_name),
                    "source_frame_index": index,
                    "annotated": source_boxes is not None
                    or config.missing_annotations == "empty",
                    "boxes": boxes,
                    "labels": [0] * len(boxes),
                    "track_ids": track_ids,
                    "source_classes": source_classes,
                    "event_encoding": "binary",
                }
            )
    if not any(frame["annotated"] for frame in frames):
        raise ValueError(
            f"No supervised frames in selected range of {video.archive}; select a later/larger range."
        )
    return (
        {"id": f"{source_split}/{sequence_id}", "split": video.split, "frames": frames},
        {
            "archive": video.archive,
            "sha256": video.sha256,
            "size_bytes": video.size_bytes,
            "split": video.split,
            "total_source_frames": len(rgb_names),
            "frame_start": video.frame_start,
            "frame_stop": stop,
            "frame_fraction": video.frame_fraction,
            "imported_frames": len(frames),
            "supervised_frames": sum(frame["annotated"] for frame in frames),
            "clipped_boxes": clipped_count,
            "dropped_boxes": dropped_count,
        },
    )


def download_fred(
    config_path: Path,
    output: Path,
    *,
    cache_dir: Path,
    dry_run: bool = False,
    compress_events: bool = True,
) -> dict[str, object]:
    """Download only selected archives, verify checksums, import atomically.

    A dry run reads config only and performs no network or filesystem writes.
    Whole ZIPs are downloaded even when selecting a smaller frame interval.
    """
    config = load_config(config_path)
    event_storage = "packed-npz-v1" if compress_events else "npy"
    plan = {**download_plan(config), "event_storage": event_storage}
    if dry_run:
        return plan
    output = output.resolve()
    if output.exists():
        provenance_path = output / "provenance.json"
        if not provenance_path.is_file():
            raise FileExistsError(f"Refusing to overwrite existing directory: {output}")
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("config_sha256") != config.content_hash:
            raise FileExistsError(
                "Output belongs to a different configuration; choose a new output directory."
            )
        if file_sha256(output / "manifest.json") != provenance.get("manifest_sha256"):
            raise ValueError(
                "Existing manifest differs from its provenance; choose a new output directory."
            )
        _, sequences = load_manifest(output / "manifest.json")
        if len(sequences) != len(config.videos):
            raise ValueError(
                "Existing import is incomplete; choose a new output directory."
            )
        suffix = ".npz" if compress_events else ".npy"
        if any(
            frame.events.suffix != suffix
            for sequence in sequences
            for frame in sequence.frames
        ):
            raise ValueError(
                "Existing import uses different event storage. Match its format with "
                "--compress-events or --no-compress-events, or choose a new --output directory."
            )
        for sequence in sequences:
            for frame in sequence.frames:
                read_frame(frame)
        return {
            **plan,
            "output": str(output),
            "status": "already-imported-and-validated",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        staged = Path(temporary)
        sequences, records = [], []
        for video in config.videos:
            archive = Path(
                hf_hub_download(
                    repo_id=config.repo_id,
                    repo_type="dataset",
                    revision=config.revision,
                    filename=video.archive,
                    cache_dir=cache_dir,
                )
            )
            if (
                archive.stat().st_size != video.size_bytes
                or file_sha256(archive) != video.sha256
            ):
                raise ValueError(f"Archive size/checksum mismatch: {video.archive}")
            sequence, record = _import_video(
                archive, video, config, staged, compress_events=compress_events
            )
            sequences.append(sequence)
            records.append(record)
        manifest = {"format_version": 1, "classes": ["drone"], "sequences": sequences}
        atomic_write_text(
            staged / "manifest.json", json.dumps(manifest, indent=2) + "\n"
        )
        _, validated = load_manifest(staged / "manifest.json")
        # Every image/map was decoded above; verify the public reader contract
        # on each sequence boundary before publishing this complete import.
        for sequence in validated:
            for frame in (sequence.frames[0], sequence.frames[-1]):
                read_frame(frame)
        provenance = {
            "format_version": 1,
            "importer_version": 2,
            "event_storage": event_storage,
            "environment": {
                "python": platform.python_version(),
                **{
                    name: version(name)
                    for name in ("huggingface-hub", "numpy", "pillow")
                },
            },
            "repo_id": config.repo_id,
            "revision": config.revision,
            "config_sha256": config.content_hash,
            "config": config.source,
            "videos": records,
            "manifest_sha256": file_sha256(staged / "manifest.json"),
            "assistance": "Codex",
        }
        atomic_write_text(
            staged / "provenance.json", json.dumps(provenance, indent=2) + "\n"
        )
        shutil.copy2(config_path, staged / "download-config.json")
        staged.rename(output)
    return {
        **plan,
        "output": str(output),
        "manifest": str(output / "manifest.json"),
        "status": "imported",
        "videos": records,
    }
