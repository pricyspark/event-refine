from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import Dataset, Sampler

from flowsis.data.masks import MaskBundle, load_mask_bundle


@dataclass(frozen=True)
class ClipSpec:
    video_id: str
    start: int
    length: int
    stride: int

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(self.start + offset * self.stride for offset in range(self.length))


class DynamicClipBatchSampler(Sampler[list[ClipSpec]]):
    """Sample one dynamic temporal geometry per batch.

    Length and stride are shared within a batch, so ordinary tensor collation
    stays efficient. Videos are sampled uniformly rather than in proportion to
    their frame count.
    """

    def __init__(
        self,
        frame_indices: Mapping[str, Sequence[int]],
        *,
        batch_size: int,
        batches_per_epoch: int,
        min_length: int = 4,
        max_length: int = 8,
        strides: Sequence[int] = (1, 2, 3),
        stride_weights: Sequence[float] = (0.7, 0.2, 0.1),
        positive_indices: Mapping[str, Sequence[int]] | None = None,
        sequence_ids: Mapping[str, Mapping[int, str]] | None = None,
        evaluation: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0 or batches_per_epoch <= 0:
            raise ValueError("Batch size and batches per epoch must be positive.")
        if min_length < 2 or max_length < min_length:
            raise ValueError("Clip lengths must satisfy 2 <= minimum <= maximum.")
        if len(strides) != len(stride_weights) or not strides:
            raise ValueError("Strides and weights must be nonempty and aligned.")
        if any(stride <= 0 for stride in strides):
            raise ValueError("Clip strides must be positive.")
        if any(weight < 0 for weight in stride_weights) or sum(stride_weights) <= 0:
            raise ValueError("Stride weights must be nonnegative with positive sum.")
        self.frame_indices = {
            video_id: frozenset(int(index) for index in indices)
            for video_id, indices in frame_indices.items()
        }
        self.positive_indices = {
            video_id: frozenset(int(index) for index in indices)
            for video_id, indices in (
                frame_indices if positive_indices is None else positive_indices
            ).items()
        }
        self.sequence_ids = sequence_ids or {}
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.min_length = min_length
        self.max_length = max_length
        self.strides = tuple(int(value) for value in strides)
        self.stride_weights = tuple(float(value) for value in stride_weights)
        self.evaluation = evaluation
        self.seed = seed
        self.epoch = 0
        self.geometries = [
            (length, stride, weight, eligible)
            for length in range(self.min_length, self.max_length + 1)
            for stride, weight in zip(self.strides, self.stride_weights, strict=True)
            if weight > 0 and (eligible := self._eligible(length, stride))
        ]
        if not self.geometries:
            raise ValueError(
                "No video interval has enough reviewed frames and a visible mask for a configured clip."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _eligible(self, length: int, stride: int) -> dict[str, list[int]]:
        offsets = tuple(offset * stride for offset in range(length))
        eligible: dict[str, list[int]] = {}
        for video_id, indices in self.frame_indices.items():
            starts = [
                index
                for index in sorted(indices)
                if all(
                    value in indices for value in range(index, index + offsets[-1] + 1)
                )
                and len(
                    {
                        self.sequence_ids.get(video_id, {}).get(value, video_id)
                        for value in range(index, index + offsets[-1] + 1)
                    }
                )
                == 1
                and any(
                    index + offset in self.positive_indices.get(video_id, ())
                    for offset in offsets
                )
            ]
            if starts:
                eligible[video_id] = starts
        return eligible

    def __iter__(self) -> Iterator[list[ClipSpec]]:
        generator = random.Random(self.seed + self.epoch)
        for batch_index in range(self.batches_per_epoch):
            if self.evaluation:
                length, stride, _weight, eligible = self.geometries[
                    batch_index % len(self.geometries)
                ]
                video_ids = tuple(sorted(eligible))
                video_position = batch_index % len(video_ids)
                video_id = video_ids[video_position]
                starts = eligible[video_id]
                draws = (
                    (self.batches_per_epoch - 1 - video_position) // len(video_ids) + 1
                ) * self.batch_size
                offset = (batch_index // len(video_ids)) * self.batch_size
                yield [
                    ClipSpec(
                        video_id,
                        starts[(offset + index) * len(starts) // draws],
                        length,
                        stride,
                    )
                    for index in range(self.batch_size)
                ]
                continue
            length, stride, _weight, eligible = generator.choices(
                self.geometries, weights=[item[2] for item in self.geometries], k=1
            )[0]
            video_ids = tuple(eligible)
            yield [
                ClipSpec(
                    video_id=(video_id := generator.choice(video_ids)),
                    start=generator.choice(eligible[video_id]),
                    length=length,
                    stride=stride,
                )
                for _ in range(self.batch_size)
            ]


def _read_video_frames(path: Path, indices: Sequence[int]) -> list[NDArray[np.uint8]]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    frames: list[NDArray[np.uint8]] = []
    try:
        requested = set(indices)
        capture.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        for index in range(indices[0], indices[-1] + 1):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {index} from {path}.")
            if index in requested:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def _crop_geometry(height: int, width: int) -> tuple[int, int, int]:
    side = min(height, width)
    return (height - side) // 2, (width - side) // 2, side


def _transform_box(
    box: Sequence[float], top: int, left: int, side: int, flip: bool
) -> torch.Tensor:
    x, y, width, height = (float(value) for value in box)
    transformed = torch.tensor(
        (
            (x - left) / side,
            (y - top) / side,
            (x + width - left) / side,
            (y + height - top) / side,
        ),
        dtype=torch.float32,
    ).clamp_(0.0, 1.0)
    if flip:
        x1, x2 = transformed[0].clone(), transformed[2].clone()
        transformed[0], transformed[2] = 1.0 - x2, 1.0 - x1
    return transformed


class TemporalClipDataset(Dataset[dict[str, Any]]):
    """Decode a ``ClipSpec`` from a frame-level Arrow dataset."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        image_size: int,
        training: bool,
        frame_loader: Any = _read_video_frames,
    ) -> None:
        if image_size <= 0:
            raise ValueError("Image size must be positive.")
        self.image_size = image_size
        self.training = training
        self.frame_loader = frame_loader
        self.rows: dict[tuple[str, int], Mapping[str, Any]] = {}
        self.video_paths: dict[str, Path] = {}
        self.frame_indices: dict[str, list[int]] = defaultdict(list)
        self.sequence_ids: dict[str, dict[int, str]] = defaultdict(dict)
        self.positive_indices: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            video_id = str(row["video_id"])
            frame_index = int(row["frame_index"])
            key = (video_id, frame_index)
            if key in self.rows:
                raise ValueError(f"Duplicate video frame row: {key}")
            path = Path(str(row["video_path"])).expanduser()
            previous = self.video_paths.setdefault(video_id, path)
            if previous != path:
                raise ValueError(f"Video {video_id} maps to multiple paths.")
            self.rows[key] = row
            self.frame_indices[video_id].append(frame_index)
            self.sequence_ids[video_id][frame_index] = str(
                row.get("sequence_id") or video_id
            )
            if any(obj.get("mask_valid", True) for obj in row.get("objects", [])):
                self.positive_indices[video_id].append(frame_index)
        if not self.rows:
            raise ValueError("Temporal dataset contains no reviewed frame rows.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, spec: ClipSpec) -> dict[str, Any]:  # type: ignore[override]
        indices = spec.frame_indices
        rows = [self.rows[(spec.video_id, index)] for index in indices]
        if len(
            {
                self.sequence_ids[spec.video_id][index]
                for index in range(indices[0], indices[-1] + 1)
                if index in self.sequence_ids[spec.video_id]
            }
        ) != 1 or any(
            (spec.video_id, index) not in self.rows
            for index in range(indices[0], indices[-1] + 1)
        ):
            raise ValueError("Clip crosses an interval boundary or missing frame.")
        if all(row.get("image_path") for row in rows):
            raw_frames = []
            for row in rows:
                with Image.open(str(row["image_path"])) as image:
                    raw_frames.append(np.array(image.convert("RGB")))
        else:
            raw_frames = self.frame_loader(self.video_paths[spec.video_id], indices)
        if len(raw_frames) != spec.length:
            raise RuntimeError("Frame loader returned the wrong number of frames.")
        height, width = raw_frames[0].shape[:2]
        top, left, side = _crop_geometry(height, width)
        flip = self.training and bool(torch.rand(()) < 0.5)
        frames = []
        for frame in raw_frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("All frames in a clip must have the same dimensions.")
            cropped = np.ascontiguousarray(frame[top : top + side, left : left + side])
            tensor = torch.from_numpy(cropped).permute(2, 0, 1).float().div_(255.0)
            tensor = F.interpolate(
                tensor[None],
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )[0]
            frames.append(tensor.flip(-1) if flip else tensor)

        track_ids = tuple(
            sorted(
                {str(obj["track_id"]) for row in rows for obj in row.get("objects", [])}
            )
        )
        track_index = {track_id: index for index, track_id in enumerate(track_ids)}
        track_count = len(track_ids)
        masks = torch.zeros(
            (spec.length, track_count, self.image_size, self.image_size),
            dtype=torch.float32,
        )
        boxes = torch.zeros((spec.length, track_count, 4), dtype=torch.float32)
        labels = torch.full((spec.length, track_count), -1, dtype=torch.long)
        mask_valid = torch.zeros((spec.length, track_count), dtype=torch.bool)
        class_valid = torch.zeros_like(mask_valid)
        base_valid = torch.zeros_like(mask_valid)
        visible = torch.zeros_like(mask_valid)
        confidence = torch.ones((spec.length, track_count), dtype=torch.float32)
        preferred = torch.zeros_like(mask_valid)
        ignore = torch.zeros(
            (spec.length, self.image_size, self.image_size), dtype=torch.bool
        )
        bundle_cache: dict[Path, MaskBundle] = {}

        for timestamp, row in enumerate(rows):
            bundle_path = Path(str(row.get("mask_bundle_path", ""))).expanduser()
            if bundle_path.is_file():
                if bundle_path not in bundle_cache:
                    bundle_cache[bundle_path] = load_mask_bundle(bundle_path)
                bundle = bundle_cache[bundle_path]
                if bundle.ignore is not None:
                    cropped_ignore = np.ascontiguousarray(
                        bundle.ignore[top : top + side, left : left + side]
                    )
                    ignore_tensor = F.interpolate(
                        torch.from_numpy(cropped_ignore)[None, None].float(),
                        size=(self.image_size, self.image_size),
                        mode="nearest",
                    )[0, 0].bool()
                    ignore[timestamp] = (
                        ignore_tensor.flip(-1) if flip else ignore_tensor
                    )
            for obj in row.get("objects", []):
                index = track_index[str(obj["track_id"])]
                labels[timestamp, index] = int(obj.get("category", -1))
                mask_valid[timestamp, index] = bool(obj.get("mask_valid", False))
                class_valid[timestamp, index] = bool(obj.get("class_valid", False))
                base_valid[timestamp, index] = bool(obj.get("base_valid", False))
                visible[timestamp, index] = bool(obj.get("visible", False))
                preferred[timestamp, index] = bool(obj.get("preferred_track", False))
                confidence[timestamp, index] = float(
                    obj.get("annotation_confidence", 1.0)
                )
                boxes[timestamp, index] = _transform_box(
                    obj.get("bbox", (0, 0, 0, 0)), top, left, side, flip
                )
                if not mask_valid[timestamp, index]:
                    continue
                path = Path(str(obj["mask_path"])).expanduser()
                if path not in bundle_cache:
                    bundle_cache[path] = load_mask_bundle(path)
                bundle = bundle_cache[path]
                mask = bundle.masks[int(obj["mask_index"])]
                cropped = np.ascontiguousarray(
                    mask[top : top + side, left : left + side]
                )
                tensor = F.interpolate(
                    torch.from_numpy(cropped)[None, None].float(),
                    size=(self.image_size, self.image_size),
                    mode="nearest",
                )[0, 0]
                masks[timestamp, index] = tensor.flip(-1) if flip else tensor

        return {
            "frames": torch.stack(frames),
            "masks": masks,
            "boxes": boxes,
            "labels": labels,
            "mask_valid": mask_valid,
            "class_valid": class_valid,
            "base_valid": base_valid,
            "visible": visible,
            "annotation_confidence": confidence,
            "preferred_track": preferred,
            "ignore": ignore,
            "track_ids": track_ids,
            "video_id": spec.video_id,
            "frame_indices": torch.tensor(indices, dtype=torch.long),
            "timestamp_seconds": torch.tensor(
                [
                    float(
                        row.get("timestamp_seconds", index / float(row.get("fps") or 1))
                    )
                    for row, index in zip(rows, indices, strict=True)
                ],
                dtype=torch.float64,
            ),
            "frame_delta": torch.full((spec.length,), spec.stride, dtype=torch.long),
        }


def collate_temporal_clips(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty temporal batch.")
    max_tracks = max(item["masks"].shape[1] for item in batch)

    def pad_tracks(tensor: torch.Tensor, value: float | int | bool) -> torch.Tensor:
        missing = max_tracks - tensor.shape[1]
        if not missing:
            return tensor
        shape = list(tensor.shape)
        shape[1] = missing
        padding = torch.full(shape, value, dtype=tensor.dtype)
        return torch.cat((tensor, padding), dim=1)

    return {
        "frames": torch.stack([item["frames"] for item in batch]),
        "masks": torch.stack([pad_tracks(item["masks"], 0.0) for item in batch]),
        "boxes": torch.stack([pad_tracks(item["boxes"], 0.0) for item in batch]),
        "labels": torch.stack([pad_tracks(item["labels"], -1) for item in batch]),
        **{
            key: torch.stack([pad_tracks(item[key], False) for item in batch])
            for key in (
                "mask_valid",
                "class_valid",
                "base_valid",
                "visible",
                "preferred_track",
            )
        },
        "annotation_confidence": torch.stack(
            [pad_tracks(item["annotation_confidence"], 0.0) for item in batch]
        ),
        "ignore": torch.stack([item["ignore"] for item in batch]),
        "frame_indices": torch.stack([item["frame_indices"] for item in batch]),
        "frame_delta": torch.stack([item["frame_delta"] for item in batch]),
        "timestamp_seconds": torch.stack([item["timestamp_seconds"] for item in batch]),
        "track_ids": [item["track_ids"] for item in batch],
        "video_id": [item["video_id"] for item in batch],
    }


__all__ = [
    "ClipSpec",
    "DynamicClipBatchSampler",
    "TemporalClipDataset",
    "collate_temporal_clips",
]
