from pathlib import Path

import numpy as np
import torch

from flowsis.data.masks import save_mask_bundle
from flowsis.data.temporal import (
    ClipSpec,
    DynamicClipBatchSampler,
    TemporalClipDataset,
    collate_temporal_clips,
)
from flowsis.cli.data.build_temporal_dataset import _split_videos


def test_dynamic_sampler_uses_shared_batch_geometry_and_dense_frames() -> None:
    sampler = DynamicClipBatchSampler(
        {"a": range(20), "b": range(10, 30)},
        positive_indices={"a": (5,), "b": (15,)},
        batch_size=4,
        batches_per_epoch=5,
        min_length=4,
        max_length=8,
        strides=(1, 2, 3),
        stride_weights=(0.7, 0.2, 0.1),
        seed=3,
    )

    for batch in sampler:
        assert len({spec.length for spec in batch}) == 1
        assert len({spec.stride for spec in batch}) == 1
        for spec in batch:
            available = set(range(20)) if spec.video_id == "a" else set(range(10, 30))
            assert set(spec.frame_indices) <= available


def test_clip_dataset_retains_track_identity_through_occlusion(tmp_path: Path) -> None:
    mask = np.zeros((8, 10), dtype=np.bool_)
    mask[2:6, 3:8] = True
    bundle = tmp_path / "0.npz"
    save_mask_bundle(bundle, mask[None], ["tool-1"])
    rows = [
        {
            "video_id": "v",
            "video_path": str(tmp_path / "v.mp4"),
            "frame_index": index,
            "objects": (
                [
                    {
                        "track_id": "tool-1",
                        "category": 0,
                        "bbox": [3, 2, 5, 4],
                        "mask_path": str(bundle),
                        "mask_index": 0,
                        "mask_valid": index == 0,
                        "class_valid": index == 0,
                        "base_valid": index == 0,
                        "visible": index == 0,
                    }
                ]
                if index in {0, 1}
                else []
            ),
        }
        for index in range(3)
    ]

    def frames(_path: Path, indices: tuple[int, ...]) -> list[np.ndarray]:
        return [np.full((8, 10, 3), index, dtype=np.uint8) for index in indices]

    dataset = TemporalClipDataset(
        rows, image_size=8, training=False, frame_loader=frames
    )
    item = dataset[ClipSpec("v", 0, 3, 1)]

    assert item["track_ids"] == ("tool-1",)
    assert item["visible"][:, 0].tolist() == [True, False, False]
    assert item["mask_valid"][:, 0].tolist() == [True, False, False]
    batch = collate_temporal_clips([item, item])
    assert batch["masks"].shape == (2, 3, 1, 8, 8)


def test_temporal_splits_keep_shared_sessions_and_tools_together() -> None:
    rows = [
        {
            "video_id": video,
            "capture_session_id": session,
            "objects": [{"physical_tool_id": tool}],
        }
        for video, session, tool in (
            ("a", "one", "x"),
            ("b", "one", "y"),
            ("c", "two", "z"),
            ("d", "three", "q"),
        )
    ]
    assignments = _split_videos(rows, validation_size=0.25, test_size=0.25, seed=2)

    assert assignments["a"] == assignments["b"]
