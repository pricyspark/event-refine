"""Shared-dataset views, sampling, and saved-frame temporal loading."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from PIL import Image

from flowsis.cli.common import dataset_from_args
from flowsis.cli.train.train_base import build_dataloader
from flowsis.data.loaders import (
    base_dataset_view,
    load_local_dataset,
    temporal_dataset_view,
)
from flowsis.data.masks import save_mask_bundle
from flowsis.data.sampling import frame_sampling_weights
from flowsis.data.temporal import (
    ClipSpec,
    DynamicClipBatchSampler,
    TemporalClipDataset,
    collate_temporal_clips,
)


def test_dynamic_sampling_balances_sources_then_groups():
    rows = [
        {"video_id": video, "sample_id": sample}
        for video, sample, count in [
            ("", "a", 2),
            ("", "b", 3),
            ("v1", "", 30),
            ("v2", "", 10),
        ]
        for _ in range(count)
    ]
    data = Dataset.from_list(rows)
    weights = frame_sampling_weights(data, video_fraction=0.4)
    assert sum(weights[:5]) == pytest.approx(0.6)
    assert sum(weights[5:]) == pytest.approx(0.4)
    assert sum(weights[:2]) == pytest.approx(0.3)
    assert sum(weights[5:35]) == pytest.approx(0.2)
    args = SimpleNamespace(
        seed=42,
        batch_size=1,
        num_workers=0,
        image_size=8,
        device="cpu",
        video_fraction=0.4,
        samples_per_epoch=10000,
    )
    sampler = build_dataloader(data, args, training=True).sampler
    draws = list(sampler)
    assert len(draws) == 10000
    assert 0.38 < sum(index >= 5 for index in draws) / len(draws) < 0.42
    assert list(sampler) != draws  # Generator advances between epochs.
    for bad in [-0.1, 1.1, float("nan")]:
        with pytest.raises(ValueError):
            frame_sampling_weights(data, video_fraction=bad)
    assert all(
        weight > 0
        for weight in frame_sampling_weights(data.select(range(5)), video_fraction=1)
    )


def test_views_keep_split_membership_and_class_order(tmp_path):
    features = Features(
        {
            "video_id": Value("string"),
            "base_eligible": Value("bool"),
            "category": ClassLabel(names=["iris scissors", "other scissors"]),
        }
    )
    split = Dataset.from_list(
        [
            {"video_id": "", "base_eligible": True, "category": 0},
            {"video_id": "v", "base_eligible": False, "category": 1},
            {"video_id": "v", "base_eligible": True, "category": 0},
        ],
        features=features,
    )
    data = DatasetDict(train=split, validation=split.select([1]))
    data.save_to_disk(tmp_path / "dataset")
    loaded = dataset_from_args(SimpleNamespace(dataset_path=tmp_path / "dataset"))
    assert len(loaded["train"]) == 2
    assert len(loaded["validation"]) == 0
    assert len(temporal_dataset_view(split)) == 2
    assert base_dataset_view(split).features["category"].names == [
        "iris scissors",
        "other scissors",
    ]


def test_saved_png_clips_resolve_relative_assets_and_preserve_time(tmp_path):
    mask = np.zeros((8, 10), bool)
    mask[2:6, 3:7] = True
    save_mask_bundle(tmp_path / "mask.npz", mask[None], ["tool-1"])
    rows = []
    for index in range(3):
        Image.new("RGB", (10, 8), (index * 50, 0, 0)).save(tmp_path / f"{index}.png")
        rows.append(
            {
                "video_id": "v",
                "video_path": "",
                "frame_index": index,
                "image_path": f"../{index}.png",
                "sequence_id": "run",
                "timestamp_seconds": [0, 0.12, 0.31][index],
                "mask_bundle_path": "../mask.npz",
                "objects": [
                    {
                        "track_id": "tool-1",
                        "mask_valid": True,
                        "class_valid": True,
                        "base_valid": True,
                        "visible": True,
                        "category": 0,
                        "bbox": [3, 2, 4, 4],
                        "mask_path": "../mask.npz",
                        "mask_index": 0,
                    }
                ],
            }
        )
    DatasetDict(train=Dataset.from_list(rows)).save_to_disk(tmp_path / "dataset")
    (tmp_path / "dataset/asset_paths.json").write_text(
        json.dumps({"version": 1, "relative_to": "dataset_directory"})
    )
    split = load_local_dataset(tmp_path / "dataset")["train"]
    assert split[0]["mask_bundle_path"] == str(tmp_path / "mask.npz")

    def no_mp4(*args):
        raise AssertionError("Saved frames must not open the MP4")

    dataset = TemporalClipDataset(
        split, image_size=8, training=False, frame_loader=no_mp4
    )
    clip = dataset[ClipSpec("v", 0, 3, 1)]
    assert clip["timestamp_seconds"].tolist() == [0, 0.12, 0.31]
    assert clip["frames"].shape == (3, 3, 8, 8)
    assert clip["frames"][2, 0].mean().item() == pytest.approx(100 / 255)
    assert collate_temporal_clips([clip])["timestamp_seconds"].shape == (1, 3)


def test_sampler_cannot_jump_boundaries_or_unaccepted_frames():
    sampler = DynamicClipBatchSampler(
        {"v": range(20)},
        sequence_ids={"v": {i: "a" if i < 10 else "b" for i in range(20)}},
        batch_size=4,
        batches_per_epoch=20,
        min_length=2,
        max_length=4,
        strides=(3,),
        stride_weights=(1,),
    )
    for batch in sampler:
        assert all(spec.frame_indices[-1] < 10 or spec.start >= 10 for spec in batch)
    with pytest.raises(ValueError, match="No video interval"):
        DynamicClipBatchSampler(
            {"v": [0, 2, 4, 6]},
            batch_size=1,
            batches_per_epoch=1,
            min_length=4,
            max_length=4,
            strides=(2,),
            stride_weights=(1,),
        )
    with pytest.raises(ValueError, match="visible mask"):
        DynamicClipBatchSampler(
            {"v": range(20)},
            positive_indices={"v": []},
            batch_size=1,
            batches_per_epoch=1,
        )


def test_evaluation_schedule_is_fixed_and_covers_videos():
    sampler = DynamicClipBatchSampler(
        {"a": range(20), "b": range(10)},
        evaluation=True,
        batch_size=2,
        batches_per_epoch=4,
        min_length=4,
        max_length=4,
        strides=(1,),
        stride_weights=(1,),
    )
    first = list(sampler)
    sampler.set_epoch(30)
    assert list(sampler) == first
    assert [batch[0].video_id for batch in first] == ["a", "b", "a", "b"]
    assert all(len({spec.video_id for spec in batch}) == 1 for batch in first)
