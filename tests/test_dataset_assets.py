"""Portable dataset and mask-area checks."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image

from flowsis.cli.common import load_dataset_dict
from flowsis.cli.train.train_base import build_dataloader
from flowsis.data.augment.basic import center_square_augment
from flowsis.data.augment.overlap import _verify_overlap_example
from flowsis.data.augment.rotate import rotation_augment
from flowsis.data.loaders import load_object_image, load_object_masks
from flowsis.data.masks import save_mask_bundle


def sample():
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    mask[6:10, 6:10] = False
    return {
        "image": Image.new("RGB", (16, 16)),
        "width": 16,
        "height": 16,
        "modified": False,
        "objects": [
            {"mask": mask, "bbox": [4, 4, 8, 8], "area": 48, "modified": False}
        ],
    }


def test_assets_resolve_after_moving_repository_and_changing_cwd(tmp_path, monkeypatch):
    root = tmp_path / "original"
    root.mkdir()
    Image.new("RGB", (16, 16)).save(root / "image.png")
    save_mask_bundle(root / "mask.npz", sample()["objects"][0]["mask"][None], ["a"])
    row = {
        "image_id": "source-image",
        "image_path": "../image.png",
        "video_path": "",
        "objects": [{"mask_path": "../mask.npz", "mask_index": 0, "instance_id": "a"}],
    }
    DatasetDict(train=Dataset.from_list([row])).save_to_disk(root / "dataset")
    (root / "dataset/asset_paths.json").write_text(
        json.dumps({"version": 1, "relative_to": "dataset_directory"})
    )
    moved = tmp_path / "moved"
    root.rename(moved)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    loaded = load_dataset_dict(moved / "dataset")
    result = load_object_masks(load_object_image(loaded["train"][0]))
    assert result["image"].size == (16, 16)
    assert result["modified"] is False
    assert result["objects"][0]["modified"] is False
    assert result["objects"][0]["source_image_id"] == "source-image"
    assert result["objects"][0]["instance_id"] == "a"
    assert result["objects"][0]["mask"].sum() == 48
    stored = load_from_disk(moved / "dataset")["train"][0]
    assert stored["image_path"] == "../image.png"
    assert "modified" not in stored
    assert "modified" not in stored["objects"][0]
    assert stored["objects"][0]["mask_path"] == "../mask.npz"


def test_legacy_paths_are_unchanged_without_contract(tmp_path):
    DatasetDict(
        train=Dataset.from_list([{"image_path": "legacy-relative.jpg"}])
    ).save_to_disk(tmp_path / "dataset")
    assert (
        load_dataset_dict(tmp_path / "dataset")["train"][0]["image_path"]
        == "legacy-relative.jpg"
    )


@pytest.mark.parametrize(
    "augment,kwargs",
    [
        (center_square_augment, {"crop_size": 8}),
        (rotation_augment, {"min_angle": 30, "max_angle": 30, "seed": 1}),
    ],
)
def test_augmentation_area_counts_mask_pixels(augment, kwargs):
    result = augment(sample(), **kwargs)
    for obj in result["objects"]:
        assert obj["area"] == np.count_nonzero(obj["mask"])
        assert obj["area"] < obj["bbox"][2] * obj["bbox"][3]


def test_overlap_area_counts_visible_mask_pixels():
    example = sample()
    obj = example["objects"][0]
    full = obj["mask"].copy()
    visible = full.copy()
    visible[:7] = False
    obj["visible_mask"] = visible
    assert _verify_overlap_example(
        example, 1, masks=full[None], visible_masks=visible[None]
    )
    assert obj["area"] == np.count_nonzero(visible)


def test_sampler_balances_sample_ids_across_single_and_multi_tool_images():
    rows = [
        {"sample_id": sample_id, "tool_ids": tools}
        for sample_id, tools in (
            ("single", ["12"]),
            ("single", ["12"]),
            ("multi", ["12", "18"]),
            ("multi", ["12", "18"]),
            ("multi", ["12", "18"]),
            ("", ["20"]),
            ("", []),
        )
    ]
    dataset = Dataset.from_list(rows)
    args = SimpleNamespace(
        seed=42, batch_size=2, num_workers=0, image_size=16, device="cpu"
    )
    loader = build_dataloader(dataset, args, training=True)
    weights = loader.sampler.weights.tolist()
    assert weights == pytest.approx([0.5, 0.5, 1 / 3, 1 / 3, 1 / 3, 1, 1])
    assert sum(weights[:2]) == pytest.approx(sum(weights[2:5]))
    validation = build_dataloader(dataset, args, training=False)
    assert not hasattr(validation.sampler, "weights")


def test_runtime_initialization_preserves_modified_guards():
    with pytest.raises(ValueError, match="previously modified sample"):
        load_object_image({"modified": True, "image_path": "must-not-open.jpg"})
    with pytest.raises(ValueError, match="previously modified object"):
        load_object_masks(
            {"objects": [{"modified": True, "mask_path": "must-not-open.npz"}]}
        )
