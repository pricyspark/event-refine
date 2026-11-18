"""Collected-data augmentation regressions."""

import copy
import sys

import numpy as np
import pytest
from PIL import Image

from flowsis.cli.train import (
    train_base,
    train_detector,
    visualize_detector_augmentations,
)
from flowsis.data.augment.overlap import overlap_augment
from flowsis.data.augment.rotate import (
    rotate_crop_bounds,
    rotation_augment,
    sample_rotation_angle,
)


def example():
    objects = []
    for left in (2, 10):
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:12, left : left + 4] = True
        objects.append(
            {"mask": mask, "bbox": [left, 4, 4, 8], "area": 32, "modified": False}
        )
    return {
        "image": Image.new("RGB", (16, 16)),
        "height": 16,
        "width": 16,
        "objects": objects,
        "modified": False,
    }


@pytest.mark.parametrize(
    "module", [train_base, train_detector, visualize_detector_augmentations]
)
def test_collected_data_cli_defaults(module, monkeypatch):
    argv = ["train"]
    if module is train_base:
        argv += ["--text-embeddings-dir", "unused-in-parser-test"]
    monkeypatch.setattr(sys, "argv", argv)
    args = module.parse_args()
    assert (args.rotation_min_degrees, args.rotation_max_degrees) == (-22.5, 22.5)
    assert args.use_overlap_augment
    assert (args.overlap_min_overlays, args.overlap_max_overlays, args.overlap_p) == (
        0,
        1,
        0.25,
    )


def test_sampled_angles_and_fallback_respect_range():
    rng = np.random.default_rng(42)
    angles = [sample_rotation_angle(2160, 2160, rng)[0] for _ in range(500)]
    assert all(-22.5 <= angle <= 22.5 for angle in angles)
    assert min(angles) < 0 < max(angles)
    assert (
        sample_rotation_angle(16, 12, rng, min_angle=10, max_angle=20, max_tries=0)[0]
        == 10
    )
    with pytest.raises(ValueError, match="Rotation angles"):
        sample_rotation_angle(16, 16, rng, min_angle=45, max_angle=0)


def test_zero_and_45_degree_bounds_are_finite():
    assert rotate_crop_bounds(16, 12, 0) == (16, 12, 0, 0)
    bounds = rotate_crop_bounds(16, 16, np.pi / 4)
    assert bounds[:2] == pytest.approx((16 / np.sqrt(2),) * 2)
    for angle in (-22.5, 0, 22.5, 45):
        result = rotation_augment(example(), seed=42, min_angle=angle, max_angle=angle)
        assert len(result["objects"]) == 2
        assert not np.logical_and(*(obj["mask"] for obj in result["objects"])).any()


def test_zero_overlay_does_not_sample_or_mark_real_objects_synthetic():
    class UnusedContext:
        def sample_examples(self, count, **kwargs):
            raise AssertionError("Zero overlays must not sample other images")

    result = overlap_augment(
        example(),
        context=UnusedContext(),
        min_overlays=0,
        max_overlays=1,
        p=0.25,
        seed=0,
    )
    assert all(
        not obj["modified"] and "synthetic_occlusion" not in obj
        for obj in result["objects"]
    )


def test_overlay_probability_and_multi_tool_disjointness():
    class Context:
        def __init__(self):
            self.overlays = 0

        def sample_examples(self, count, **kwargs):
            self.overlays += count
            return [copy.deepcopy(example()) for _ in range(count)]

    context = Context()
    for seed in range(300):
        result = overlap_augment(
            example(),
            context=context,
            min_overlays=0,
            max_overlays=1,
            p=0.25,
            seed=seed,
        )
        masks = np.stack([obj["mask"] for obj in result["objects"]])
        assert np.max(masks.sum(axis=0)) == 1
    assert 45 < context.overlays < 105
