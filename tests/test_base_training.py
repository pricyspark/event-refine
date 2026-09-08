import numpy as np
import pytest
import torch
from PIL import Image
from pathlib import Path

from flowsis.cli.train.train_base import (
    collate_examples,
    resolve_warmup_steps,
    training_epoch_summary,
)
from flowsis.data.loaders import relocate_segmentation_paths
from flowsis.matching import segmentation_objective


def example(objects: list[dict[str, object]]) -> dict[str, object]:
    return {
        "image": Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)),
        "image_id": 1,
        "height": 16,
        "width": 16,
        "objects": objects,
    }


def object_record(category: int, x: int) -> dict[str, object]:
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, x : x + 4] = True
    return {
        "category": category,
        "bbox": [float(x), 4.0, 4.0, 8.0],
        "area": 32.0,
        "mask": mask,
    }


def test_collate_supports_multiple_instances_and_empty_images() -> None:
    batch = collate_examples(
        [example([object_record(0, 2), object_record(0, 10)]), example([])],
        image_size=16,
    )
    assert batch["batch_size"] == 2
    assert batch["target_masks"].shape == (2, 16, 16)
    assert batch["object_image_indices"].tolist() == [0, 0]
    assert batch["annotations"][1]["annotations"] == []


def test_all_empty_batch_has_canonical_empty_tensors() -> None:
    batch = collate_examples([example([]), example([])], image_size=16)
    assert batch["target_masks"].shape == (0, 16, 16)
    assert batch["object_boxes"].shape == (0, 4)
    assert batch["object_labels"].shape == (0,)


def test_object_then_image_loss_reduction_balances_crowding() -> None:
    logits = torch.zeros(3, 2, 2, requires_grad=True)
    targets = torch.stack((torch.zeros(2, 2), torch.ones(2, 2), torch.ones(2, 2)))
    result = segmentation_objective(
        logits,
        targets,
        torch.tensor([0, 1, 1]),
        batch_size=2,
        bce_weight=1.0,
        dice_weight=0.0,
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    torch.testing.assert_close(result["loss"], expected)


def test_segmentation_loss_ignores_marked_pixels() -> None:
    logits = torch.tensor([[[20.0, -20.0]]], requires_grad=True)
    targets = torch.zeros_like(logits)
    result = segmentation_objective(
        logits,
        targets,
        torch.tensor([0]),
        batch_size=1,
        valid_pixels=torch.tensor([[[False, True]]]),
        bce_weight=1.0,
        dice_weight=0.0,
    )

    assert float(result["loss"].detach()) < 1e-6


def test_serialized_paths_relocate_between_machines(tmp_path: Path) -> None:
    images_dir = tmp_path / "frames"
    masks_dir = tmp_path / "masks"
    (images_dir / "24").mkdir(parents=True)
    (masks_dir / "24").mkdir(parents=True)
    image_path = images_dir / "24" / "frame-000266_cluster-0038.jpg"
    mask_path = masks_dir / "24" / "266.npz"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    mask_path.write_bytes(b"mask-placeholder")
    row = {
        "image_path": "/another/computer/frames/24/frame-000266_cluster-0038.jpg",
        "objects": [
            {
                "video_id": 24,
                "frame_idx": 266,
                "mask_path": "/another/computer/masks/24/266.npz",
            }
        ],
    }
    relocated = relocate_segmentation_paths(
        row, images_dir=images_dir, masks_dir=masks_dir
    )
    assert relocated["image_path"] == str(image_path)
    assert relocated["objects"][0]["mask_path"] == str(mask_path)


def test_warmup_ratio_scales_with_experiment_length() -> None:
    assert (
        resolve_warmup_steps(
            warmup_steps=0,
            warmup_ratio=0.05,
            total_steps=101,
        )
        == 5
    )


def test_warmup_ratio_and_steps_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either"):
        resolve_warmup_steps(
            warmup_steps=10,
            warmup_ratio=0.05,
            total_steps=100,
        )


def test_training_epoch_summary_reports_throughput_and_stability() -> None:
    summary = training_epoch_summary(
        loss_sum=6.0,
        loss_square_sum=14.0,
        batches=3,
        images=12,
        seconds=2.0,
        grad_norm_sum=9.0,
        max_grad_norm=5.0,
        skipped_optimizer_steps=1,
        peak_memory_mb=1024.0,
    )

    assert summary == {
        "train_loss": 2.0,
        "train_loss_std": pytest.approx((2.0 / 3.0) ** 0.5),
        "train_batches": 3,
        "train_images": 12,
        "train_images_per_second": 6.0,
        "train_batches_per_second": 1.5,
        "mean_grad_norm": 3.0,
        "max_grad_norm": 5.0,
        "skipped_optimizer_steps": 1,
        "peak_memory_mb": 1024.0,
    }
