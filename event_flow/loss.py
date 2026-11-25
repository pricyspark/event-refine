"""Matched focal/L1/GIoU detection supervision, developed with Codex assistance."""

import torch
import torch.nn.functional as F

from flowsis.matching import (
    center_to_corners,
    match_object_queries,
    pairwise_generalized_iou,
)


def detection_loss(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Single image; targets are normalized xyxy in the padded input plane."""
    if (
        logits.ndim != 3
        or logits.shape[0] != 1
        or boxes.shape != (*logits.shape[:2], 4)
    ):
        raise ValueError(
            "Expected one image of aligned [1,Q,C] logits and [1,Q,4] boxes."
        )
    if target_boxes.shape != (target_labels.numel(), 4) or target_labels.ndim != 1:
        raise ValueError("Target boxes/labels do not align.")
    if target_labels.numel() > boxes.shape[1]:
        raise ValueError("More ground-truth objects than detector queries.")
    matches = match_object_queries(
        logits,
        boxes,
        torch.zeros_like(target_labels),
        target_labels,
        target_boxes,
    )
    targets = torch.zeros_like(logits)
    targets[0, matches, target_labels] = 1
    probability = logits.sigmoid()
    pt = probability * targets + (1 - probability) * (1 - targets)
    alpha = 0.25 * targets + 0.75 * (1 - targets)
    normalizer = max(1, target_labels.numel())
    classification = (
        F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        * alpha
        * (1 - pt).square()
    ).sum() / normalizer
    predicted_corners = center_to_corners(boxes[0, matches])
    if target_labels.numel():
        bbox = F.l1_loss(predicted_corners, target_boxes, reduction="sum") / normalizer
        giou = (
            1 - pairwise_generalized_iou(predicted_corners, target_boxes).diag()
        ).mean()
    else:
        bbox = boxes.sum() * 0
        giou = bbox
    return {
        "loss": classification + 5 * bbox + 2 * giou,
        "classification": classification,
        "bbox": bbox,
        "giou": giou,
    }


def normalized_targets(
    boxes: torch.Tensor, *, original_size: tuple[int, int], image_size: int
) -> torch.Tensor:
    height, width = original_size
    scale = image_size / max(height, width)
    x_scale = max(1, round(width * scale)) / width / image_size
    y_scale = max(1, round(height * scale)) / height / image_size
    return boxes * boxes.new_tensor([x_scale, y_scale, x_scale, y_scale])
