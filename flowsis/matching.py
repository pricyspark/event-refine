from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def corners_to_center(boxes: torch.Tensor) -> torch.Tensor:
    top_left, bottom_right = boxes.split(2, dim=-1)
    size = bottom_right - top_left
    return torch.cat((top_left + 0.5 * size, size), dim=-1)


def center_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    center, size = boxes.split(2, dim=-1)
    half_size = 0.5 * size
    return torch.cat((center - half_size, center + half_size), dim=-1)


def pairwise_generalized_iou(
    left_boxes: torch.Tensor,
    right_boxes: torch.Tensor,
) -> torch.Tensor:
    left_top_left, left_bottom_right = left_boxes[:, None].split(2, dim=-1)
    right_top_left, right_bottom_right = right_boxes[None].split(2, dim=-1)
    intersection_top_left = torch.maximum(left_top_left, right_top_left)
    intersection_bottom_right = torch.minimum(left_bottom_right, right_bottom_right)
    intersection_size = (intersection_bottom_right - intersection_top_left).clamp_min(
        0.0
    )
    intersection = intersection_size.prod(dim=-1)
    left_area = (left_bottom_right - left_top_left).clamp_min(0.0).prod(dim=-1)
    right_area = (right_bottom_right - right_top_left).clamp_min(0.0).prod(dim=-1)
    union = left_area + right_area - intersection
    iou = intersection / union.clamp_min(1e-7)
    enclosing_top_left = torch.minimum(left_top_left, right_top_left)
    enclosing_bottom_right = torch.maximum(left_bottom_right, right_bottom_right)
    enclosing_area = (enclosing_bottom_right - enclosing_top_left).prod(dim=-1)
    return iou - (enclosing_area - union) / enclosing_area.clamp_min(1e-7)


def match_object_queries(
    query_logits: torch.Tensor,
    query_boxes: torch.Tensor,
    object_image_indices: torch.Tensor,
    object_labels: torch.Tensor,
    object_boxes: torch.Tensor,
    *,
    class_weight: float = 2.0,
    bbox_weight: float = 5.0,
    giou_weight: float = 2.0,
) -> torch.Tensor:
    """Hungarian-match each target to one final detector query in its image."""
    if query_logits.ndim != 3 or query_boxes.ndim != 3:
        raise ValueError("Expected batched detector query logits and boxes.")
    if query_logits.shape[:2] != query_boxes.shape[:2]:
        raise ValueError("Detector query logits and boxes must share [B,Q].")
    if query_boxes.shape[-1] != 4 or object_boxes.ndim != 2:
        raise ValueError("Detector and object boxes must have four coordinates.")
    num_objects = object_boxes.shape[0]
    if object_image_indices.shape != (num_objects,) or object_labels.shape != (
        num_objects,
    ):
        raise ValueError("Object indices, labels, and boxes must have matching rows.")
    if num_objects == 0:
        return torch.empty(0, dtype=torch.long, device=query_boxes.device)
    if object_labels.min() < 0 or object_labels.max() >= query_logits.shape[-1]:
        raise ValueError("Object label is outside the detector logit range.")

    matched_queries = torch.full(
        (num_objects,), -1, dtype=torch.long, device=query_boxes.device
    )
    predicted_corners = center_to_corners(query_boxes.float())
    target_centers = corners_to_center(object_boxes.float())
    for image_index in object_image_indices.unique(sorted=True):
        target_indices = torch.nonzero(
            object_image_indices == image_index, as_tuple=False
        ).flatten()
        image = int(image_index.item())
        labels = object_labels[target_indices]
        class_cost = -query_logits[image].float().sigmoid()[:, labels].transpose(0, 1)
        box_cost = torch.cdist(
            target_centers[target_indices], query_boxes[image].float(), p=1
        )
        giou_cost = -pairwise_generalized_iou(
            object_boxes[target_indices].float(), predicted_corners[image]
        )
        cost = (
            class_weight * class_cost + bbox_weight * box_cost + giou_weight * giou_cost
        )
        target_rows, query_columns = linear_sum_assignment(cost.detach().cpu().numpy())
        assigned_targets = target_indices[
            torch.as_tensor(target_rows, device=target_indices.device)
        ]
        matched_queries[assigned_targets] = torch.as_tensor(
            query_columns, device=matched_queries.device
        )
    if (matched_queries < 0).any():
        raise RuntimeError(
            "Failed to match every segmentation object; an image may contain more "
            "targets than detector queries."
        )
    return matched_queries


def segmentation_objective(
    mask_logits: torch.Tensor,
    target_masks: torch.Tensor,
    object_image_indices: torch.Tensor,
    *,
    batch_size: int,
    object_weights: torch.Tensor | None = None,
    valid_pixels: torch.Tensor | None = None,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    dice_smooth: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Average matched mask losses per object and then over non-empty images."""
    if mask_logits.shape != target_masks.shape:
        raise ValueError("Mask logits and targets must have equal shapes.")
    if mask_logits.shape[0] != object_image_indices.shape[0]:
        raise ValueError("Object image indices must align with masks.")
    if batch_size <= 0 or dice_smooth <= 0.0:
        raise ValueError("Batch size and Dice smoothing must be positive.")
    if mask_logits.shape[0] == 0:
        zero = mask_logits.sum() * 0.0
        return {"loss": zero, "bce": zero.detach(), "dice_loss": zero.detach()}
    if object_weights is None:
        object_weights = mask_logits.new_ones(mask_logits.shape[0])
    if object_weights.shape != (mask_logits.shape[0],):
        raise ValueError("object_weights must have one value per mask.")
    if bool((object_weights < 0).any()):
        raise ValueError("object_weights must be non-negative.")
    if valid_pixels is None:
        valid_pixels = torch.ones_like(target_masks)
    if valid_pixels.shape != target_masks.shape:
        raise ValueError("valid_pixels must have the same shape as target masks.")
    valid_pixels = valid_pixels.to(mask_logits.dtype)

    reduce_dims = tuple(range(1, mask_logits.ndim))
    bce_pixels = (
        F.binary_cross_entropy_with_logits(mask_logits, target_masks, reduction="none")
        * valid_pixels
    )
    valid_counts = valid_pixels.sum(dim=reduce_dims).clamp_min(1.0)
    bce_values = bce_pixels.sum(dim=reduce_dims) / valid_counts
    probabilities = mask_logits.sigmoid()
    intersection = (probabilities * target_masks * valid_pixels).sum(dim=reduce_dims)
    denominator = (probabilities * valid_pixels).sum(dim=reduce_dims) + (
        target_masks * valid_pixels
    ).sum(dim=reduce_dims)
    dice_values = 1.0 - (2.0 * intersection + dice_smooth) / (denominator + dice_smooth)

    def per_image_mean(values: torch.Tensor) -> torch.Tensor:
        totals = values.new_zeros(batch_size)
        weights = values.new_zeros(batch_size)
        totals.scatter_add_(0, object_image_indices, values * object_weights)
        weights.scatter_add_(0, object_image_indices, object_weights)
        valid = weights > 0
        if not bool(valid.any()):
            return values.sum() * 0.0
        return (totals / weights.clamp_min(1e-8))[valid].mean()

    bce = per_image_mean(bce_values)
    dice_loss = per_image_mean(dice_values)
    return {
        "loss": bce_weight * bce + dice_weight * dice_loss,
        "bce": bce.detach(),
        "dice_loss": dice_loss.detach(),
    }


__all__ = [
    "center_to_corners",
    "corners_to_center",
    "match_object_queries",
    "pairwise_generalized_iou",
    "segmentation_objective",
]
