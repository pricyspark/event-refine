"""Class-aware box AP with 101-point interpolation; Codex-assisted implementation."""

from dataclasses import dataclass

import torch


@dataclass
class DetectionRecord:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    target_boxes: torch.Tensor
    target_labels: torch.Tensor


def box_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    intersection = (
        (
            torch.minimum(left[:, None, 2:], right[None, :, 2:])
            - torch.maximum(left[:, None, :2], right[None, :, :2])
        )
        .clamp_min(0)
        .prod(dim=-1)
    )
    left_area = (left[:, 2:] - left[:, :2]).clamp_min(0).prod(dim=-1)
    right_area = (right[:, 2:] - right[:, :2]).clamp_min(0).prod(dim=-1)
    return intersection / (
        left_area[:, None] + right_area[None] - intersection
    ).clamp_min(1e-8)


def detection_metrics(
    records: list[DetectionRecord], classes: tuple[str, ...]
) -> dict[str, object]:
    """AP50/AP50:95 for non-crowd boxes; not the complete COCO evaluator.

    Inputs are CPU tensors in original image coordinates. Empty annotated frames
    contribute false positives. Classes without ground truth are excluded from
    the mean and reported as null.
    """
    per_class: dict[str, dict[str, float] | None] = {}
    for label, name in enumerate(classes):
        targets = [
            record.target_boxes[record.target_labels == label] for record in records
        ]
        count = sum(boxes.shape[0] for boxes in targets)
        if not count:
            per_class[name] = None
            continue
        predictions: list[tuple[float, int, torch.Tensor]] = []
        for image, record in enumerate(records):
            selected = record.labels == label
            predictions.extend(
                (float(score), image, box)
                for score, box in zip(
                    record.scores[selected], record.boxes[selected], strict=True
                )
            )
        predictions.sort(key=lambda item: item[0], reverse=True)
        average_precisions = []
        for threshold in torch.linspace(0.5, 0.95, 10).tolist():
            matched = [torch.zeros(len(boxes), dtype=torch.bool) for boxes in targets]
            hits = []
            for _, image, box in predictions:
                overlaps = box_iou(box[None], targets[image])[0]
                overlaps[matched[image]] = -1
                if overlaps.numel() and float(overlaps.max()) + 1e-7 >= threshold:
                    matched[image][overlaps.argmax()] = True
                    hits.append(1.0)
                else:
                    hits.append(0.0)
            if not hits:
                average_precisions.append(0.0)
                continue
            true_positives = torch.tensor(hits).cumsum(0)
            precision = true_positives / torch.arange(1, len(hits) + 1)
            recall = true_positives / count
            interpolated = [
                (
                    float(precision[recall >= level].max())
                    if bool((recall >= level).any())
                    else 0.0
                )
                for level in torch.linspace(0, 1, 101)
            ]
            average_precisions.append(sum(interpolated) / 101)
        per_class[name] = {
            "AP50": average_precisions[0],
            "AP50_95": sum(average_precisions) / 10,
        }
    valid = [values for values in per_class.values() if values is not None]
    return {
        "AP50": sum(values["AP50"] for values in valid) / len(valid) if valid else None,
        "AP50_95": (
            sum(values["AP50_95"] for values in valid) / len(valid) if valid else None
        ),
        "per_class": per_class,
        "annotated_frames": len(records),
    }
