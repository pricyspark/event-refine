from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset

from flowsis.artifacts import atomic_write_text
from flowsis.base_checkpoint import load_base
from flowsis.cli.common import dataset_from_args, ensure_split_exists
from flowsis.cli.train.train_base import build_dataloader, build_dataset
from flowsis.matching import center_to_corners, match_object_queries
from flowsis.utils import build_autocast_context, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate set-based FlowSIS instance and union segmentation."
    )
    parser.add_argument("--model-path", type=Path, default=Path("outputs/base/final"))
    parser.add_argument("--dataset-path", default="data/new/segmentation-dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--output-path", type=Path, default=Path("outputs/base-eval.json")
    )
    parser.add_argument(
        "--pr-output-dir",
        type=Path,
        help="Write box and mask AP50 precision-recall curves as CSV and PNG.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--detection-threshold", type=float)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--oracle-target-text",
        action="store_true",
        help="Use target one-hot text for matched queries to isolate text-mixture errors.",
    )
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--device")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--detector-amp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _mask_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[float, float]:
    prediction = prediction.bool()
    target = target.bool()
    intersection = float((prediction & target).sum().item())
    prediction_area = float(prediction.sum().item())
    target_area = float(target.sum().item())
    union = prediction_area + target_area - intersection
    dice = (2.0 * intersection + 1.0) / (prediction_area + target_area + 1.0)
    iou = (intersection + 1.0) / (union + 1.0)
    return dice, iou


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    values = mask.float()[None, None]
    dilated = F.max_pool2d(values, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-values, 3, stride=1, padding=1)
    return (dilated != eroded)[0, 0]


def _boundary_f1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    predicted_boundary = _boundary(prediction)
    target_boundary = _boundary(target)
    predicted_neighborhood = F.max_pool2d(
        predicted_boundary.float()[None, None], 3, stride=1, padding=1
    )[0, 0].bool()
    target_neighborhood = F.max_pool2d(
        target_boundary.float()[None, None], 3, stride=1, padding=1
    )[0, 0].bool()
    precision = (
        predicted_boundary & target_neighborhood
    ).sum() / predicted_boundary.sum().clamp_min(1)
    recall = (
        target_boundary & predicted_neighborhood
    ).sum() / target_boundary.sum().clamp_min(1)
    return float(
        (2.0 * precision * recall / (precision + recall).clamp_min(1e-7)).item()
    )


def _box_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(left[..., None, :2], right[..., :2])
    bottom_right = torch.minimum(left[..., None, 2:], right[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0.0).prod(dim=-1)
    left_area = (
        (left[..., 2:] - left[..., :2]).clamp_min(0.0).prod(dim=-1, keepdim=True)
    )
    right_area = (right[..., 2:] - right[..., :2]).clamp_min(0.0).prod(dim=-1)
    return intersection / (left_area + right_area - intersection).clamp_min(1e-7)


def _average_precision(records: list[tuple[float, bool]], targets: int) -> float:
    if targets == 0:
        return 0.0
    records.sort(key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, is_true_positive) in enumerate(records, start=1):
        if is_true_positive:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / targets


def _precision_recall_curve(
    records: list[tuple[float, bool]], targets: int
) -> dict[str, np.ndarray]:
    ranked = sorted(records, key=lambda item: item[0], reverse=True)
    scores = np.asarray([score for score, _ in ranked], dtype=np.float64)
    true_positives = np.asarray(
        [is_true_positive for _, is_true_positive in ranked], dtype=np.float64
    )
    cumulative_true_positives = np.cumsum(true_positives)
    cumulative_false_positives = np.cumsum(1.0 - true_positives)
    recall = (
        cumulative_true_positives / targets
        if targets
        else np.zeros_like(cumulative_true_positives)
    )
    precision = cumulative_true_positives / np.maximum(
        cumulative_true_positives + cumulative_false_positives,
        np.finfo(np.float64).eps,
    )
    return {"scores": scores, "recall": recall, "precision": precision}


def write_precision_recall_curves(
    output_dir: Path,
    curves: dict[str, dict[str, np.ndarray]],
    average_precision: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "precision_recall.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "score", "recall", "precision"])
        for name, curve in curves.items():
            for score, recall, precision in zip(
                curve["scores"],
                curve["recall"],
                curve["precision"],
                strict=True,
            ):
                writer.writerow([name, score, recall, precision])

    figure, axis = plt.subplots(figsize=(8, 7))
    for name, curve in curves.items():
        axis.step(
            curve["recall"],
            curve["precision"],
            where="post",
            label=f"{name} (AP={average_precision[name]:.3f})",
        )
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.01),
        xlabel="Recall",
        ylabel="Precision",
        title="FlowSIS precision-recall at IoU 0.50",
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall.png", dpi=180)
    plt.close(figure)


def _greedy_ap_records(
    scores: torch.Tensor,
    labels: torch.Tensor,
    similarity: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    threshold: float,
) -> list[tuple[float, bool]]:
    records: list[tuple[float, bool]] = []
    used: set[int] = set()
    for query in torch.argsort(scores, descending=True).tolist():
        label = int(labels[query].item())
        candidates = [
            index
            for index in range(target_labels.numel())
            if index not in used and int(target_labels[index].item()) == label
        ]
        match = None
        if candidates:
            values = similarity[query, candidates]
            best = int(torch.argmax(values).item())
            if float(values[best].item()) >= threshold:
                match = candidates[best]
                used.add(match)
        records.append((float(scores[query].item()), match is not None))
    return records


@torch.inference_mode()
def evaluate_model_with_curves(
    model: Any, loader: Any, args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    threshold = (
        model.detection_threshold
        if args.detection_threshold is None
        else args.detection_threshold
    )
    values: dict[str, list[float]] = defaultdict(list)
    per_class: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_count: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    box_records: list[tuple[float, bool]] = []
    mask_records: list[tuple[float, bool]] = []
    target_count = prediction_count = true_positive_count = 0
    empty_images = empty_false_positives = image_count = 0
    model.eval()
    for batch in loader:
        inputs = model.detector.preprocess(batch["images"], device_preprocess=True)
        pixels = cast(torch.Tensor, inputs["pixel_values"]).to(model.device)
        pixel_mask = inputs.get("pixel_mask")
        if isinstance(pixel_mask, torch.Tensor):
            pixel_mask = pixel_mask.to(model.device)
        else:
            pixel_mask = None
        object_images = batch["object_image_indices"].to(model.device)
        object_labels = batch["object_labels"].to(model.device)
        object_boxes = batch["object_boxes"].to(model.device)
        target_masks = batch["target_masks"].to(model.device) >= 0.5
        with model.detector_autocast_context():
            detector_output = model.detector.forward_features(
                pixels, pixel_mask=pixel_mask
            )
        matches = match_object_queries(
            detector_output.query_logits,
            detector_output.query_boxes,
            object_images,
            object_labels,
            object_boxes,
        )
        class_weights = None
        if getattr(args, "oracle_target_text", False) and matches.numel():
            class_weights = model.head.query_text_fusion.predicted_class_weights(
                detector_output.query_logits
            ).clone()
            class_weights[object_images, matches] = F.one_hot(
                object_labels, num_classes=class_weights.shape[-1]
            ).to(class_weights.dtype)
        with build_autocast_context(enabled=args.amp, device=model.device):
            output = model.head(detector_output, class_weights=class_weights)
        scores, labels = output.detector.query_logits.sigmoid().max(dim=-1)
        predicted_boxes = center_to_corners(output.detector.query_boxes).clamp(0.0, 1.0)
        accepted = scores >= threshold
        image_object_counts = torch.bincount(
            object_images, minlength=batch["batch_size"]
        )
        prediction_count += int(accepted.sum().item())
        target_count += object_labels.numel()
        if matches.numel():
            oracle_logits = output.mask_logits[object_images, matches]
            oracle_logits = F.interpolate(
                oracle_logits[:, None],
                size=target_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            for target_index in range(matches.numel()):
                image = int(object_images[target_index].item())
                query = int(matches[target_index].item())
                target = target_masks[target_index]
                oracle_prediction = (
                    oracle_logits[target_index].sigmoid() >= args.mask_threshold
                )
                oracle_dice, oracle_iou = _mask_metrics(oracle_prediction, target)
                values["oracle_dice"].append(oracle_dice)
                values["oracle_iou"].append(oracle_iou)
                label = int(object_labels[target_index].item())
                valid = (
                    bool(accepted[image, query]) and int(labels[image, query]) == label
                )
                dice, iou, boundary = 0.0, 0.0, 0.0
                if valid:
                    dice, iou = oracle_dice, oracle_iou
                    boundary = _boundary_f1(oracle_prediction, target)
                values["matched_dice"].append(dice)
                values["matched_iou"].append(iou)
                values["boundary_f1"].append(boundary)
                per_class[label]["matched_dice"].append(dice)
                per_class[label]["matched_iou"].append(iou)
                count = int(image_object_counts[image].item())
                stratum = "single" if count == 1 else "multi"
                per_count[stratum]["matched_dice"].append(dice)
                per_count[stratum]["matched_iou"].append(iou)
                box_overlap = float(
                    _box_iou(
                        predicted_boxes[image, query : query + 1],
                        object_boxes[target_index : target_index + 1],
                    )[0, 0].item()
                )
                if valid and box_overlap >= 0.5:
                    true_positive_count += 1
        for image in range(batch["batch_size"]):
            target_indices = torch.nonzero(
                object_images == image, as_tuple=False
            ).flatten()
            target_labels = object_labels[target_indices]
            target_box_values = object_boxes[target_indices]
            box_similarity = _box_iou(predicted_boxes[image], target_box_values)
            box_records.extend(
                _greedy_ap_records(
                    scores[image],
                    labels[image],
                    box_similarity,
                    target_labels,
                    threshold=0.5,
                )
            )
            if target_indices.numel():
                low_targets = F.interpolate(
                    target_masks[target_indices, None].float(),
                    size=output.mask_logits.shape[-2:],
                    mode="nearest",
                )[:, 0].bool()
            else:
                low_targets = target_masks.new_empty(
                    (0, *output.mask_logits.shape[-2:])
                )
            query_masks = output.mask_logits[image].sigmoid() >= args.mask_threshold
            if target_indices.numel():
                intersection = (
                    (query_masks[:, None] & low_targets[None]).sum(dim=(-2, -1)).float()
                )
                union = (
                    (query_masks[:, None] | low_targets[None]).sum(dim=(-2, -1)).float()
                )
                mask_similarity = intersection / union.clamp_min(1.0)
            else:
                mask_similarity = scores.new_empty((scores.shape[1], 0))
            mask_records.extend(
                _greedy_ap_records(
                    scores[image],
                    labels[image],
                    mask_similarity,
                    target_labels,
                    threshold=0.5,
                )
            )
            accepted_queries = torch.nonzero(accepted[image], as_tuple=False).flatten()
            target_union = (
                target_masks[target_indices].any(dim=0)
                if target_indices.numel()
                else target_masks.new_zeros(target_masks.shape[-2:])
            )
            if accepted_queries.numel():
                union_logits = output.mask_logits[image, accepted_queries].amax(dim=0)
                union_prediction = (
                    F.interpolate(
                        union_logits[None, None],
                        size=target_union.shape,
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0].sigmoid()
                    >= args.mask_threshold
                )
            else:
                union_prediction = torch.zeros_like(target_union)
            union_dice, union_iou = _mask_metrics(union_prediction, target_union)
            values["union_dice"].append(union_dice)
            values["union_iou"].append(union_iou)
            object_count = target_indices.numel()
            stratum = (
                "empty"
                if object_count == 0
                else "single" if object_count == 1 else "multi"
            )
            per_count[stratum]["union_dice"].append(union_dice)
            per_count[stratum]["union_iou"].append(union_iou)
            if not target_indices.numel():
                empty_images += 1
                empty_false_positives += accepted_queries.numel()
            image_count += 1
            if args.max_images is not None and image_count >= args.max_images:
                break
        if args.max_images is not None and image_count >= args.max_images:
            break
    summary = {
        name: sum(metric) / max(len(metric), 1) for name, metric in values.items()
    }
    summary.update(
        {
            "box_ap50": _average_precision(box_records, target_count),
            "mask_ap50": _average_precision(mask_records, target_count),
            "precision": true_positive_count / max(prediction_count, 1),
            "recall": true_positive_count / max(target_count, 1),
            "false_positives_per_empty_image": empty_false_positives
            / max(empty_images, 1),
            "images": image_count,
            "targets": target_count,
            "predictions": prediction_count,
            "per_class": {
                model.label_names[label]: {
                    name: sum(metric) / max(len(metric), 1)
                    for name, metric in metrics.items()
                }
                for label, metrics in per_class.items()
            },
            "object_count_strata": {
                stratum: {
                    name: sum(metric) / max(len(metric), 1)
                    for name, metric in metrics.items()
                }
                for stratum, metrics in per_count.items()
            },
        }
    )
    curves = {
        "box AP50": _precision_recall_curve(box_records, target_count),
        "mask AP50": _precision_recall_curve(mask_records, target_count),
    }
    return summary, curves


def evaluate_model(model: Any, loader: Any, args: argparse.Namespace) -> dict[str, Any]:
    summary, _ = evaluate_model_with_curves(model, loader, args)
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    model = load_base(
        args.model_path,
        device=device,
        use_amp=args.amp,
        detector_amp=args.detector_amp,
    )
    dataset = dataset_from_args(args)
    ensure_split_exists(dataset, args.split, role="evaluation")
    split = cast(Dataset, dataset[args.split])
    if args.max_images is not None:
        split = split.select(range(min(args.max_images, len(split))))
    data_args = argparse.Namespace(
        image_size=model.detector.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )
    evaluation_dataset = build_dataset(split, data_args, training=False)
    loader = build_dataloader(evaluation_dataset, data_args, training=False)
    summary, curves = evaluate_model_with_curves(model, loader, args)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    atomic_write_text(args.output_path, text)
    if args.pr_output_dir is not None:
        write_precision_recall_curves(
            args.pr_output_dir,
            curves,
            {
                "box AP50": summary["box_ap50"],
                "mask AP50": summary["mask_ap50"],
            },
        )


if __name__ == "__main__":
    main()
