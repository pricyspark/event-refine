from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn.functional as F
from datasets import ClassLabel, Dataset, DatasetDict, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader

from flowsis.artifacts import atomic_torch_save
from flowsis.base import FlowSISBase
from flowsis.base_checkpoint import load_base, save_base_checkpoint
from flowsis.cli.train import train_base
from flowsis.data.object_records import get_object_feature_schema
from flowsis.data.temporal import (
    DynamicClipBatchSampler,
    TemporalClipDataset,
    collate_temporal_clips,
)
from flowsis.matching import match_object_queries, segmentation_objective
from flowsis.pretrained import load_flow_estimator
from flowsis.temporal import TemporalRefinementBranch, resize_flow
from flowsis.utils import (
    build_autocast_context,
    build_grad_scaler,
    get_device,
    set_seed,
)

TEMPORAL_CHECKPOINT_VERSION = 2
TEMPORAL_CHECKPOINT_FILE = "temporal.pt"


def _label_names(split: Dataset) -> tuple[str, ...]:
    schema = get_object_feature_schema(split.features["objects"])
    category = schema["category"]
    if hasattr(category, "feature"):
        category = category.feature
    if not isinstance(category, ClassLabel):
        raise TypeError("objects.category must use a datasets.ClassLabel.")
    return tuple(category.names)


def build_loader(
    split: Dataset,
    args: argparse.Namespace,
    *,
    training: bool,
) -> tuple[DataLoader, DynamicClipBatchSampler]:
    dataset = TemporalClipDataset(
        split,
        image_size=args.image_size,
        training=training,
    )
    sampler = DynamicClipBatchSampler(
        dataset.frame_indices,
        batch_size=args.batch_size,
        batches_per_epoch=(
            args.batches_per_epoch
            if training
            else min(args.validation_batches, args.batches_per_epoch)
        ),
        min_length=args.min_clip_length if training else args.max_clip_length,
        max_length=args.max_clip_length,
        strides=args.strides if training else (1,),
        stride_weights=args.stride_weights if training else (1.0,),
        positive_indices=dataset.positive_indices,
        seed=args.seed + (0 if training else 10_000),
    )
    return (
        DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_temporal_clips,
            pin_memory=args.device is None or str(args.device).startswith("cuda"),
            persistent_workers=args.num_workers > 0,
        ),
        sampler,
    )


def _base_sequence(
    base: FlowSISBase,
    frames: torch.Tensor,
    *,
    train_head: bool,
) -> Any:
    batch_size, length = frames.shape[:2]
    raw_frames = [frame for frame in frames.flatten(0, 1)]
    inputs = base.detector.preprocess(raw_frames, device_preprocess=True)
    pixels = cast(torch.Tensor, inputs["pixel_values"]).to(base.device)
    pixel_mask = inputs.get("pixel_mask")
    pixel_mask = (
        pixel_mask.to(base.device) if isinstance(pixel_mask, torch.Tensor) else None
    )
    with torch.no_grad(), base.detector_autocast_context():
        detector = base.detector.forward_features(pixels, pixel_mask=pixel_mask)
    context = torch.enable_grad() if train_head else torch.no_grad()
    with context:
        output = base.head(detector)
    output.mask_logits = output.mask_logits.unflatten(0, (batch_size, length))
    output.pixel_features = output.pixel_features.unflatten(0, (batch_size, length))
    output.class_weights = output.class_weights.unflatten(0, (batch_size, length))
    output.detector.query_logits = output.detector.query_logits.unflatten(
        0, (batch_size, length)
    )
    output.detector.query_boxes = output.detector.query_boxes.unflatten(
        0, (batch_size, length)
    )
    return output


def select_training_observations(
    output: Any,
    labels: torch.Tensor,
    boxes: torch.Tensor,
    base_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle-match valid observations without teaching the runtime selector."""
    batch_size, length, track_count = labels.shape
    height, width = output.mask_logits.shape[-2:]
    logits = output.mask_logits.new_zeros(
        (batch_size, length, track_count, 1, height, width)
    )
    scores = output.mask_logits.new_zeros((batch_size, length, track_count))
    class_probabilities = output.mask_logits.new_zeros(
        (batch_size, length, track_count, output.class_weights.shape[-1])
    )
    observed = base_valid & labels.ge(0)
    for batch_index in range(batch_size):
        for timestamp in range(length):
            target_indices = torch.nonzero(
                observed[batch_index, timestamp], as_tuple=False
            ).flatten()
            if not target_indices.numel():
                continue
            queries = match_object_queries(
                output.detector.query_logits[batch_index, timestamp][None],
                output.detector.query_boxes[batch_index, timestamp][None],
                torch.zeros_like(target_indices),
                labels[batch_index, timestamp, target_indices],
                boxes[batch_index, timestamp, target_indices],
            )
            logits[batch_index, timestamp, target_indices, 0] = output.mask_logits[
                batch_index, timestamp, queries
            ]
            probabilities = output.class_weights[batch_index, timestamp, queries]
            class_probabilities[batch_index, timestamp, target_indices] = probabilities
            scores[batch_index, timestamp, target_indices] = probabilities.amax(dim=-1)
    return logits, scores, class_probabilities, observed


def temporal_objective(
    temporal: TemporalRefinementBranch,
    flow_estimator: torch.nn.Module,
    frames: torch.Tensor,
    base_output: Any,
    batch: Mapping[str, Any],
    *,
    visibility_weight: float,
    preferred_track_weight: float,
    use_annotation_confidence: bool,
) -> dict[str, torch.Tensor]:
    labels = batch["labels"].to(frames.device)
    boxes = batch["boxes"].to(frames.device)
    base_valid = batch["base_valid"].to(frames.device)
    mask_valid = batch["mask_valid"].to(frames.device)
    visible = batch["visible"].to(frames.device)
    target_masks = batch["masks"].to(frames.device)
    confidence = batch["annotation_confidence"].to(frames.device)
    preferred = batch["preferred_track"].to(frames.device)
    ignore = batch["ignore"].to(frames.device)
    frame_delta = batch["frame_delta"].to(frames.device)
    base_logits, scores, class_probabilities, observed = select_training_observations(
        base_output, labels, boxes, base_valid
    )
    batch_size, length, track_count = labels.shape
    output_size = base_logits.shape[-2:]
    targets = F.interpolate(
        target_masks.flatten(0, 2)[:, None], size=output_size, mode="nearest"
    ).unflatten(0, (batch_size, length, track_count))[:, :, :, 0]
    missing = temporal.missing_base_logit.to(base_logits.dtype)
    previous_logits = torch.where(
        observed[:, 0, :, None, None, None], base_logits[:, 0], missing
    )
    previous_hidden = base_logits.new_zeros(
        (batch_size, track_count, temporal.hidden_channels, *output_size)
    )
    previous_classes = class_probabilities[:, 0]
    mask_logits: list[torch.Tensor] = []
    visibility_logits: list[torch.Tensor] = []
    for timestamp in range(1, length):
        with torch.no_grad():
            flow = flow_estimator(frames[:, timestamp], frames[:, timestamp - 1])
        instance_flow = resize_flow(flow, output_size)[:, None].expand(
            -1, track_count, -1, -1, -1
        )
        current_classes = class_probabilities[:, timestamp]
        agreement = 1.0 - 0.5 * (current_classes - previous_classes).abs().sum(-1)
        agreement = torch.where(
            observed[:, timestamp], agreement, torch.zeros_like(agreement)
        )
        current_features = base_output.pixel_features[:, timestamp]
        previous_features = base_output.pixel_features[:, timestamp - 1]
        result = temporal(
            frames[:, timestamp, None]
            .expand(-1, track_count, -1, -1, -1)
            .flatten(0, 1),
            frames[:, timestamp - 1, None]
            .expand(-1, track_count, -1, -1, -1)
            .flatten(0, 1),
            base_logits[:, timestamp].flatten(0, 1),
            previous_logits.flatten(0, 1),
            instance_flow.flatten(0, 1),
            current_features=current_features[:, None]
            .expand(-1, track_count, -1, -1, -1)
            .flatten(0, 1),
            previous_features=previous_features[:, None]
            .expand(-1, track_count, -1, -1, -1)
            .flatten(0, 1),
            previous_hidden=previous_hidden.flatten(0, 1),
            observation_valid=observed[:, timestamp].flatten(),
            observation_score=scores[:, timestamp].flatten(),
            class_agreement=agreement.flatten(),
            frame_delta=frame_delta[:, timestamp]
            .flatten()[:, None]
            .expand(-1, track_count)
            .flatten(),
        )
        previous_logits = result.final_logits.unflatten(0, (batch_size, track_count))
        assert result.hidden_state is not None and result.visibility_logits is not None
        previous_hidden = result.hidden_state.unflatten(0, (batch_size, track_count))
        previous_classes = torch.where(
            observed[:, timestamp, :, None], current_classes, previous_classes
        )
        mask_logits.append(previous_logits[:, :, 0])
        visibility_logits.append(
            result.visibility_logits.unflatten(0, (batch_size, track_count))[:, :, 0]
        )

    predicted_masks = torch.stack(mask_logits, dim=1)
    predicted_visibility = torch.stack(visibility_logits, dim=1)
    valid_masks = mask_valid[:, 1:]
    image_indices = (
        torch.arange(batch_size * (length - 1), device=frames.device)[:, None]
        .expand(-1, track_count)
        .reshape(batch_size, length - 1, track_count)
    )
    selected_logits = predicted_masks[valid_masks]
    selected_targets = targets[:, 1:][valid_masks]
    selected_indices = image_indices[valid_masks]
    valid_pixels = (
        F.interpolate(
            ignore[:, 1:].flatten(0, 1)[:, None].float(),
            size=output_size,
            mode="nearest",
        ).unflatten(0, (batch_size, length - 1))[:, :, 0]
        < 0.5
    )
    selected_valid_pixels = valid_pixels[:, :, None].expand(
        -1, -1, track_count, -1, -1
    )[valid_masks]
    weights = torch.ones_like(confidence[:, 1:])
    if use_annotation_confidence:
        weights = weights * confidence[:, 1:]
    if preferred_track_weight != 1.0:
        weights = weights * torch.where(preferred[:, 1:], preferred_track_weight, 1.0)
    mask_result = segmentation_objective(
        selected_logits,
        selected_targets,
        selected_indices,
        batch_size=batch_size * (length - 1),
        object_weights=weights[valid_masks],
        valid_pixels=selected_valid_pixels,
    )
    track_valid = labels.ge(0).any(dim=1)
    visibility_valid = track_valid[:, None].expand(-1, length - 1, -1)
    visibility_loss = F.binary_cross_entropy_with_logits(
        predicted_visibility[visibility_valid],
        visible[:, 1:][visibility_valid].float(),
    )
    return {
        "loss": mask_result["loss"] + visibility_weight * visibility_loss,
        "mask_loss": mask_result["loss"].detach(),
        "visibility_loss": visibility_loss.detach(),
        "bce": mask_result["bce"],
        "dice_loss": mask_result["dice_loss"],
    }


def _still_replay_loader(
    args: argparse.Namespace, base: FlowSISBase
) -> DataLoader | None:
    if not args.finetune_mask_head:
        return None
    if args.still_dataset_path is None:
        raise ValueError("Mask-head fine-tuning requires --still-dataset-path replay.")
    data = load_from_disk(str(args.still_dataset_path))
    if not isinstance(data, DatasetDict) or args.train_split not in data:
        raise ValueError(
            "Still replay data must be a DatasetDict with the train split."
        )
    split = cast(Dataset, data[args.train_split])
    if _label_names(split) != base.label_names:
        raise ValueError("Still replay and base model class orders differ.")
    replay_args = SimpleNamespace(
        images_dir=args.still_images_dir,
        masks_dir=args.still_masks_dir,
        image_size=args.image_size,
        use_overlap_augment=False,
        overlap_min_overlays=0,
        overlap_max_overlays=0,
        overlap_p=0.0,
        use_rotation_augment=False,
        use_roi_square_augment=False,
        use_photometric_augment=False,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        bce_loss_weight=1.0,
        dice_loss_weight=1.0,
        dice_smooth=1.0,
        detector_loss_weight=0.0,
    )
    prepared = train_base.build_dataset(split, replay_args, training=True)
    return train_base.build_dataloader(prepared, replay_args, training=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the residual temporal refiner.")
    parser.add_argument(
        "--dataset-path", type=Path, default=Path("data/new/temporal-dataset")
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/temporal"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batches-per-epoch", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=100)
    parser.add_argument("--min-clip-length", type=int, default=4)
    parser.add_argument("--max-clip-length", type=int, default=8)
    parser.add_argument("--strides", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument(
        "--stride-weights", type=float, nargs="+", default=(0.7, 0.2, 0.1)
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--visibility-loss-weight", type=float, default=0.1)
    parser.add_argument("--preferred-track-weight", type=float, default=1.0)
    parser.add_argument("--ignore-annotation-confidence", action="store_true")
    parser.add_argument("--flow-model", default="raft_small")
    parser.add_argument("--flow-checkpoint")
    parser.add_argument("--finetune-mask-head", action="store_true")
    parser.add_argument("--base-head-lr-scale", type=float, default=0.1)
    parser.add_argument("--still-dataset-path", type=Path)
    parser.add_argument("--still-images-dir", type=Path, default=Path("data/frames"))
    parser.add_argument("--still-masks-dir", type=Path, default=Path("data/masks"))
    parser.add_argument("--still-replay-weight", type=float, default=1.0)
    parser.add_argument("--device")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _mean(metrics: list[Mapping[str, torch.Tensor]]) -> dict[str, float]:
    return (
        {
            key: sum(float(item[key]) for item in metrics) / max(len(metrics), 1)
            for key in metrics[0]
        }
        if metrics
        else {}
    )


def main() -> None:
    args = parse_args()
    if len(args.strides) != len(args.stride_weights):
        raise ValueError("--strides and --stride-weights must align.")
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    data = load_from_disk(str(args.dataset_path))
    if not isinstance(data, DatasetDict):
        raise TypeError("Temporal data must be a DatasetDict.")
    train_split = cast(Dataset, data[args.train_split])
    validation_split = cast(Dataset, data[args.validation_split])
    base = load_base(args.base_model, device=device, use_amp=args.amp)
    if _label_names(train_split) != base.label_names:
        raise ValueError("Temporal dataset and base model class orders differ.")
    base.requires_grad_(False).eval()
    if args.finetune_mask_head:
        base.head.requires_grad_(True).train()
    flow = load_flow_estimator(args.flow_model, args.flow_checkpoint).to(device)
    flow.requires_grad_(False).eval()
    train_loader, train_sampler = build_loader(train_split, args, training=True)
    validation_loader, validation_sampler = build_loader(
        validation_split, args, training=False
    )
    sample = next(iter(train_loader))
    frames = sample["frames"].to(device)
    with torch.no_grad():
        sample_output = _base_sequence(base, frames, train_head=False)
    temporal = TemporalRefinementBranch(
        feature_channels=sample_output.pixel_features.shape[2]
    ).to(device)
    temporal_parameters = list(temporal.parameters())
    groups = [{"params": temporal_parameters, "lr": args.learning_rate}]
    trainable_parameters = list(temporal_parameters)
    if args.finetune_mask_head:
        head_parameters = list(base.head.parameters())
        groups.append(
            {
                "params": head_parameters,
                "lr": args.learning_rate * args.base_head_lr_scale,
            }
        )
        trainable_parameters.extend(head_parameters)
    optimizer = AdamW(groups, weight_decay=args.weight_decay)
    scaler = build_grad_scaler(enabled=args.amp, device=device)
    replay_loader = _still_replay_loader(args, base)
    replay = cycle(replay_loader) if replay_loader is not None else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    best = math.inf
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        temporal.train()
        train_metrics = []
        for batch in train_loader:
            frames = batch["frames"].to(device)
            with build_autocast_context(enabled=args.amp, device=device):
                base_output = _base_sequence(
                    base, frames, train_head=args.finetune_mask_head
                )
                result = temporal_objective(
                    temporal,
                    flow,
                    frames,
                    base_output,
                    batch,
                    visibility_weight=args.visibility_loss_weight,
                    preferred_track_weight=args.preferred_track_weight,
                    use_annotation_confidence=not args.ignore_annotation_confidence,
                )
                loss = result["loss"]
                if replay is not None:
                    still = train_base.compute_batch_loss(
                        base,
                        next(replay),
                        stage="head",
                        target_text_ratio=0.0,
                        args=SimpleNamespace(
                            bce_loss_weight=1.0,
                            dice_loss_weight=1.0,
                            dice_smooth=1.0,
                            detector_loss_weight=0.0,
                        ),
                    )
                    loss = loss + args.still_replay_weight * still["loss"]
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Temporal training produced a non-finite loss."
                )
            if scaler is None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.gradient_clip)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            train_metrics.append(result)

        validation_sampler.set_epoch(0)
        temporal.eval()
        validation_metrics = []
        with torch.no_grad():
            for batch in validation_loader:
                frames = batch["frames"].to(device)
                with build_autocast_context(enabled=args.amp, device=device):
                    output = _base_sequence(base, frames, train_head=False)
                    validation_metrics.append(
                        temporal_objective(
                            temporal,
                            flow,
                            frames,
                            output,
                            batch,
                            visibility_weight=args.visibility_loss_weight,
                            preferred_track_weight=1.0,
                            use_annotation_confidence=True,
                        )
                    )
        train_mean = _mean(train_metrics)
        validation_mean = _mean(validation_metrics)
        print({"epoch": epoch + 1, "train": train_mean, "validation": validation_mean})
        checkpoint = {
            "format_version": TEMPORAL_CHECKPOINT_VERSION,
            "architecture": "residual_temporal_refiner",
            "state_dict": temporal.state_dict(),
            "config": {
                "channels": temporal.channels,
                "feature_channels": temporal.feature_channels,
                "context_channels": temporal.context_channels,
                "hidden_channels": temporal.hidden_channels,
                "residual_limit": temporal.residual_limit,
            },
            "epoch": epoch + 1,
            "validation": validation_mean,
        }
        atomic_torch_save(checkpoint, args.output_dir / TEMPORAL_CHECKPOINT_FILE)
        if validation_mean.get("loss", math.inf) < best:
            best = validation_mean["loss"]
            atomic_torch_save(checkpoint, args.output_dir / "temporal-best.pt")
            if args.finetune_mask_head:
                save_base_checkpoint(base, args.output_dir / "base-best")


if __name__ == "__main__":
    main()
