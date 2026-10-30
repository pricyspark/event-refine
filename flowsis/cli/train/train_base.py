from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from datasets import ClassLabel, Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_scheduler

from flowsis.artifacts import atomic_write_text
from flowsis.base import FlowSISBase
from flowsis.base_checkpoint import (
    load_base_checkpoint,
    save_base_checkpoint,
)
from flowsis.cli.common import (
    add_detector_arguments,
    append_log_event,
    dataset_from_args,
    ensure_split_exists,
    log_event,
)
from flowsis.cli.train.training_manifest import write_run_manifest
from flowsis.data import (
    CallablePipeline,
    PreparedDataset,
    load_object_image,
    load_object_masks,
)
from flowsis.data.augment import (
    center_square_augment,
    overlap_augment,
    photometric_augment,
    roi_square_augment,
    rotation_augment,
)
from flowsis.data.images import get_image
from flowsis.data.object_records import (
    get_object_feature_schema,
    get_object_records,
    numeric_image_id,
)
from flowsis.data.prompts import load_prompt_bank
from flowsis.matching import match_object_queries, segmentation_objective
from flowsis.pretrained import BaseDetector, load_detector
from flowsis.pretrained.image_processing import image_to_rgb_tensor
from flowsis.set_mask import SetMaskHead, TextConditioning
from flowsis.utils import (
    build_autocast_context,
    build_grad_scaler,
    get_device,
    load_training_state,
    resolve_resume_checkpoint,
    save_training_state,
    set_seed,
)

StageName = Literal["head", "neck", "all"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the end-to-end set-based FlowSIS segmentation model."
    )
    parser.add_argument("--dataset-path", default="data/segmentation-dataset")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument(
        "--max-train-images",
        type=int,
        help="Limit the training split for smoke tests and small-set overfitting.",
    )
    parser.add_argument(
        "--max-validation-images",
        type=int,
        help="Limit validation work for smoke tests; omit for real experiments.",
    )
    parser.add_argument("--text-embeddings-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/base")
    parser.add_argument("--resume-from")
    add_detector_arguments(
        parser, model_flag="--detector-model", model_dest="detector_model_source"
    )
    parser.add_argument("--train-stages", default="head:5,neck:3,all:2")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--mask-dim", type=int, default=256)
    parser.add_argument(
        "--text-conditioning",
        choices=("frozen", "learned", "none"),
        default="frozen",
    )
    parser.add_argument("--text-temperature", type=float, default=1.0)
    parser.add_argument(
        "--pixel-detail", choices=("stride4", "stride8"), default="stride4"
    )
    parser.add_argument("--no-box-prior", action="store_true")
    parser.add_argument("--box-expansion", type=float, default=1.2)
    parser.add_argument("--box-sharpness", type=float, default=30.0)
    parser.add_argument("--box-prior-strength", type=float, default=0.25)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--detector-lr", type=float, default=1e-5)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--detector-loss-weight", type=float, default=1.0)
    parser.add_argument("--bce-loss-weight", type=float, default=1.0)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--dice-smooth", type=float, default=1.0)
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int)
    parser.add_argument("--max-ap50-drop", type=float, default=0.01)
    parser.add_argument("--max-empty-fp-increase", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        help=(
            "Warm up for this fraction of optimizer steps. Mutually exclusive "
            "with a non-zero --warmup-steps value."
        ),
    )
    parser.add_argument(
        "--scheduler-type",
        choices=("linear", "cosine", "constant_with_warmup"),
        default="linear",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--detector-amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the detector under CUDA FP16 autocast. Disabled by default because "
            "D-FINE box-distribution decoding can overflow in FP16."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument(
        "--use-overlap-augment", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overlap-min-overlays", type=int, default=0)
    parser.add_argument("--overlap-max-overlays", type=int, default=1)
    parser.add_argument("--overlap-p", type=float, default=0.25)
    parser.add_argument(
        "--use-rotation-augment", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-roi-square-augment", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-photometric-augment", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--rotation-min-degrees", type=float, default=0.0)
    parser.add_argument("--rotation-max-degrees", type=float, default=45.0)
    return parser.parse_args()


def parse_stage_spec(spec: str) -> list[tuple[StageName, int]]:
    stages: list[tuple[StageName, int]] = []
    for item in spec.split(","):
        name, separator, epochs_text = item.strip().partition(":")
        if not name:
            continue
        if name not in {"head", "neck", "all"}:
            raise ValueError(f"Unsupported training stage {name!r}.")
        epochs = int(epochs_text if separator else "1")
        if epochs <= 0:
            raise ValueError("Stage durations must be positive.")
        stages.append((cast(StageName, name), epochs))
    if not stages:
        raise ValueError("At least one training stage is required.")
    return stages


def expand_stages(
    stages: Sequence[tuple[StageName, int]],
) -> list[tuple[StageName, int, int]]:
    return [
        (stage, stage_epoch, epochs)
        for stage, epochs in stages
        for stage_epoch in range(epochs)
    ]


def build_dataset(
    split: Dataset, args: argparse.Namespace, *, training: bool
) -> Dataset:
    loader = CallablePipeline((load_object_image, load_object_masks))
    if not training:
        return cast(
            Dataset,
            PreparedDataset(
                split,
                loader=loader,
                augment=CallablePipeline(
                    (center_square_augment,), ({"crop_size": args.image_size},)
                ),
            ),
        )
    augmentations = []
    kwargs: list[dict[str, Any]] = []
    if args.use_overlap_augment:
        augmentations.append(overlap_augment)
        kwargs.append(
            {
                "min_overlays": args.overlap_min_overlays,
                "max_overlays": args.overlap_max_overlays,
                "p": args.overlap_p,
            }
        )
    if args.use_rotation_augment:
        augmentations.append(rotation_augment)
        kwargs.append(
            {
                "pad": 1,
                "min_angle": args.rotation_min_degrees,
                "max_angle": args.rotation_max_degrees,
            }
        )
    if args.use_roi_square_augment:
        augmentations.append(roi_square_augment)
        kwargs.append({"crop_size": args.image_size})
    else:
        augmentations.append(center_square_augment)
        kwargs.append({"crop_size": args.image_size})
    if args.use_photometric_augment:
        augmentations.append(photometric_augment)
        kwargs.append({})
    return cast(
        Dataset,
        PreparedDataset(
            split,
            loader=loader,
            augment=CallablePipeline(augmentations, kwargs),
            seed=args.seed,
        ),
    )


def _normalized_box(obj: dict[str, Any], example: dict[str, Any]) -> list[float]:
    x, y, width, height = (float(value) for value in obj["bbox"])
    image_width = max(float(example["width"]), 1.0)
    image_height = max(float(example["height"]), 1.0)
    return [
        x / image_width,
        y / image_height,
        (x + width) / image_width,
        (y + height) / image_height,
    ]


def collate_examples(batch: list[dict[str, Any]], *, image_size: int) -> dict[str, Any]:
    image_tensors = [
        image_to_rgb_tensor(get_image(example, convert_mode="RGB")) for example in batch
    ]
    images: torch.Tensor | list[torch.Tensor]
    if all(image.shape == image_tensors[0].shape for image in image_tensors):
        images = torch.stack(image_tensors)
    else:
        images = image_tensors
    records = [
        (image_index, example, obj)
        for image_index, example in enumerate(batch)
        for obj in get_object_records(example)
        if "mask" in obj
    ]
    masks = [
        F.interpolate(
            torch.from_numpy(obj["mask"].astype("float32", copy=False))[None, None],
            size=(image_size, image_size),
            mode="nearest",
        )[0, 0]
        for _, _, obj in records
    ]
    return {
        "images": images,
        "annotations": [
            {
                "image_id": numeric_image_id(example.get("image_id"), index),
                "annotations": [
                    {
                        "bbox": [float(value) for value in obj["bbox"]],
                        "category_id": int(obj["category"]),
                        "area": float(obj.get("area", obj["bbox"][2] * obj["bbox"][3])),
                        "iscrowd": 0,
                    }
                    for obj in get_object_records(example)
                ],
            }
            for index, example in enumerate(batch)
        ],
        "object_image_indices": torch.tensor(
            [index for index, _, _ in records], dtype=torch.long
        ),
        "object_labels": torch.tensor(
            [int(obj["category"]) for _, _, obj in records], dtype=torch.long
        ),
        "object_boxes": torch.tensor(
            [_normalized_box(obj, example) for _, example, obj in records],
            dtype=torch.float32,
        ).reshape(-1, 4),
        "object_weights": torch.tensor(
            [float(obj.get("annotation_confidence", 1.0)) for _, _, obj in records],
            dtype=torch.float32,
        ),
        "target_masks": (
            torch.stack(masks)
            if masks
            else torch.empty((0, image_size, image_size), dtype=torch.float32)
        ),
        "batch_size": len(batch),
    }


def build_dataloader(
    dataset: Dataset,
    args: argparse.Namespace,
    *,
    training: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    base_dataset = getattr(dataset, "base_dataset", dataset)
    columns = set(getattr(base_dataset, "column_names", ()))
    if training and "sample_id" in columns:
        # Each capture setup receives equal total weight, regardless of how
        # many views or tools it contains. Ungrouped images stand alone.
        sample_keys = [
            ("sample", str(value)) if value else ("image", str(index))
            for index, value in enumerate(base_dataset["sample_id"])
        ]
        images_per_sample = Counter(sample_keys)
        weights = [1.0 / images_per_sample[key] for key in sample_keys]
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=partial(collate_examples, image_size=args.image_size),
        generator=generator,
        pin_memory=args.device is None or str(args.device).startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )


def load_label_names(split: Dataset) -> tuple[str, ...]:
    features = split.features
    if features is None:
        raise ValueError("Segmentation dataset must define a feature schema.")
    schema = get_object_feature_schema(features["objects"])
    category = schema["category"]
    if hasattr(category, "feature"):
        category = category.feature
    if not isinstance(category, ClassLabel):
        raise TypeError("objects.category must be a contiguous datasets.ClassLabel.")
    return tuple(
        name if name else f"class_{index}" for index, name in enumerate(category.names)
    )


@torch.no_grad()
def build_model(
    detector: BaseDetector,
    prompt_bank: torch.Tensor,
    label_names: tuple[str, ...],
    args: argparse.Namespace,
) -> FlowSISBase:
    was_training = detector.training
    detector.eval()
    sample = torch.zeros(
        (1, 3, args.image_size, args.image_size), device=detector.device
    )
    sample_mask = torch.ones(
        (1, args.image_size, args.image_size),
        dtype=torch.long,
        device=detector.device,
    )
    features = detector.forward_features(sample, pixel_mask=sample_mask)
    detector.train(was_training)
    head_config = {
        "detail_channels": int(features.detail_features.shape[1]),
        "feature_channels": tuple(
            int(level.shape[1]) for level in features.feature_maps
        ),
        "query_dim": int(features.query_embeddings.shape[-1]),
        "mask_dim": args.mask_dim,
        "text_conditioning": cast(TextConditioning, args.text_conditioning),
        "text_temperature": args.text_temperature,
        "use_detail_features": args.pixel_detail == "stride4",
        "use_box_prior": not args.no_box_prior,
        "box_expansion": args.box_expansion,
        "box_sharpness": args.box_sharpness,
        "box_prior_initial_strength": args.box_prior_strength,
    }
    head = SetMaskHead(prompt_embeddings=prompt_bank, **head_config)
    return FlowSISBase(
        detector,
        head,
        label_names,
        head_config=head_config,
        detection_threshold=args.detection_threshold,
        mask_threshold=args.mask_threshold,
        max_detections=args.max_detections,
        use_amp=args.amp,
        detector_amp=args.detector_amp,
        device_preprocess=True,
    ).to(detector.device)


def configure_stage(
    model: FlowSISBase,
    stage: StageName,
    backbone_parameters: Sequence[torch.nn.Parameter],
) -> None:
    model.head.requires_grad_(True)
    model.detector.requires_grad_(stage != "head")
    for parameter in backbone_parameters:
        parameter.requires_grad_(stage == "all")
    model.head.train()
    model.detector.train(stage != "head")


def teacher_ratio(stage: StageName, stage_epoch: int, stage_epochs: int) -> float:
    if stage == "head":
        return 1.0
    if stage == "all":
        return 0.0
    if stage_epochs <= 1:
        return 0.0
    return 1.0 - stage_epoch / (stage_epochs - 1)


def preprocess_batch(
    model: FlowSISBase,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, torch.Tensor]]]:
    inputs = model.detector.preprocess(
        batch["images"], batch["annotations"], device_preprocess=True
    )
    pixels = cast(torch.Tensor, inputs["pixel_values"]).to(model.device)
    pixel_mask = inputs.get("pixel_mask")
    if isinstance(pixel_mask, torch.Tensor):
        pixel_mask = pixel_mask.to(model.device)
    else:
        pixel_mask = None
    labels = [
        {key: value.to(model.device) for key, value in target.items()}
        for target in cast(list[dict[str, torch.Tensor]], inputs["labels"])
    ]
    return pixels, pixel_mask, labels


def compute_batch_loss(
    model: FlowSISBase,
    batch: dict[str, Any],
    *,
    stage: StageName,
    target_text_ratio: float,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    pixels, pixel_mask, detection_labels = preprocess_batch(model, batch)
    object_image_indices = batch["object_image_indices"].to(model.device)
    object_labels = batch["object_labels"].to(model.device)
    object_boxes = batch["object_boxes"].to(model.device).clamp(0.0, 1.0)
    target_masks = batch["target_masks"].to(model.device)
    object_weights = batch["object_weights"].to(model.device)

    context = torch.no_grad() if stage == "head" else torch.enable_grad()
    with context, model.detector_autocast_context():
        detector_output = model.detector.forward_features(
            pixels,
            pixel_mask=pixel_mask,
            labels=None if stage == "head" else detection_labels,
        )
    matched_queries = match_object_queries(
        detector_output.query_logits,
        detector_output.query_boxes,
        object_image_indices,
        object_labels,
        object_boxes,
    )
    class_weights = model.head.query_text_fusion.predicted_class_weights(
        detector_output.query_logits
    )
    if target_text_ratio > 0.0 and matched_queries.numel():
        class_weights = class_weights.clone()
        target_weights = F.one_hot(
            object_labels, num_classes=class_weights.shape[-1]
        ).to(class_weights.dtype)
        predicted = class_weights[object_image_indices, matched_queries]
        class_weights[object_image_indices, matched_queries] = (
            target_text_ratio * target_weights + (1.0 - target_text_ratio) * predicted
        )
    output = model.head(detector_output, class_weights=class_weights)
    output.matches = matched_queries
    matched_logits = output.mask_logits[object_image_indices, matched_queries]
    if matched_logits.numel():
        matched_logits = F.interpolate(
            matched_logits[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        mask_result = segmentation_objective(
            matched_logits,
            target_masks,
            object_image_indices,
            batch_size=batch["batch_size"],
            object_weights=object_weights,
            bce_weight=args.bce_loss_weight,
            dice_weight=args.dice_loss_weight,
            dice_smooth=args.dice_smooth,
        )
    else:
        zero = output.mask_logits.sum() * 0.0
        mask_result = {
            "loss": zero,
            "bce": zero.detach(),
            "dice_loss": zero.detach(),
        }
    detection_loss = (
        detector_output.loss
        if detector_output.loss is not None
        else output.mask_logits.sum() * 0.0
    )
    total = mask_result["loss"] + args.detector_loss_weight * detection_loss
    return {
        "loss": total,
        "detector_loss": detection_loss.detach(),
        "mask_loss": mask_result["loss"].detach(),
        "bce": mask_result["bce"],
        "dice_loss": mask_result["dice_loss"],
    }


def evaluate(
    model: FlowSISBase,
    loader: DataLoader,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals = {
        name: 0.0 for name in ("loss", "detector_loss", "mask_loss", "bce", "dice_loss")
    }
    count = 0
    for batch in loader:
        with (
            torch.no_grad(),
            build_autocast_context(enabled=args.amp, device=model.device),
        ):
            result = compute_batch_loss(
                model, batch, stage="all", target_text_ratio=0.0, args=args
            )
        for name in totals:
            totals[name] += float(result[name].item())
        count += 1
    return {name: total / max(count, 1) for name, total in totals.items()}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_warmup_steps(
    *,
    warmup_steps: int,
    warmup_ratio: float | None,
    total_steps: int,
) -> int:
    if warmup_steps < 0:
        raise ValueError("Warmup steps must be non-negative.")
    if warmup_ratio is None:
        return warmup_steps
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("Warmup ratio must be in [0, 1).")
    if warmup_steps:
        raise ValueError("Set either --warmup-steps or --warmup-ratio, not both.")
    return round(total_steps * warmup_ratio)


def training_epoch_summary(
    *,
    loss_sum: float,
    loss_square_sum: float,
    batches: int,
    images: int,
    seconds: float,
    grad_norm_sum: float,
    max_grad_norm: float,
    skipped_optimizer_steps: int,
    peak_memory_mb: float | None,
) -> dict[str, float | int | None]:
    mean_loss = loss_sum / max(batches, 1)
    loss_variance = max(loss_square_sum / max(batches, 1) - mean_loss**2, 0.0)
    return {
        "train_loss": mean_loss,
        "train_loss_std": math.sqrt(loss_variance),
        "train_batches": batches,
        "train_images": images,
        "train_images_per_second": images / seconds if seconds > 0.0 else 0.0,
        "train_batches_per_second": batches / seconds if seconds > 0.0 else 0.0,
        "mean_grad_norm": grad_norm_sum / max(batches, 1),
        "max_grad_norm": max_grad_norm,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "peak_memory_mb": peak_memory_mb,
    }


def main() -> None:
    args = parse_args()
    if (
        min(args.detector_loss_weight, args.bce_loss_weight, args.dice_loss_weight)
        < 0.0
    ):
        raise ValueError("Loss weights must be non-negative.")
    if args.gradient_clip <= 0.0 or args.image_size <= 0:
        raise ValueError("Gradient clipping and image size must be positive.")
    if any(
        limit is not None and limit <= 0
        for limit in (args.max_train_images, args.max_validation_images)
    ):
        raise ValueError("Dataset image limits must be positive when provided.")
    stages = parse_stage_spec(args.train_stages)
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested with --device, but torch.cuda.is_available() is "
            "False. Run the PyTorch CUDA diagnostic documented in README.md before "
            "starting a full training run."
        )
    log_event(
        "runtime",
        {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "mask_head_amp": args.amp and device.type == "cuda",
            "detector_amp": args.detector_amp and device.type == "cuda",
        },
    )
    dataset = dataset_from_args(args)
    ensure_split_exists(dataset, args.train_split, role="train")
    ensure_split_exists(dataset, args.validation_split, role="validation")
    train_split = cast(Dataset, dataset[args.train_split])
    validation_split = cast(Dataset, dataset[args.validation_split])
    label_names = load_label_names(train_split)
    if args.max_train_images is not None:
        train_split = train_split.select(
            range(min(args.max_train_images, len(train_split)))
        )
    if args.max_validation_images is not None:
        validation_split = validation_split.select(
            range(min(args.max_validation_images, len(validation_split)))
        )
    log_event(
        "dataset",
        {
            "train_split": args.train_split,
            "train_images": len(train_split),
            "validation_split": args.validation_split,
            "validation_images": len(validation_split),
        },
    )
    prompt_bank = load_prompt_bank(label_names, args.text_embeddings_dir)
    detector = cast(
        BaseDetector,
        load_detector(
            args.detector_model_source,
            architecture=args.detector_architecture,
            image_size=args.image_size,
            device=device,
        ),
    )
    model = build_model(detector, prompt_bank, label_names, args)
    train_loader = build_dataloader(
        build_dataset(train_split, args, training=True),
        args,
        training=True,
    )
    validation_loader = build_dataloader(
        build_dataset(validation_split, args, training=False),
        args,
        training=False,
    )
    backbone_parameters, other_detector_parameters = (
        model.detector.split_backbone_parameters()
    )
    optimizer = AdamW(
        [
            {"params": model.head.parameters(), "lr": args.head_lr},
            {"params": other_detector_parameters, "lr": args.detector_lr},
            {"params": backbone_parameters, "lr": args.backbone_lr},
        ],
        weight_decay=args.weight_decay,
    )
    stage_epochs = expand_stages(stages)
    total_steps = len(train_loader) * len(stage_epochs)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_scheduler(
        args.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=resolve_warmup_steps(
            warmup_steps=args.warmup_steps,
            warmup_ratio=args.warmup_ratio,
            total_steps=total_steps,
        ),
        num_training_steps=max(total_steps, 1),
    )
    scaler = build_grad_scaler(enabled=args.amp, device=device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = output_dir / "training_log.jsonl"
    run_config_path = write_run_manifest(
        output_dir,
        args,
        model_config=model.detector.model_config,
        resolved={
            "device": str(device),
            "output_dir": str(output_dir),
            "detector_architecture": model.detector.architecture,
            "detector_model_source": model.detector.source,
            "label_names": label_names,
            "train_stages": stages,
            "total_steps": total_steps,
        },
    )
    log_event("saved_run_config", {"path": str(run_config_path)})
    start_epoch = 0
    global_step = 0
    resume_dir = resolve_resume_checkpoint(args.resume_from)
    if resume_dir is not None:
        checkpoint, _ = load_base_checkpoint(resume_dir)
        if checkpoint.head_config != model.head_config:
            raise ValueError("Resume checkpoint architecture does not match this run.")
        model.head.load_state_dict(checkpoint.head_state_dict)
        loaded_detector = cast(
            BaseDetector,
            load_detector(str(resume_dir), device=device, image_size=args.image_size),
        )
        model.detector.load_state_dict(loaded_detector.state_dict())
        state = load_training_state(
            resume_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        start_epoch, global_step = state.epoch, state.global_step

    baseline = evaluate(model, validation_loader, args)
    from flowsis.cli.eval.evaluate_base import evaluate_model

    args.max_images = args.max_validation_images
    baseline_quality = evaluate_model(model, validation_loader, args)
    baseline_summary = {**baseline, **baseline_quality}
    log_event("baseline_validation", baseline_summary)
    append_log_event(
        training_log_path,
        "baseline_validation",
        baseline_summary,
    )
    best_dice = float("-inf")
    for epoch_index, (stage, stage_epoch, num_stage_epochs) in enumerate(stage_epochs):
        if epoch_index < start_epoch:
            continue
        if args.max_steps is not None and global_step >= args.max_steps:
            break
        configure_stage(model, stage, backbone_parameters)
        ratio = teacher_ratio(stage, stage_epoch, num_stage_epochs)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()
        running_loss = 0.0
        running_loss_square = 0.0
        running_grad_norm = 0.0
        maximum_grad_norm = 0.0
        batches = 0
        images = 0
        skipped_optimizer_steps = 0
        for batch in train_loader:
            if args.max_steps is not None and global_step >= args.max_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            with build_autocast_context(enabled=args.amp, device=device):
                result = compute_batch_loss(
                    model,
                    batch,
                    stage=stage,
                    target_text_ratio=ratio,
                    args=args,
                )
            if not torch.isfinite(result["loss"]):
                failure = {
                    "epoch": epoch_index + 1,
                    "stage": stage,
                    "global_step": global_step + 1,
                    "loss": float(result["loss"].detach().item()),
                }
                log_event("nonfinite_loss", failure)
                append_log_event(training_log_path, "nonfinite_loss", failure)
                raise FloatingPointError(
                    f"Non-finite training loss at step {global_step + 1}."
                )
            optimizer_step_succeeded = True
            if scaler is None:
                result["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at step {global_step + 1}."
                    )
                optimizer.step()
            else:
                scale_before = scaler.get_scale()
                scaler.scale(result["loss"]).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_succeeded = scaler.get_scale() >= scale_before
                if not optimizer_step_succeeded:
                    skipped_optimizer_steps += 1
            if optimizer_step_succeeded:
                scheduler.step()
            global_step += 1
            batches += 1
            images += int(batch["batch_size"])
            loss_value = float(result["loss"].item())
            grad_norm_value = float(grad_norm.detach().item())
            running_loss += loss_value
            running_loss_square += loss_value**2
            running_grad_norm += grad_norm_value
            maximum_grad_norm = max(maximum_grad_norm, grad_norm_value)
        synchronize(device)
        elapsed = time.perf_counter() - started
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else None
        )
        validation = evaluate(model, validation_loader, args)
        quality = evaluate_model(model, validation_loader, args)
        passes_guardrails = (
            quality["box_ap50"] >= baseline_quality["box_ap50"] - args.max_ap50_drop
            and quality["false_positives_per_empty_image"]
            <= baseline_quality["false_positives_per_empty_image"]
            + args.max_empty_fp_increase
        )
        summary = {
            "epoch": epoch_index + 1,
            "stage": stage,
            "target_text_ratio": ratio,
            "global_step": global_step,
            "seconds": elapsed,
            "learning_rates": {
                "head": optimizer.param_groups[0]["lr"],
                "detector": optimizer.param_groups[1]["lr"],
                "backbone": optimizer.param_groups[2]["lr"],
            },
            "grad_scaler_scale": scaler.get_scale() if scaler is not None else None,
            "passes_detector_guardrails": passes_guardrails,
            **training_epoch_summary(
                loss_sum=running_loss,
                loss_square_sum=running_loss_square,
                batches=batches,
                images=images,
                seconds=elapsed,
                grad_norm_sum=running_grad_norm,
                max_grad_norm=maximum_grad_norm,
                skipped_optimizer_steps=skipped_optimizer_steps,
                peak_memory_mb=peak_memory_mb,
            ),
            **{f"validation_{name}": value for name, value in validation.items()},
            **{f"validation_{name}": value for name, value in quality.items()},
        }
        log_event("epoch", summary)
        append_log_event(training_log_path, "epoch", summary)
        matched_dice = float(quality.get("matched_dice", 0.0))
        if passes_guardrails and matched_dice > best_dice:
            best_dice = matched_dice
            save_base_checkpoint(model, output_dir / "best")
        if (epoch_index + 1) % args.save_every_epochs == 0:
            checkpoint_dir = output_dir / f"checkpoint-{global_step:06d}"
            save_base_checkpoint(model, checkpoint_dir)
            save_training_state(
                checkpoint_dir,
                optimizer,
                scheduler,
                epoch=epoch_index + 1,
                global_step=global_step,
                scaler=scaler,
            )
            atomic_write_text(
                output_dir / "last_checkpoint", checkpoint_dir.name + "\n"
            )
    save_base_checkpoint(model, output_dir / "final")


if __name__ == "__main__":
    main()
