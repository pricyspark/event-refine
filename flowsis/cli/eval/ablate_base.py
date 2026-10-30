from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import torch
from datasets import Dataset

from flowsis.artifacts import atomic_write_text
from flowsis.base_checkpoint import load_base
from flowsis.cli.common import dataset_from_args, ensure_split_exists
from flowsis.cli.eval.evaluate_base import evaluate_model
from flowsis.cli.train.train_base import build_dataloader, build_dataset
from flowsis.utils import get_device, set_seed

CONDITIONS = ("baseline", "oracle_text", "no_text", "no_box_prior", "stride8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run same-checkpoint set-mask interventions. Retrain these architecture "
            "choices separately for causal comparisons."
        )
    )
    parser.add_argument("--model-path", type=Path, default=Path("outputs/base/final"))
    parser.add_argument("--dataset-path", default="data/segmentation-dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS
    )
    parser.add_argument(
        "--output-path", type=Path, default=Path("outputs/base-ablation.json")
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--detection-threshold", type=float)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.set_defaults(oracle_target_text=False)
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
    loader = build_dataloader(
        build_dataset(split, data_args, training=False), data_args, training=False
    )
    original_conditioning = model.head.query_text_fusion.conditioning
    original_box_prior = model.head.use_box_prior
    original_detail = model.head.pixel_decoder.use_detail_features
    results = {}
    for condition in args.conditions:
        model.head.query_text_fusion.conditioning = original_conditioning
        model.head.use_box_prior = original_box_prior
        model.head.pixel_decoder.use_detail_features = original_detail
        args.oracle_target_text = condition == "oracle_text"
        if condition == "no_text":
            model.head.query_text_fusion.conditioning = "none"
        elif condition == "no_box_prior":
            model.head.use_box_prior = False
        elif condition == "stride8":
            model.head.pixel_decoder.use_detail_features = False
        results[condition] = evaluate_model(model, loader, args)
    output = {
        "comparison_type": "same_checkpoint_intervention",
        "warning": "Retraining is required for causal architectural comparisons.",
        "conditions": results,
    }
    text = json.dumps(output, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    atomic_write_text(args.output_path, text)


if __name__ == "__main__":
    main()
