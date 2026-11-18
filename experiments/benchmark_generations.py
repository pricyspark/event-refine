"""Paired model-generation benchmark, prepared with Codex assistance."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from datasets import ClassLabel, Dataset, DatasetDict, disable_progress_bars
import torch

from flowsis.base_checkpoint import load_base
from flowsis.cli.eval.evaluate_base import (
    evaluate_model_with_curves,
    write_precision_recall_curves,
)
from flowsis.cli.train.train_base import (
    build_dataloader,
    build_dataset,
    load_label_names,
)
from flowsis.data.loaders import load_local_dataset
from flowsis.data.object_records import get_object_feature_schema

CHECKPOINTS = {
    "prior_head10": "outputs/base/dfine-tool-class-epoch18-head10-v1/best",
    "prior_gentle": "outputs/base/dfine-tool-class-epoch18-head10neck3all2-v1/best",
    "current_overlap25": "outputs/overnight/coarse-v2/base/overlap25/best",
    "current_overlap_off": "outputs/overnight/coarse-v2/base/overlap-off/best",
}


def common_validation(
    prior: DatasetDict, current: DatasetDict, hashes: dict[str, str]
) -> tuple[Dataset, dict]:
    """Use shared validation images and independently verify each export's groups."""
    previous = {row["image_id"]: row for row in prior["validation"]}
    previous_train_ids = set(prior["train"]["image_id"])
    current_train_ids = set(current["train"]["image_id"])
    previous_train_groups = set(prior["train"]["sample_id"])
    current_train_groups = set(current["train"]["sample_id"])
    labels = load_label_names(current["validation"])
    old_labels = load_label_names(prior["validation"])
    indices, excluded = [], []
    records = []
    for index, row in enumerate(current["validation"]):
        image_id = row["image_id"]
        if row.get("video_id") or image_id not in previous:
            continue
        old = previous[image_id]
        # Sample IDs were renumbered: compare each group in its OWN export.
        if (
            image_id in previous_train_ids
            or image_id in current_train_ids
            or old["sample_id"] in previous_train_groups
            or row["sample_id"] in current_train_groups
        ):
            raise ValueError(f"Training exposure in common validation: {image_id}")
        expected = hashes.get(old["image_path"])
        actual = hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest()
        if expected != actual:
            excluded.append(image_id)
            continue
        current_names = [labels[obj["category"]] for obj in row["objects"]]
        prior_names = [old_labels[obj["category"]] for obj in old["objects"]]
        if current_names != prior_names:
            raise ValueError(f"Changed taxonomy assignment for {image_id}")
        indices.append(index)
        records.append(
            {
                "image_id": image_id,
                "image_path": row["image_path"],
                "image_sha256": actual,
                "prior_sample_id": old["sample_id"],
                "current_sample_id": row["sample_id"],
                "labels": current_names,
                "mask_sha256": {
                    obj["mask_path"]: hashlib.sha256(
                        Path(obj["mask_path"]).read_bytes()
                    ).hexdigest()
                    for obj in row["objects"]
                },
            }
        )
    if not indices:
        raise ValueError("No hash-verified shared validation frames.")
    selected = current["validation"].select(indices)
    metadata = {
        "images": len(selected),
        "capture_groups": len({row["sample_id"] for row in selected}),
        "class_support": dict(
            Counter(label for row in records for label in row["labels"])
        ),
        "excluded_image_hash_mismatch": excluded,
        "frames": records,
        "notes": [
            "Intersection of archived 2026-10-30 validation and current validation still images.",
            "Image contents match archived SHA-256; group exposure checked separately because Sample IDs were renumbered.",
            "Current annotations, identical preprocessing, FP32, thresholds 0.5, and canonical-name ID remapping for every model.",
            "No test images. Development comparison: both validation pools have been used for model selection.",
            "Small correlated capture-group sample; no independent-frame significance claims or attribution to a single training change.",
        ],
    }
    return selected, metadata


def remap_categories(
    source: Dataset, source_labels: tuple[str, ...], target_labels: tuple[str, ...]
) -> Dataset:
    mapping = {
        index: target_labels.index(label)
        for index, label in enumerate(source_labels)
        if label in target_labels
    }
    rows = []
    for row in source:
        objects = []
        for obj in row["objects"]:
            if obj["category"] not in mapping:
                raise ValueError(
                    f"Target class absent from checkpoint: {source_labels[obj['category']]}"
                )
            objects.append({**obj, "category": mapping[obj["category"]]})
        rows.append({**row, "objects": objects})
    features = deepcopy(source.features)
    get_object_feature_schema(features["objects"])["category"] = ClassLabel(
        names=list(target_labels)
    )
    return Dataset.from_list(rows, features=features)


def benchmark_detectors(
    output: Path, selected: Dataset, source_labels: tuple[str, ...], device: str
) -> None:
    models = {
        "prior_detector": Path(
            "outputs/detectors/dfine-tool-class-20epochs/checkpoint-001170"
        ),
        "prior_detector_final": Path(
            "outputs/detectors/dfine-tool-class-20epochs/final"
        ),
        "current_detector": Path(
            "outputs/detectors/coarse-sections-v1/checkpoint-005600"
        ),
    }
    summary = {}
    for name, checkpoint in models.items():
        config = json.loads((checkpoint / "config.json").read_text())
        labels = tuple(
            config["id2label"][str(i)] for i in range(len(config["id2label"]))
        )
        directory = output / name
        DatasetDict(
            validation=remap_categories(selected, source_labels, labels)
        ).save_to_disk(str(directory / "dataset"))
        metrics = {}
        for iou in (0.5, 0.75):
            command = [
                sys.executable,
                "-m",
                "flowsis.cli.eval.evaluate_detector",
                "--detector",
                "dfine",
                "--model",
                str(checkpoint),
                "--dataset-path",
                str(directory / "dataset"),
                "--split",
                "validation",
                "--batch-size",
                "1",
                "--image-size",
                "640",
                "--device",
                device,
                "--iou-threshold",
                str(iou),
                "--output-dir",
                str(directory / f"ap{iou}"),
            ]
            with (directory / f"ap{iou}.log").open("w") as handle:
                subprocess.run(
                    command,
                    env=dict(
                        os.environ,
                        OMP_NUM_THREADS="2",
                        MKL_NUM_THREADS="2",
                        MPLBACKEND="Agg",
                    ),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            metrics[str(iou)] = json.loads(
                (directory / f"ap{iou}/metrics.json").read_text()
            )
        summary[name] = {"checkpoint": str(checkpoint), "metrics": metrics}
        (output / "detector_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print("Detector benchmark complete:", name, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/presentation/coarse-v2/benchmark"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    disable_progress_bars()
    torch.set_num_threads(args.threads)
    prior = load_local_dataset(
        "experiments/baselines/2026-10-30-tool-class/frozen_dataset"
    )
    current = load_local_dataset("../flowsis-data/exports/combined-sections-coarse")
    hashes = {
        row["path"]: row["sha256"]
        for row in json.loads(
            Path(
                "experiments/baselines/2026-10-30-tool-class/external_assets.sha256.json"
            ).read_text()
        )
    }
    selected, metadata = common_validation(prior, current, hashes)
    source_labels = load_label_names(current["validation"])
    (args.output_dir / "cohort.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Paired cohort: {metadata['images']} images, {metadata['capture_groups']} groups, {metadata['class_support']}",
        flush=True,
    )
    evaluation_args = SimpleNamespace(
        detection_threshold=0.5,
        mask_threshold=0.5,
        oracle_target_text=False,
        amp=False,
        max_images=None,
    )
    loader_args = SimpleNamespace(
        image_size=640, batch_size=1, num_workers=0, device=args.device, seed=42
    )
    summary = {}
    for name, path in CHECKPOINTS.items():
        print(f"Evaluating {name}: {path}", flush=True)
        model = load_base(path, device=args.device, use_amp=False, detector_amp=False)
        mapped = remap_categories(selected, source_labels, model.label_names)
        loader = build_dataloader(
            build_dataset(mapped, loader_args, training=False),
            loader_args,
            training=False,
        )
        started = time.monotonic()
        metrics, curves = evaluate_model_with_curves(model, loader, evaluation_args)
        directory = args.output_dir / name
        directory.mkdir()
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        write_precision_recall_curves(
            directory / "pr",
            curves,
            {"box AP50": metrics["box_ap50"], "mask AP50": metrics["mask_ap50"]},
        )
        summary[name] = {
            "checkpoint": path,
            "labels": model.label_names,
            "device": args.device,
            "seconds": time.monotonic() - started,
            "metrics": metrics,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(
            name,
            "matched Dice",
            metrics["matched_dice"],
            "mask AP50",
            metrics["mask_ap50"],
            flush=True,
        )
        del model, loader
        gc.collect()
    benchmark_detectors(args.output_dir, selected, source_labels, args.device)
    print(f"Benchmark complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
