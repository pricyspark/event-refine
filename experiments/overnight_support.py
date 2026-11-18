"""Dataset and artifact checks for the overnight recipe, with Codex assistance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict


class SplitAudit(TypedDict):
    rows: int
    base_rows: int
    still_rows: int
    video_ids: list[str]
    samples: list[str]
    tool_ids: list[str]
    classes: dict[str, int]
    multi_tool_rows: int
    background_rows: int
    temporal_videos: dict[str, int]


class DatasetAudit(TypedDict):
    labels: list[str]
    splits: dict[str, SplitAudit]
    warnings: list[str]
    temporal_length: int
    fingerprint: str


def inspect_dataset(path: Path) -> DatasetAudit:
    from datasets import DatasetDict
    from flowsis.cli.data.embed_text_prompts import taxonomy_from_dataset
    from flowsis.data.loaders import (
        base_dataset_view,
        load_local_dataset,
        temporal_dataset_view,
    )
    from flowsis.data.object_records import get_object_records
    from flowsis.data.temporal import TemporalClipDataset, DynamicClipBatchSampler

    data = load_local_dataset(path)
    if not isinstance(data, DatasetDict) or not {"train", "validation"} <= data.keys():
        raise ValueError("Export must contain nonempty train and validation splits.")
    labels = list(taxonomy_from_dataset(path))
    if not labels or any(
        Path(label).name != label or label in {".", ".."} for label in labels
    ):
        raise ValueError("Exported labels must be nonempty safe embedding filenames.")
    normalized: dict[str, list[str]] = {}
    for label in labels:
        normalized.setdefault(
            label.replace("_", "").replace(" ", "").casefold(), []
        ).append(label)
    duplicates = [values for values in normalized.values() if len(values) > 1]
    if duplicates:
        raise ValueError(
            f"Labels differ only in separators/case: {duplicates}; fix source taxonomy before export."
        )
    splits: dict[str, SplitAudit] = {}
    warnings: list[str] = []
    for name, subset in data.items():
        classes = dict.fromkeys(labels, 0)
        multi = background = 0
        for row in subset:
            objects = get_object_records(row)
            multi += len(objects) > 1
            background += not objects
            for obj in objects:
                classes[labels[int(obj["category"])]] += 1
            for field in ("image_path", "mask_bundle_path"):
                if row.get(field) and not Path(row[field]).is_file():
                    raise FileNotFoundError(f"{name}: missing {field}: {row[field]}")
            for obj in objects:
                if obj.get("mask_path") and not Path(obj["mask_path"]).is_file():
                    raise FileNotFoundError(obj["mask_path"])
        temporal = (
            temporal_dataset_view(subset)
            if "video_id" in subset.column_names
            else subset.select([])
        )
        geometry: dict[str, int] = {}
        if len(temporal):
            clips = TemporalClipDataset(temporal, image_size=64, training=False)
            for length in (2, 4, 8):
                try:
                    sampler = DynamicClipBatchSampler(
                        clips.frame_indices,
                        batch_size=1,
                        batches_per_epoch=1,
                        min_length=length,
                        max_length=length,
                        strides=(1,),
                        stride_weights=(1,),
                        positive_indices=clips.positive_indices,
                        sequence_ids=clips.sequence_ids,
                    )
                    geometry[str(length)] = len(sampler.geometries[0][3])
                except ValueError:
                    geometry[str(length)] = 0
        splits[name] = {
            "rows": len(subset),
            "base_rows": len(base_dataset_view(subset)),
            "still_rows": sum(not row.get("video_id") for row in subset),
            "video_ids": sorted(
                {str(row.get("video_id")) for row in subset if row.get("video_id")}
            ),
            "samples": sorted(
                {str(row.get("sample_id")) for row in subset if row.get("sample_id")}
            ),
            "tool_ids": sorted(
                {str(tool) for row in subset for tool in row.get("tool_ids", [])}
            ),
            "classes": classes,
            "multi_tool_rows": multi,
            "background_rows": background,
            "temporal_videos": geometry,
        }
    for name in ("train", "validation"):
        if not splits[name]["base_rows"]:
            raise ValueError(f"No base-eligible rows in {name}.")
    trained_labels = {
        labels[int(obj["category"])]
        for row in base_dataset_view(data["train"])
        for obj in get_object_records(row)
        if obj.get("class_valid", True)
        and obj.get("mask_valid", True)
        and obj.get("visible", True)
    }
    missing = sorted(set(labels) - trained_labels)
    if missing:
        raise ValueError(f"No training examples for exported labels: {missing}")
    for field in ("video_ids", "samples"):
        if set(splits["train"][field]) & set(splits["validation"][field]):
            warnings.append(
                f"Train/validation share {field}; results measure within-group development."
            )
    if "test_unseen" in splits:
        report = path / "build_report.json"
        held = (
            set(
                json.loads(report.read_text())
                .get("physical_tool_holdout", {})
                .get("unseen_tool_ids", [])
            )
            if report.is_file()
            else set()
        )
        if not held:
            raise ValueError(
                "test_unseen requires build_report.json with reserved Tool IDs."
            )
        for name in ("train", "validation", "test_known"):
            if name in splits and held & set(splits[name]["tool_ids"]):
                raise ValueError(f"Reserved unseen Tool IDs leaked into {name}.")
        warnings.append(
            "test_unseen contains scenes; unseen-target claims require separating known companions by Tool ID."
        )
    if not splits["validation"]["multi_tool_rows"]:
        warnings.append(
            "Validation has no multi-tool images; overlap effectiveness is unmeasured."
        )
    if not splits["validation"]["background_rows"]:
        warnings.append(
            "Validation has no empty scenes; background false positives are unmeasured."
        )
    absent = [
        key for key, value in splits["validation"]["classes"].items() if not value
    ]
    if absent:
        warnings.append(f"Validation classes without positives: {', '.join(absent)}.")
    length = next(
        (
            n
            for n in (8, 4, 2)
            if all(
                splits[s]["temporal_videos"].get(str(n), 0)
                for s in ("train", "validation")
            )
        ),
        0,
    )
    if not length:
        warnings.append(
            "No shared supervised train/validation clip geometry; temporal stages will be skipped."
        )
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file() and (
            file.suffix == ".arrow"
            or file.name
            in {
                "dataset_info.json",
                "build_report.json",
                "split_assignments.json",
                "asset_paths.json",
            }
        ):
            digest.update(str(file.relative_to(path)).encode())
            digest.update(file.read_bytes())
    return {
        "labels": labels,
        "splits": splits,
        "warnings": warnings,
        "temporal_length": length,
        "fingerprint": digest.hexdigest(),
    }


def prompt_suite(labels: list[str]) -> dict[str, list[str]]:
    """Bootstrap prompts; only label/template are embedded without a reviewed suite."""
    from flowsis.data.prompts import siglip2_pipeline_prompt

    return {
        label: [
            label,
            siglip2_pipeline_prompt(label),
            f"An image showing {label.replace('_', ' ')}.",
        ]
        for label in labels
    }


def check_temporal_checkpoint(path: Path) -> None:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("training_preprocessing") != "unit_rgb_to_uint8_detector_v1":
        raise ValueError(f"Missing corrected preprocessing marker: {path}")
    if not payload.get("state_dict") or not all(
        torch.isfinite(value).all() for value in payload["state_dict"].values()
    ):
        raise ValueError(f"Nonfinite or empty temporal weights: {path}")


def identity_checkpoint(source: Path, output: Path) -> None:
    import torch

    payload = torch.load(source, map_location="cpu", weights_only=True)
    # Exactly disable propagation and residual corrections while retaining the
    # trained visibility branch and the identical association/selection policy.
    payload["state_dict"]["propagation_strength"].zero_()
    payload["state_dict"]["residual_strength"].zero_()
    payload["ablation"] = "identity_mask_refinement_trained_visibility"
    torch.save(payload, output)


def validation_views(dataset_path: Path, output: Path) -> dict[str, Path]:
    from datasets import DatasetDict
    from flowsis.data.loaders import load_local_dataset, base_dataset_view

    validation = base_dataset_view(load_local_dataset(dataset_path)["validation"])
    views: dict[str, Path] = {}
    for name, video in (("stills", False), ("videos", True)):
        selected = validation.select(
            [
                i
                for i, row in enumerate(validation)
                if bool(row.get("video_id")) == video
            ]
        )
        if len(selected):
            destination = output / name
            DatasetDict(validation=selected).save_to_disk(str(destination))
            views[name] = destination
    return views


def check_base_sequence(
    base_path: Path, dataset_path: Path, output: Path, device: str, image_size: int
) -> None:
    """Compare a saved base's runtime and temporal inputs on an actual clip."""
    import torch
    from flowsis.base_checkpoint import load_base
    from flowsis.cli.eval.evaluate_temporal import evaluation_clips
    from flowsis.cli.train.train_temporal import _base_sequence
    from flowsis.data.loaders import load_local_dataset, temporal_dataset_view
    from flowsis.data.temporal import TemporalClipDataset

    base = load_base(base_path, device=device, use_amp=False, detector_amp=False).eval()
    dataset = TemporalClipDataset(
        temporal_dataset_view(load_local_dataset(dataset_path)["validation"]),
        image_size=image_size,
        training=False,
    )
    specs = evaluation_clips(dataset, 2, 1)
    if not specs:
        raise ValueError("No clip available for saved-base preprocessing parity")
    frames = dataset[specs[0]]["frames"][:1].unsqueeze(0).to(device)
    original = frames.clone()
    raw = frames[0].mul(255).round().clamp(0, 255).to(torch.uint8)
    inputs = base.detector.preprocess(
        [raw[0]], device_preprocess=base.device_preprocess
    )
    pixel_mask = inputs.get("pixel_mask")
    if isinstance(pixel_mask, torch.Tensor):
        pixel_mask = pixel_mask.to(device)
    with torch.inference_mode():
        reference = base(
            inputs["pixel_values"].to(device), pixel_mask=pixel_mask
        ).mask_logits
        sequential = _base_sequence(base, frames, train_head=False).mask_logits[:, 0]
    if not torch.isfinite(reference).all() or not torch.isfinite(sequential).all():
        raise ValueError("Nonfinite saved-base logits in preprocessing parity check")
    torch.testing.assert_close(sequential, reference, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(frames, original, rtol=0, atol=0)
    output.mkdir(parents=True, exist_ok=True)
    (output / "preprocessing_parity.json").write_text(
        json.dumps(
            {
                "base_model": str(base_path.resolve()),
                "video_id": specs[0].video_id,
                "max_abs_logit_difference": float((reference - sequential).abs().max()),
                "clip_inputs_unchanged": True,
                "note": "Input correctness diagnostic, not a learned temporal improvement.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check saved-base preprocessing parity before temporal training."
    )
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()
    check_base_sequence(
        args.base_model,
        args.dataset_path,
        args.output_dir,
        args.device,
        args.image_size,
    )
