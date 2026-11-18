"""Paired sequential temporal evaluation, prepared with Codex assistance."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import torch
import torch.nn.functional as F

from flowsis.base import FlowSISBase
from flowsis.base_checkpoint import load_base
from flowsis.cli.train.train_temporal import (
    _base_sequence,
    _label_names,
    select_training_observations,
)
from flowsis.data.loaders import load_local_dataset, temporal_dataset_view
from flowsis.data.temporal import ClipSpec, TemporalClipDataset, collate_temporal_clips
from flowsis.pretrained.flow import PTLFlowEstimator, load_flow_estimator
from flowsis.temporal import TemporalRefinementBranch, resize_flow
from flowsis.temporal_checkpoint import load_temporal_checkpoint
from flowsis.temporal_model import FlowSISTemporal


class FrameMetric(TypedDict):
    mode: str
    video_id: str
    clip_start: int
    position: int
    frame_index: int
    track_id: str
    base_dice: float
    temporal_dice: float
    delta_dice: float
    base_iou: float
    temporal_iou: float
    delta_iou: float


class ClipSample(TypedDict):
    frames: torch.Tensor
    masks: torch.Tensor
    labels: torch.Tensor
    mask_valid: torch.Tensor
    visible: torch.Tensor
    ignore: torch.Tensor
    track_ids: tuple[str, ...]


class ModeSummary(TypedDict):
    overall: dict[str, float]
    subsequent_frames: dict[str, float] | None
    per_video: dict[str, dict[str, float]]
    by_position: dict[str, dict[str, float]]
    equal_video_delta_dice: float


def evaluation_clips(
    dataset: TemporalClipDataset,
    length: int,
    clips_per_video: int,
    *,
    all_frames: bool = False,
) -> list[ClipSpec]:
    """Evenly select disjoint chunks without crossing export interval boundaries."""
    result = []
    for video, indices in sorted(dataset.frame_indices.items()):
        runs: list[list[int]] = []
        for index in sorted(indices):
            if (
                not runs
                or index != runs[-1][-1] + 1
                or dataset.sequence_ids[video][index]
                != dataset.sequence_ids[video][runs[-1][-1]]
            ):
                runs.append([])
            runs[-1].append(index)
        candidates = [
            ClipSpec(video, run[offset], min(length or len(run), len(run) - offset), 1)
            for run in runs
            for offset in range(0, len(run), length or len(run))
            if all_frames or len(run) - offset >= 2
        ]
        if all_frames:
            result.extend(candidates)
            continue
        count = min(clips_per_video, len(candidates))
        positions = np.linspace(0, len(candidates) - 1, count, dtype=int)
        result.extend(candidates[int(index)] for index in positions)
    return result


def mask_metrics(
    logits: torch.Tensor | None,
    target: torch.Tensor,
    ignore: torch.Tensor,
    threshold: float,
) -> tuple[float, float]:
    if logits is None:
        return 0.0, 0.0  # Missing/wrong-class runtime selections count as misses.
    values = F.interpolate(
        logits.reshape(1, 1, *logits.shape[-2:]).float(),
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite mask logits in temporal evaluation.")
    valid = ~ignore.bool()
    prediction = (values.sigmoid() >= threshold) & valid
    truth = target.bool() & valid
    intersection = float((prediction & truth).sum())
    total = float(prediction.sum() + truth.sum())
    union = float((prediction | truth).sum())
    return (2 * intersection + 1) / (total + 1), (intersection + 1) / (union + 1)


def frame_record(
    mode: str,
    spec: ClipSpec,
    position: int,
    track: str,
    base: tuple[float, float],
    temporal: tuple[float, float],
) -> FrameMetric:
    return {
        "mode": mode,
        "video_id": spec.video_id,
        "clip_start": spec.start,
        "position": position,
        "frame_index": spec.frame_indices[position],
        "track_id": track,
        "base_dice": base[0],
        "temporal_dice": temporal[0],
        "delta_dice": temporal[0] - base[0],
        "base_iou": base[1],
        "temporal_iou": temporal[1],
        "delta_iou": temporal[1] - base[1],
    }


@torch.inference_mode()
def oracle_clip(
    base: FlowSISBase,
    flow: PTLFlowEstimator,
    temporal: TemporalRefinementBranch,
    sample: ClipSample,
    spec: ClipSpec,
) -> list[FrameMetric]:
    """GT query/track matching isolates refinement from runtime association."""
    batch = collate_temporal_clips([sample])
    frames = batch["frames"].to(base.device)
    track_count = batch["labels"].shape[-1]
    if track_count == 0:
        return []
    previous_logits = previous_hidden = previous_classes = previous_features = None
    rows = []
    for position in range(spec.length):
        output = _base_sequence(
            base, frames[:, position : position + 1], train_head=False
        )
        logits, scores, classes, observed = select_training_observations(
            output,
            batch["labels"][:, position : position + 1].to(base.device),
            batch["boxes"][:, position : position + 1].to(base.device),
            batch["base_valid"][:, position : position + 1].to(base.device),
        )
        logits, scores = logits[0, 0], scores[0, 0]
        classes, observed = classes[0, 0], observed[0, 0]
        features = output.pixel_features[:, 0].expand(track_count, -1, -1, -1)
        if position == 0:
            refined = torch.where(
                observed[:, None, None, None], logits, temporal.missing_base_logit
            )
            previous_hidden = logits.new_zeros(
                (track_count, temporal.hidden_channels, *logits.shape[-2:])
            )
            previous_classes = classes
        else:
            assert previous_classes is not None and previous_logits is not None
            assert previous_hidden is not None and previous_features is not None
            agreement = 1 - 0.5 * (classes - previous_classes).abs().sum(-1)
            backward = flow(frames[:, position], frames[:, position - 1])
            result = temporal(
                frames[:, position].expand(track_count, -1, -1, -1),
                frames[:, position - 1].expand(track_count, -1, -1, -1),
                logits,
                previous_logits,
                resize_flow(backward, logits.shape[-2:]).expand(
                    track_count, -1, -1, -1
                ),
                current_features=features,
                previous_features=previous_features,
                previous_hidden=previous_hidden,
                observation_valid=observed,
                observation_score=scores,
                class_agreement=torch.where(observed, agreement, 0.0),
                frame_delta=logits.new_full((track_count,), spec.stride),
            )
            refined, previous_hidden = result.final_logits, result.hidden_state
            previous_classes = torch.where(observed[:, None], classes, previous_classes)
        for track in range(track_count):
            target = batch["masks"][0, position, track].to(base.device)
            ignore = batch["ignore"][0, position].to(base.device)
            if (
                not batch["mask_valid"][0, position, track]
                or not (target.bool() & ~ignore.bool()).any()
            ):
                continue
            rows.append(
                frame_record(
                    "oracle_matched",
                    spec,
                    position,
                    sample["track_ids"][track],
                    mask_metrics(
                        logits[track] if observed[track] else None,
                        target,
                        ignore,
                        base.mask_threshold,
                    ),
                    mask_metrics(refined[track], target, ignore, base.mask_threshold),
                )
            )
        previous_logits, previous_features = refined, features
    return rows


@torch.inference_mode()
def runtime_clip(
    model: FlowSISTemporal,
    sample: ClipSample,
    spec: ClipSpec,
    *,
    diagnostics: list[dict[str, object]] | None = None,
    background_valid: list[bool] | None = None,
    image_paths: Sequence[str] | None = None,
) -> list[FrameMetric]:
    """Compare stateless top-score base selection with actual temporal selection."""
    model.reset()
    rows = []
    for position, rgb in enumerate(sample["frames"]):
        bgr = (rgb.permute(1, 2, 0).numpy()[..., ::-1] * 255).round().astype(np.uint8)
        bgr = np.ascontiguousarray(bgr)
        prediction = model.base.infer_frame(bgr).instances
        refined = model.infer_frame(bgr)  # No GT enters runtime inference.
        valid = sample["mask_valid"][position] & sample["visible"][position]
        targets = torch.nonzero(valid, as_tuple=False).flatten()
        if diagnostics is not None:
            selected = refined.selected
            chosen = (
                int(prediction.scores.argmax()) if prediction.scores.numel() else None
            )
            query_indices = getattr(prediction, "query_indices", None)
            diagnostics.append(
                {
                    "video_id": spec.video_id,
                    "clip_start": spec.start,
                    "frame_index": spec.frame_indices[position],
                    "position": position,
                    "visible_targets": targets.numel(),
                    "annotated_empty": bool(
                        background_valid and background_valid[position]
                    ),
                    "target_label": (
                        int(sample["labels"][position, int(targets[0])])
                        if targets.numel() == 1
                        else None
                    ),
                    "base_label": (
                        int(prediction.labels[chosen]) if chosen is not None else None
                    ),
                    "base_score": (
                        float(prediction.scores[chosen]) if chosen is not None else None
                    ),
                    "selected_label": selected.label if selected else None,
                    "selected_query": selected.index if selected else None,
                    "selected_track_id": selected.track_id if selected else None,
                    "selected_score": selected.score if selected else None,
                    "used_base_fallback": getattr(refined, "used_base_fallback", False),
                    "image_path": (
                        image_paths[position] if image_paths is not None else ""
                    ),
                    "base_query": (
                        int(query_indices[chosen])
                        if chosen is not None and query_indices is not None
                        else None
                    ),
                    "tracks": json.dumps(
                        [
                            {
                                "id": track.track_id,
                                "label": track.label,
                                "query": track.last_detection_index,
                                "hits": track.hits,
                                "misses": track.misses,
                                "confirmed": track.is_confirmed,
                                "score": track.score,
                                "box": track.box,
                            }
                            for track in getattr(refined, "tracks", ())
                        ]
                    ),
                }
            )
        if targets.numel() != 1:
            continue  # A unique visible target is required for this policy comparison.
        track = int(targets[0])
        target = sample["masks"][position, track].to(model.device)
        ignore = sample["ignore"][position].to(model.device)
        if not (target.bool() & ~ignore.bool()).any():
            continue
        label = int(sample["labels"][position, track])
        base_logits = None
        if prediction.scores.numel():
            chosen = int(prediction.scores.argmax())
            if int(prediction.labels[chosen]) == label:
                base_logits = prediction.mask_logits[chosen]
        temporal_logits = None
        if (
            refined.selected is not None
            and refined.selected.label == label
            and refined.selected.track_id is not None
        ):
            temporal_logits = refined.logits_by_track.get(refined.selected.track_id)
        rows.append(
            frame_record(
                "runtime_single_target",
                spec,
                position,
                sample["track_ids"][track],
                mask_metrics(base_logits, target, ignore, model.base.mask_threshold),
                mask_metrics(
                    temporal_logits, target, ignore, model.base.mask_threshold
                ),
            )
        )
    return rows


@torch.inference_mode()
def stateless_clip(
    base: FlowSISBase, sample: ClipSample, spec: ClipSpec
) -> list[FrameMetric]:
    rows = []
    for position, rgb in enumerate(sample["frames"]):
        bgr = np.ascontiguousarray(
            (rgb.permute(1, 2, 0).numpy()[..., ::-1] * 255).round().astype(np.uint8)
        )
        prediction = base.infer_frame(bgr).instances
        targets = torch.nonzero(
            sample["mask_valid"][position] & sample["visible"][position], as_tuple=False
        ).flatten()
        if targets.numel() != 1:
            continue
        track = int(targets[0])
        target = sample["masks"][position, track].to(base.device)
        ignore = sample["ignore"][position].to(base.device)
        if not (target.bool() & ~ignore.bool()).any():
            continue
        logits = None
        if prediction.scores.numel():
            query = int(prediction.scores.argmax())
            if int(prediction.labels[query]) == int(sample["labels"][position, track]):
                logits = prediction.mask_logits[query]
        metrics = mask_metrics(logits, target, ignore, base.mask_threshold)
        rows.append(
            frame_record(
                "runtime_single_target",
                spec,
                position,
                sample["track_ids"][track],
                metrics,
                metrics,
            )
        )
    return rows


def summarize_frames(rows: list[FrameMetric]) -> dict[str, ModeSummary]:
    def mean_metrics(group: list[FrameMetric]) -> dict[str, float]:
        values = [cast(Mapping[str, object], row) for row in group]
        return {
            "count": len(group),
            "videos": len({row["video_id"] for row in group}),
            "delta_dice_std": statistics.pstdev(row["delta_dice"] for row in group),
            **{
                key: sum(float(cast(float, row[key])) for row in values) / len(group)
                for key in (
                    "base_dice",
                    "temporal_dice",
                    "delta_dice",
                    "base_iou",
                    "temporal_iou",
                    "delta_iou",
                )
            },
            "improved_fraction": sum(row["delta_dice"] > 1e-6 for row in group)
            / len(group),
            "worsened_fraction": sum(row["delta_dice"] < -1e-6 for row in group)
            / len(group),
        }

    result: dict[str, ModeSummary] = {}
    for mode in sorted({row["mode"] for row in rows}):
        selected = [row for row in rows if row["mode"] == mode]
        videos: dict[str, list[FrameMetric]] = defaultdict(list)
        positions: dict[int, list[FrameMetric]] = defaultdict(list)
        for row in selected:
            videos[row["video_id"]].append(row)
            positions[row["position"]].append(row)
        result[mode] = {
            "overall": mean_metrics(selected),
            "subsequent_frames": (
                mean_metrics([r for r in selected if r["position"] > 0])
                if any(r["position"] > 0 for r in selected)
                else None
            ),
            "per_video": {key: mean_metrics(group) for key, group in videos.items()},
            "by_position": {
                str(key): mean_metrics(group)
                for key, group in sorted(positions.items())
            },
            "equal_video_delta_dice": sum(
                float(mean_metrics(group)["delta_dice"]) for group in videos.values()
            )
            / len(videos),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--temporal-model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--clips-per-video", type=int, default=6)
    parser.add_argument(
        "--mode", choices=("both", "oracle", "runtime", "base"), default="both"
    )
    parser.add_argument("--flow-model", default="raft_small")
    parser.add_argument("--flow-checkpoint", default="things")
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Evaluate every interval frame, including singleton tails; clip-length 0 means full intervals.",
    )
    parser.add_argument("--base-fallback", action="store_true")
    parser.add_argument("--skip-initial-refinement", action="store_true")
    parser.add_argument("--identity-refinement", action="store_true")
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-frames-per-video", type=int)
    args = parser.parse_args()
    if args.mode != "base" and not args.temporal_model:
        parser.error("--temporal-model is required except in base mode.")
    if any(
        value is not None and value < 1
        for value in (args.max_videos, args.max_frames_per_video)
    ):
        parser.error("Subset limits must be positive.")
    if (
        args.clip_length < 2 and not (args.all_frames and args.clip_length == 0)
    ) or args.clips_per_video < 1:
        parser.error("Clip length must be >=2 and clips per video positive.")
    base = load_base(
        args.base_model, device=args.device, use_amp=False, detector_amp=False
    )
    data = load_local_dataset(args.dataset_path)
    split = temporal_dataset_view(data[args.split])

    if args.max_videos or args.max_frames_per_video:
        videos = sorted(set(split["video_id"]))[: args.max_videos]
        counts: dict[str, int] = defaultdict(int)
        indices = []
        for index in sorted(
            range(len(split)),
            key=lambda i: (split[i]["video_id"], split[i]["frame_index"]),
        ):
            row = split[index]
            video = row["video_id"]
            if video in videos and (
                args.max_frames_per_video is None
                or counts[video] < args.max_frames_per_video
            ):
                indices.append(index)
                counts[video] += 1
        split = split.select(indices)
    if _label_names(split) != base.label_names:
        raise ValueError("Dataset vocabulary does not match the base checkpoint.")
    dataset = TemporalClipDataset(split, image_size=args.image_size, training=False)
    specs = evaluation_clips(
        dataset, args.clip_length, args.clips_per_video, all_frames=args.all_frames
    )
    if not specs:
        raise ValueError("No eligible evaluation intervals.")
    flow = temporal = runtime = None
    if args.mode != "base":
        flow = (
            load_flow_estimator(args.flow_model, args.flow_checkpoint)
            .to(args.device)
            .eval()
        )
        temporal = load_temporal_checkpoint(args.temporal_model, device=args.device)
        if args.identity_refinement:
            with torch.no_grad():
                temporal.propagation_strength.zero_()
                temporal.residual_strength.zero_()
        options = {}
        if args.base_fallback or args.skip_initial_refinement:
            options = {
                "base_fallback": args.base_fallback,
                "skip_initial_refinement": args.skip_initial_refinement,
            }
        runtime = FlowSISTemporal(base, flow, temporal, **options)
    diagnostics: list[dict[str, object]] = []
    rows: list[FrameMetric] = []
    for spec in specs:
        sample = cast(ClipSample, dataset[spec])
        if args.mode in ("both", "oracle"):
            assert flow is not None and temporal is not None
            rows.extend(oracle_clip(base, flow, temporal, sample, spec))
        if args.mode in ("both", "runtime"):
            assert runtime is not None
            empty = [
                not dataset.rows[(spec.video_id, index)].get("objects")
                and dataset.rows[(spec.video_id, index)].get("base_eligible", False)
                for index in spec.frame_indices
            ]
            rows.extend(
                runtime_clip(
                    runtime,
                    sample,
                    spec,
                    diagnostics=diagnostics,
                    background_valid=empty,
                    image_paths=[
                        str(dataset.rows[(spec.video_id, index)].get("image_path", ""))
                        for index in spec.frame_indices
                    ],
                )
            )
        if args.mode == "base":
            rows.extend(stateless_clip(base, sample, spec))
        print(
            f"Evaluated {spec.video_id} start={spec.start} length={spec.length}",
            flush=True,
        )
    if not rows:
        raise ValueError(
            "No visible mask-valid evaluation targets for the requested mode."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_mode = summarize_frames(rows)
    if diagnostics:
        with (args.output_dir / "runtime_diagnostics.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
            writer.writeheader()
            writer.writerows(diagnostics)
    summary = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "label_names": list(base.label_names),
        "clips": len(specs),
        "input_frames": sum(spec.length for spec in specs),
        "runtime_coverage": {
            "frames": len(diagnostics),
            "annotated_empty_frames": sum(
                bool(row["annotated_empty"]) for row in diagnostics
            ),
            "selected_on_annotated_empty": sum(
                bool(row["annotated_empty"]) and row["selected_label"] is not None
                for row in diagnostics
            ),
        },
        "metrics": metrics_by_mode,
        "notes": [
            "Paired comparisons on identical frames; state resets at each disjoint clip/interval boundary.",
            "Oracle mode uses GT matching and seeds frame 0 from base logits without refinement.",
            "Runtime compares top-score stateless base to the actual temporal selector, including confirmation warm-up misses.",
            "Runtime scores only frames with one visible mask-valid target; missing/wrong-class selections score zero.",
            "Later positions can have fewer videos; use counts, paired deltas and per-video results, not raw frame-0 comparisons.",
            "Frames within a recording are correlated; no frame-level significance claims are made.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if rows:
        with (args.output_dir / "frames.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for mode, metrics in metrics_by_mode.items():
        positions = metrics["by_position"]
        x = [int(key) for key in positions]
        for label, style in (("base_dice", "--"), ("temporal_dice", "-")):
            axes[0].plot(
                x,
                [v[label] for v in positions.values()],
                style,
                label=f"{mode}: {label}",
            )
        axes[1].plot(x, [v["delta_dice"] for v in positions.values()], label=mode)
        axes[2].step(
            x, [v["count"] for v in positions.values()], where="mid", label=mode
        )
    axes[2].set(
        ylabel="Scored targets/snippets", xlabel="Frame position since state reset"
    )
    axes[0].set(ylabel="Mean Dice", xlabel="Frame position since state reset")
    axes[1].set(
        ylabel="Paired temporal − base Dice", xlabel="Frame position since state reset"
    )
    axes[1].axhline(0, color="grey", linewidth=1)
    for axis in axes:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(args.output_dir / "frame_position.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
