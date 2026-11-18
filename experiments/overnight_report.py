"""Overnight results and model provenance, prepared with Codex assistance."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import TypedDict

from experiments.overnight_support import DatasetAudit


class RunState(TypedDict, total=False):
    status: str
    outcomes: list[str]
    detector_history: list[tuple[int, int, float]]
    detector: str
    base: str
    temporal: str
    standalone_base: str
    dataset_audit: DatasetAudit


def represented_ap(path: Path) -> float:
    metrics = json.loads(path.read_text())
    aps = [
        float(row["ap"])
        for row in metrics["classes"].values()
        if row["ground_truth"] > 0
    ]
    if not aps or not all(math.isfinite(value) for value in aps):
        raise ValueError(f"No finite represented-class AP in {path}")
    return statistics.mean(aps)


def write_report(output: Path, state: RunState) -> None:
    (output / "summary.json").write_text(json.dumps(state, indent=2) + "\n")
    lines = [
        "# Overnight training report",
        "",
        "Prepared with Codex assistance.",
        "",
        f"Status: **{state['status']}**",
        "",
        f"Detector: `{state.get('detector', 'pending')}`",
        f"Base paired with temporal: `{state.get('base', 'pending')}`",
        f"Standalone base selected by validation: `{state.get('standalone_base', 'pending')}`",
        f"Temporal: `{state.get('temporal', 'pending')}`",
        "",
        "## Detector validation loss",
        "",
        "| Epoch | Loss |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {epoch} | {loss:.6f} |"
        for epoch, _, loss in state.get("detector_history", [])
    )
    lines += [
        "",
        "## Detector AP candidates",
        "",
        "| Candidate | IoU | Represented-class macro AP | Micro AP |",
        "| --- | ---: | ---: | ---: |",
    ]
    for path in sorted(output.glob("detector/*/metrics.json")):
        result = json.loads(path.read_text())
        lines.append(
            f"| {path.parent.name} | {result['iou_threshold']} | {represented_ap(path):.4f} | {result['micro_ap']:.4f} |"
        )
    lines += [
        "",
        "## Base validation",
        "",
        "| Run/view | Matched Dice | Oracle Dice | Mask AP50 | Box AP50 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for path in sorted(output.glob("base/*/validation*.json")):
        result = json.loads(path.read_text())
        lines.append(
            f"| {path.parent.name}/{path.stem} | "
            + " | ".join(
                f"{result[key]:.4f}"
                for key in ("matched_dice", "oracle_dice", "mask_ap50", "box_ap50")
            )
            + " |"
        )
    for path in sorted(output.glob("temporal/eval-*/summary.json")):
        result = json.loads(path.read_text())
        lines += [
            "",
            f"## Temporal {path.parent.name}",
            "",
            "| Mode | Position | Count | Base Dice | Temporal Dice | Paired delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for mode, metrics in result["metrics"].items():
            for position, row in metrics["by_position"].items():
                if int(position) not in {0, 1, 2, 3, 4, 7, 15, 31, 63, 127, 255}:
                    continue
                lines.append(
                    f"| {mode} | {position} | {row['count']} | {row['base_dice']:.4f} | {row['temporal_dice']:.4f} | {row['delta_dice']:+.4f} |"
                )
            lines += [
                "",
                f"{mode}: equal-video mean paired Dice delta = **{metrics['equal_video_delta_dice']:+.4f}**. All positions and recording-level results remain in the JSON/CSV.",
            ]
        lines += ["", f"Frame-level CSV, per-video JSON and plot: `{path.parent}`."]
    lines += [
        "",
        "## Interpretation",
        "",
        "- Detector candidates are compared by macro AP50 over represented validation classes; AP75 is also saved.",
        "- Base best is guardrail-qualified matched Dice. Oracle/matched Dice gaps can indicate classification or query acceptance limits.",
        "- The no-overlap control changes base augmentation only; both bases inherit the same synthetically trained detector.",
        "- Oracle temporal results isolate mask refinement with GT matching. Runtime includes actual selection, tracking and confirmation warm-up; frame 0 can therefore be a miss.",
        "- Frame positions compare the SAME frames between models. Raw differences from frame 0 also reflect scene changes and different video support.",
        "- Check dataset_audit.json for actual class support, capture/video sharing, multi-tool and empty-scene coverage. Test cohorts are opt-in and never select checkpoints.",
        "- Real multi-tool priority has not been added to sampling/losses. Rotation augmentation is explicitly −22.5° to +22.5°.",
        "- Identity-refinement evaluation zeros mask propagation/residual corrections while retaining the same tracker, trained visibility branch and selector. It isolates mask correction, not every learned temporal component.",
        "- Temporal remains paired with the main base even if the optional control becomes the stronger standalone base.",
        "- No improvement or completion time is guaranteed; inspect paired deltas and class support before adopting a model.",
        "",
        "## Stage outcomes",
        "",
    ]
    lines.extend(f"- {item}" for item in state.get("outcomes", []))
    audit = state.get("dataset_audit")
    if audit:
        lines += [
            "",
            "## Data coverage and warnings",
            "",
            "| Split | Rows | Videos | Multi-tool | Background |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        lines.extend(
            f"| {name} | {row['rows']} | {len(row['video_ids'])} | {row['multi_tool_rows']} | {row['background_rows']} |"
            for name, row in audit["splits"].items()
        )
        lines += ["", *[f"- {warning}" for warning in audit["warnings"]]]
    lines += [
        "",
        "## Starting points for further testing",
        "",
        "Optimization curves: `learning_curves.png`. PR curves and temporal frame-position plots are saved alongside each evaluation.",
        "",
    ]
    main_metrics = output / "base/overlap25/validation.json"
    if main_metrics.is_file():
        result = json.loads(main_metrics.read_text())
        weak = sorted(
            result.get("per_class", {}).items(),
            key=lambda item: item[1]["matched_dice"],
        )[:3]
        if weak:
            weak_labels = ", ".join(
                f"{name} ({metrics['matched_dice']:.3f})" for name, metrics in weak
            )
            lines.append(f"- Review low-Dice classes first: {weak_labels}.")
        if result["oracle_dice"] - result["matched_dice"] > 0.05:
            lines.append(
                "- Inspect class/query selection and thresholds: oracle masks exceed matched Dice by more than 0.05."
            )
    lines += [
        "- Compare learned versus identity refinement on the same videos and frame positions; retain cold starts and counts.",
        "- Repeat promising control comparisons with another seed before claiming augmentation gains.",
        "- Add independent real overlaps and empty scenes wherever the audit shows gaps; multiple tools do not necessarily occlude one another.",
        "- Freeze the recipe before evaluating untouched test_known/test_unseen cohorts. Current test_unseen metrics include known companions unless filtered separately.",
        "",
        "## Saving and loading",
        "",
        "Save every file listed in MODEL_MANIFEST.json. Base inference needs the complete base directory (base.pt plus detector weights/config/preprocessor), not base.pt alone.",
        "Temporal inference additionally needs temporal-best.pt, its matching main base, and RAFT weights. Preserve provenance.json, commands.jsonl and dataset split manifests for reproduction.",
        "Load bases with flowsis.base_checkpoint.load_base; refiners with flowsis.temporal_checkpoint.load_temporal_checkpoint. The standalone control may differ from the base paired with temporal.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def model_manifest(output: Path, state: dict, flow_checkpoint: str) -> None:
    manifest: dict = {"flow_checkpoint": flow_checkpoint, "models": {}}
    for role in ("detector", "base", "standalone_base", "temporal"):
        if not state.get(role):
            continue
        source = Path(state[role]).resolve()
        files = (
            [source]
            if source.is_file()
            else [
                source / name
                for name in (
                    "config.json",
                    "preprocessor_config.json",
                    "flowsis_detector.json",
                    "model.safetensors",
                    "base.pt",
                )
                if (source / name).is_file()
            ]
        )
        manifest["models"][role] = {
            "path": str(source),
            "files": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
            },
        }
    manifest["temporal_base_pair"] = state.get("base")
    flow_path = Path(flow_checkpoint)
    candidates = (
        [flow_path]
        if flow_path.is_file()
        else sorted(
            (Path.home() / ".cache/torch/hub/checkpoints").glob(
                f"raft_small-{flow_checkpoint}*.ckpt"
            )
        )
    )
    manifest["flow_files"] = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in candidates
    }
    (output / "MODEL_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


def learning_curves(output: Path, detector_run: Path) -> None:
    """Export optimization evidence separately from held-out test claims."""

    def events(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]

    detector = [
        row
        for row in events(detector_run / "training_log.jsonl")
        if row.get("event") == "validation_epoch"
    ]
    bases = {
        name: [
            row
            for row in events(output / f"base/{name}/training_log.jsonl")
            if row.get("event") == "epoch"
        ]
        for name in ("overlap25", "overlap-off")
    }
    if not detector and not any(bases.values()):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    if detector:
        axes[0].plot(
            [int(row["epoch"]) + 1 for row in detector],
            [row["loss"] for row in detector],
        )
    axes[0].set(title="Detector validation loss", xlabel="Epoch", ylabel="Loss")
    for name, rows in bases.items():
        for ax, metric in zip(axes[1:], ("matched_dice", "mask_ap50"), strict=True):
            usable = [row for row in rows if f"validation_{metric}" in row]
            if usable:
                ax.plot(
                    [row["epoch"] for row in usable],
                    [row[f"validation_{metric}"] for row in usable],
                    label=name,
                )
                ax.legend()
    axes[1].set(
        title="Base validation Dice", xlabel="Epoch", ylabel="Matched Dice", ylim=(0, 1)
    )
    axes[2].set(
        title="Base validation mask AP50", xlabel="Epoch", ylabel="AP50", ylim=(0, 1)
    )
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=160)
    plt.close(figure)
