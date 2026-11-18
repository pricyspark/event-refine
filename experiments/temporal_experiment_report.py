"""Controlled temporal experiment analysis, prepared with Codex assistance."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from flowsis.artifacts import atomic_write_text

if TYPE_CHECKING:
    from experiments.run_temporal_experiments import Job


def frame_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["video_id"], row["frame_index"], row["track_id"]


def summarize_failures(rows: list[dict[str, str]]) -> dict[str, int | float | None]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["video_id"], row["clip_start"], row["track_id"])].append(row)
    delays = []
    never_correct = longest = 0
    for group in groups.values():
        group.sort(key=lambda row: int(row["position"]))
        first = next(
            (
                int(row["position"])
                for row in group
                if float(row["temporal_dice"]) >= 0.5
            ),
            None,
        )
        if first is None:
            never_correct += 1
        else:
            delays.append(first - int(group[0]["position"]))
        run = 0
        previous = -2
        for row in group:
            position = int(row["position"])
            if position != previous + 1:
                run = 0
            run = run + 1 if float(row["temporal_dice"]) < 0.5 else 0
            longest = max(longest, run)
            previous = position
    return {
        "frames_dice_below_0_5": sum(float(row["temporal_dice"]) < 0.5 for row in rows),
        "longest_low_dice_run_frames": longest,
        "target_clip_groups": len(groups),
        "target_clip_groups_never_correct": never_correct,
        "median_frames_from_first_scored_target_to_correct_successful_only": (
            statistics.median(delays) if delays else None
        ),
    }


def write_report(
    output: Path,
    jobs: list[Job],
    completed: set[str],
    failures: dict[str, str],
    status: str,
) -> None:
    records = []
    baseline: dict[tuple[str, str, str], float] | None = None
    cohort: set[tuple[str, str, str]] | None = None
    for job in jobs:
        if (
            job["kind"] != "eval"
            or job["name"] not in completed
            or job["candidate"] == "oracle"
        ):
            continue
        directory = Path(job["output"])
        result = json.loads((directory / "summary.json").read_text())
        with (directory / "frames.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        keys = [frame_key(row) for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Repeated scored frames: {job['name']}")
        if cohort is None:
            cohort = set(keys)
        elif set(keys) != cohort:
            raise ValueError(f"Scored frames differ across experiments: {job['name']}")
        if job["candidate"] != "strong-base":
            scores = {frame_key(row): float(row["base_dice"]) for row in rows}
            if baseline is None:
                baseline = scores
            elif any(abs(scores[key] - baseline[key]) > 1e-4 for key in scores):
                raise ValueError(
                    f"Stateless base scores changed across experiments: {job['name']}"
                )
        metrics = result["metrics"]["runtime_single_target"]
        overall = metrics["overall"]
        coverage = result["runtime_coverage"]
        empty = coverage["annotated_empty_frames"]
        records.append(
            {
                "job": job["name"],
                "candidate": job["candidate"],
                "policy": job["policy"],
                "length": job["length"],
                "refinement": job["refinement"],
                "frames": overall["count"],
                "base_dice": overall["base_dice"],
                "output_dice": overall["temporal_dice"],
                "delta_dice": overall["delta_dice"],
                "equal_video_delta_dice": metrics["equal_video_delta_dice"],
                "post_start_delta_dice": (
                    metrics["subsequent_frames"]["delta_dice"]
                    if metrics["subsequent_frames"]
                    else None
                ),
                "empty_frames": empty,
                "selected_fp_fraction_on_empty": (
                    coverage["selected_on_annotated_empty"] / empty if empty else None
                ),
                **summarize_failures(rows),
            }
        )
    if records:
        with (output / "comparisons.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    lines = [
        "# Temporal experiment report",
        "",
        "Prepared with Codex assistance.",
        "",
        f"Status: **{status}**",
        "",
        "All runtime recipes use the same scored validation frames. Single-frame tails are retained. Test data is not evaluated.",
        "",
        "| Candidate | Policy | Length | Refinement | Frames | Base Dice | Output Dice | Delta (pp) | Equal-video delta (pp) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        lines.append(
            f"| {row['candidate']} | {row['policy']} | {row['length']} | {row['refinement']} | {row['frames']} | {row['base_dice']:.4f} | {row['output_dice']:.4f} | {100*row['delta_dice']:+.2f} | {100*row['equal_video_delta_dice']:+.2f} |"
        )
    lines += [
        "",
        "## Interpreting the experiments",
        "",
        "- current: original tracker confirmation and refinement behavior.",
        "- fallback: return the strongest current base mask when no confirmed selection exists; do not promote tentative tracks internally.",
        "- history: fallback plus seed new-track state with base logits/zero hidden state; begin refinement only with genuine previous history.",
        "- identity: disable propagation/residual mask correction. Other model components and tracking/selection remain.",
        "- strong-base: stateless top-score output from the separately selected base; its paired delta column is zero by construction, so compare its Output Dice with the runtime recipes.",
        "- Oracle-matched diagnostics are in oracle-* directories; their GT-assisted scores are not overall runtime results.",
        "- Empty coverage is reported explicitly. No empty frames means false-positive tradeoffs are unmeasured.",
        "- comparisons.csv includes first-correct delay (Dice >= 0.5), never-correct target/snippet counts, and low-Dice runs. Delay medians describe successful groups only; inspect failure counts too.",
        "- runtime_diagnostics.csv records every input frame: selected class/query/track, confidence, fallback use, raw image path and tracker state. Match it with frames.csv to inspect failures.",
        "- Frame-position plots retain counts; later positions can have fewer snippets. Per-video summaries are in each evaluation folder.",
        "- These are one-seed, within-recording validation results. Wall times include decoding and duplicate baseline inference; they are not deployment latency benchmarks.",
        "- This suite tests policies and optional longer training; simulated association-error training and full multi-object tracking metrics are not implemented here.",
        "",
    ]
    candidates = [row for row in records if row["candidate"] != "strong-base"]
    if candidates:
        best = max(candidates, key=lambda row: row["output_dice"])
        lines += [
            f"Best observed runtime recipe by frame-weighted validation Dice: **{best['job']}** ({best['output_dice']:.4f}). This is exploratory validation selection, not an independent test claim.",
            "",
        ]
    lines += ["## Stage status", "", f"Completed: {len(completed)} / {len(jobs)}."]
    lines.extend(f"- FAILED {name}: {error}" for name, error in failures.items())
    atomic_write_text(output / "REPORT.md", "\n".join(lines) + "\n")
