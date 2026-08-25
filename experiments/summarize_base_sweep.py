#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def is_stable(epoch: dict[str, Any]) -> bool:
    values = (
        epoch.get("train_loss"),
        epoch.get("train_loss_std"),
        epoch.get("mean_grad_norm"),
        epoch.get("max_grad_norm"),
    )
    return (
        all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in values
        )
        and epoch.get("skipped_optimizer_steps", 0) == 0
        and epoch.get("passes_detector_guardrails", False)
    )


def summarize_run(path: Path) -> dict[str, Any] | None:
    events = load_events(path)
    epochs = [event for event in events if event.get("event") == "epoch"]
    eligible = [epoch for epoch in epochs if is_stable(epoch)]
    if not epochs:
        return None
    has_nonfinite_failure = any(
        event.get("event") == "nonfinite_loss" for event in events
    )
    has_repeated_overflow = any(
        epoch.get("skipped_optimizer_steps", 0) > 0 for epoch in epochs[1:]
    )
    run_is_stable = bool(eligible) and not (
        has_nonfinite_failure or has_repeated_overflow
    )
    best = max(
        eligible or epochs,
        key=lambda epoch: float(epoch.get("validation_matched_dice", float("-inf"))),
    )
    full_training = [epoch for epoch in epochs if epoch.get("stage") == "all"]
    speed_epochs = full_training or epochs
    throughputs = [
        float(epoch["train_images_per_second"])
        for epoch in speed_epochs
        if isinstance(epoch.get("train_images_per_second"), (int, float))
    ]
    memories = [
        float(epoch["peak_memory_mb"])
        for epoch in speed_epochs
        if isinstance(epoch.get("peak_memory_mb"), (int, float))
    ]
    return {
        "run": path.parent.name,
        "status": "stable" if run_is_stable else "reject",
        "best_epoch": best.get("epoch"),
        "matched_dice": best.get("validation_matched_dice"),
        "box_ap50": best.get("validation_box_ap50"),
        "mask_ap50": best.get("validation_mask_ap50"),
        "empty_fp": best.get("validation_false_positives_per_empty_image"),
        "images_per_second": (
            sum(throughputs) / len(throughputs) if throughputs else None
        ),
        "peak_memory_mb": max(memories) if memories else None,
    }


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize base-training sweep logs.")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("outputs/experiments/base"),
    )
    args = parser.parse_args()
    rows = [
        summary
        for path in sorted(args.root.glob("*/training_log.jsonl"))
        if (summary := summarize_run(path)) is not None
    ]
    if not rows:
        raise SystemExit(f"No completed epoch logs found under {args.root}")
    columns = (
        "run",
        "status",
        "best_epoch",
        "matched_dice",
        "box_ap50",
        "mask_ap50",
        "empty_fp",
        "images_per_second",
        "peak_memory_mb",
    )
    print("\t".join(columns))
    for row in sorted(
        rows,
        key=lambda item: (
            item["status"] != "stable",
            -float(item["matched_dice"] or 0.0),
        ),
    ):
        print("\t".join(format_value(row[column]) for column in columns))


if __name__ == "__main__":
    main()
