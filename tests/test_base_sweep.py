import json
from pathlib import Path

from experiments.summarize_base_sweep import summarize_run


def write_events(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def epoch_event(epoch: int, dice: float) -> dict[str, object]:
    return {
        "event": "epoch",
        "epoch": epoch,
        "stage": "all",
        "train_loss": 1.0,
        "train_loss_std": 0.1,
        "mean_grad_norm": 2.0,
        "max_grad_norm": 3.0,
        "skipped_optimizer_steps": 0,
        "passes_detector_guardrails": True,
        "validation_matched_dice": dice,
        "validation_box_ap50": 0.8,
        "validation_mask_ap50": 0.7,
        "validation_false_positives_per_empty_image": 0.1,
        "train_images_per_second": 4.0 + epoch,
        "peak_memory_mb": 1000.0 + epoch,
    }


def test_summarize_run_selects_best_eligible_epoch(tmp_path: Path) -> None:
    path = tmp_path / "run" / "training_log.jsonl"
    path.parent.mkdir()
    write_events(path, [epoch_event(1, 0.7), epoch_event(2, 0.8)])

    summary = summarize_run(path)

    assert summary is not None
    assert summary["status"] == "stable"
    assert summary["best_epoch"] == 2
    assert summary["matched_dice"] == 0.8
    assert summary["images_per_second"] == 5.5
    assert summary["peak_memory_mb"] == 1002.0


def test_summarize_run_rejects_nonfinite_failure(tmp_path: Path) -> None:
    path = tmp_path / "run" / "training_log.jsonl"
    path.parent.mkdir()
    write_events(
        path,
        [epoch_event(1, 0.7), {"event": "nonfinite_loss", "global_step": 2}],
    )

    summary = summarize_run(path)

    assert summary is not None
    assert summary["status"] == "reject"
