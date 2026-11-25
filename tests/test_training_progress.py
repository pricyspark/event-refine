"""Training visibility and bounded validation behavior."""

import argparse
import json
import sys

import pytest
import torch

from event_flow.cli import main, train
from test_event_flow import tiny_detector, write_manifest


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "batch_size,max_steps,validation_limit,train_counts,val_counts",
    [(1, 2, 1, [1, 2], [1]), (2, 2, 3, [2, 3], [2, 3]), (2, 1, 1, [2], [1])],
)
def test_training_prints_progress_and_bounds_validation(
    tmp_path,
    capsys,
    monkeypatch,
    batch_size,
    max_steps,
    validation_limit,
    train_counts,
    val_counts,
):
    from event_flow.training import batch_clip_loss

    observed = []

    def capture_loss(model, clips, *args, **kwargs):
        loss = batch_clip_loss(model, clips, *args, **kwargs)
        observed.append((model.training, len(clips), float(loss.detach())))
        return loss

    monkeypatch.setattr("event_flow.cli.batch_clip_loss", capture_loss)
    manifest = write_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    validation = dict(
        document["sequences"][0], id="validation-sequence", split="validation"
    )
    document["sequences"].append(validation)
    manifest.write_text(json.dumps(document))
    detector_dir = tmp_path / "detector"
    tiny_detector().save_pretrained(detector_dir)
    output = tmp_path / "trained"
    train(
        argparse.Namespace(
            manifest=manifest,
            split="train",
            validation_split="validation",
            clip_length=2,
            max_gap=0.05,
            epochs=1,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=1e-4,
            temporal_learning_rate=1e-3,
            base_loss_weight=0.5,
            seed=42,
            detector_model=str(detector_dir),
            architecture="dfine",
            image_size=64,
            device="cpu",
            hidden_dim=16,
            output=output,
            command="train",
            log_every=1,
            max_validation_clips=validation_limit,
        )
    )
    records = [
        json.loads(line)
        for line in (output / "progress.jsonl").read_text().splitlines()
    ]
    events = [record["event"] for record in records]
    assert events[:3] == ["startup", "data_ready", "model_ready"]
    assert "first_clip_loaded" in events
    assert events.count("checkpoint_saved") == 2
    progress = [record for record in records if record["event"] == "progress"]
    training = [record for record in progress if record["phase"] == "train"]
    validation = [record for record in progress if record["phase"] == "validation"]
    assert [record["completed_clips"] for record in training] == train_counts
    assert all(record["total_clips"] == train_counts[-1] for record in training)
    assert [record["completed_clips"] for record in validation] == val_counts
    assert all(record["total_clips"] == validation_limit for record in validation)
    assert all(
        record["elapsed_seconds"] >= 0 and record["eta_seconds"] >= 0
        for record in progress
    )
    assert training[-1]["eta_seconds"] == 0
    assert '"event": "progress"' in capsys.readouterr().out
    summary = json.loads((output / "training.jsonl").read_text())
    assert summary["steps"] == max_steps
    assert summary["training_clips"] == train_counts[-1]
    assert summary["training_loss"] == training[-1]["mean_loss"]
    assert summary["validation_loss"] == validation[-1]["mean_loss"]
    for training_mode, key in [(True, "training_loss"), (False, "validation_loss")]:
        values = [
            (count, loss) for mode, count, loss in observed if mode == training_mode
        ]
        expected = sum(count * loss for count, loss in values) / sum(
            count for count, _ in values
        )
        assert summary[key] == pytest.approx(expected)
    config = json.loads((output / "run_config.json").read_text())
    assert config["batch_size"] == batch_size
    assert (output / "best/event_flow.pt").is_file()
    assert (output / "last/event_flow.pt").is_file()


def test_invalid_logging_interval_fails_before_loading_data(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "event-flow",
            "train",
            "--manifest",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "output"),
            "--log-every",
            "0",
        ],
    )
    with pytest.raises(ValueError, match="logging interval"):
        main()


@pytest.mark.parametrize("size", ["0", "-1"])
def test_invalid_batch_size_fails_before_loading_data(tmp_path, monkeypatch, size):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "event-flow",
            "train",
            "--manifest",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "out"),
            "--batch-size",
            size,
        ],
    )
    with pytest.raises(ValueError, match="batch size"):
        main()
