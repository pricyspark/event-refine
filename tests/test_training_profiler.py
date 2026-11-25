"""Scheduled profiler export and real training stage coverage."""

import argparse
import json
import sys

import pytest
import torch

from event_flow.cli import main, train
from event_flow.profiling import training_profiler
from test_event_flow import tiny_detector, write_manifest


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("batch_size,active_steps,max_steps", [(1, 2, 4), (2, 1, 2)])
def test_training_trace_records_only_requested_steps(
    tmp_path, batch_size, active_steps, max_steps
):
    manifest = write_manifest(tmp_path)
    detector_dir = tmp_path / "detector"
    tiny_detector().save_pretrained(detector_dir)
    output = tmp_path / "profiled"
    train(
        argparse.Namespace(
            manifest=manifest,
            split="train",
            validation_split=None,
            clip_length=1,
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
            max_validation_clips=None,
            profile=True,
            profile_warmup=1,
            profile_steps=active_steps,
        )
    )
    directory = output / "profile"
    trace = json.loads((directory / "trace.json").read_text())
    complete_events = [
        event for event in trace["traceEvents"] if event.get("ph") == "X"
    ]
    names = [event["name"] for event in complete_events]
    for stage in (
        "data_load",
        "preprocess_and_transfer",
        "forward",
        "matching_and_loss",
        "backward",
        "gradient_clip",
        "optimizer_step",
    ):
        assert names.count(stage) == active_steps
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["completed_steps_at_export"] == 1 + active_steps
    assert metadata["active_steps"] == active_steps
    assert not metadata["cuda_requested"]
    assert metadata["cpu_threads"] == 1
    assert "aten::" in (directory / "operators.txt").read_text()
    assert json.loads((output / "training.jsonl").read_text())["steps"] == max_steps


def test_disabled_profiler_creates_no_files(tmp_path):
    output = tmp_path / "output"
    with training_profiler(
        output,
        enabled=False,
        device=torch.device("cpu"),
        warmup_steps=1,
        active_steps=1,
    ) as profiler:
        assert profiler is None
    assert not output.exists()


@pytest.mark.parametrize("extra", [["--profile-warmup", "0"], ["--profile-steps", "0"]])
def test_invalid_profile_window_fails_before_loading_data(tmp_path, monkeypatch, extra):
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
            "--profile",
            *extra,
        ],
    )
    with pytest.raises(ValueError, match="Profiler warmup"):
        main()


def test_incomplete_capture_window_is_rejected_before_model_load(tmp_path, monkeypatch):
    manifest = write_manifest(tmp_path)
    monkeypatch.setattr(
        "event_flow.cli.load_detector",
        lambda *args, **kwargs: pytest.fail(
            "Loaded model before checking profiling window"
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "event-flow",
            "train",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "output"),
            "--profile",
            "--max-steps",
            "2",
            "--device",
            "cpu",
        ],
    )
    with pytest.raises(ValueError, match="at least 15 training steps"):
        main()


def test_profile_window_counts_batches(tmp_path, monkeypatch):
    manifest = write_manifest(tmp_path)
    monkeypatch.setattr(
        "event_flow.cli.load_detector", lambda *a, **kw: pytest.fail("Loaded detector")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "event-flow",
            "train",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "out"),
            "--clip-length",
            "1",
            "--batch-size",
            "2",
            "--profile",
            "--profile-warmup",
            "1",
            "--profile-steps",
            "2",
        ],
    )
    with pytest.raises(ValueError, match="at least 3 training steps"):
        main()
