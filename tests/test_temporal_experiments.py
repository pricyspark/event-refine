"""Controlled suite planning/resume guards, prepared with Codex assistance."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments import run_temporal_experiments as suite
from flowsis.cli.eval.evaluate_temporal import frame_record, summarize_frames
from flowsis.data.temporal import ClipSpec


def options(tmp_path, **changes):
    values = dict(
        dataset_path=tmp_path / "dataset",
        output_dir=tmp_path / "output",
        device="cpu",
        image_size=64,
        flow_checkpoint="things",
        max_videos=None,
        max_frames_per_video=None,
        lengths=["4", "full"],
        skip_oracle=False,
        train_candidates=False,
        training_batches=75,
        training_epochs=16,
        num_workers=0,
        seed=42,
    )
    return argparse.Namespace(**(values | changes))


def test_plan_covers_all_policies_and_both_refiners_without_test_data(tmp_path):
    jobs = suite.make_plan(
        options(tmp_path),
        tmp_path / "base",
        tmp_path / "temporal.pt",
        tmp_path / "strong",
        8,
        7,
    )
    assert len(jobs) == 15
    assert all(job["kind"] == "eval" for job in jobs)
    assert all("--all-frames" in job["arguments"] for job in jobs)
    assert all(
        job["arguments"][job["arguments"].index("--split") + 1] == "validation"
        for job in jobs
    )
    assert {job["policy"] for job in jobs} == {"current", "fallback", "history"}
    assert sum("--identity-refinement" in job["arguments"] for job in jobs) == 7


def test_long_clip_training_keeps_validation_geometry_and_budget_constant(tmp_path):
    jobs = suite.make_plan(
        options(tmp_path, train_candidates=True),
        tmp_path / "base",
        tmp_path / "temporal.pt",
        tmp_path / "strong",
        8,
        7,
    )
    train = [job for job in jobs if job["kind"] == "train"]
    assert len(train) == 2
    for job in train:
        args = job["arguments"]
        assert args[args.index("--validation-clip-length") + 1] == "8"
        assert args[args.index("--epochs") + 1] == "16"
        assert args[args.index("--batches-per-epoch") + 1] == "75"
        assert "--finetune-mask-head" not in args
    assert {
        job["arguments"][job["arguments"].index("--max-clip-length") + 1]
        for job in train
    } == {"8", "16"}


def fixture_run(tmp_path, monkeypatch):
    from experiments import overnight_support
    from flowsis.data import loaders

    run = tmp_path / "run"
    run.mkdir()
    base = tmp_path / "base"
    base.mkdir()
    (base / "base.pt").write_bytes(b"head")
    temporal = tmp_path / "temporal.pt"
    torch.save(
        {
            "training_preprocessing": "unit_rgb_to_uint8_detector_v1",
            "state_dict": {"x": torch.ones(1)},
        },
        temporal,
    )
    (run / "summary.json").write_text(
        json.dumps(
            {"base": str(base), "standalone_base": str(base), "temporal": str(temporal)}
        )
    )
    (run / "provenance.json").write_text(
        json.dumps({"args": {"dataset_path": str(tmp_path / "dataset")}})
    )
    monkeypatch.setattr(
        suite,
        "inspect_dataset",
        lambda _: {
            "temporal_length": 8,
            "fingerprint": "fixed",
            "splits": {"validation": {"temporal_videos": {"8": 1}}},
        },
    )
    monkeypatch.setattr(loaders, "load_local_dataset", lambda _: {"validation": []})
    monkeypatch.setattr(loaders, "temporal_dataset_view", lambda rows: rows)
    return run


def write_fake_evaluation(command):
    output = Path(command[command.index("--output-dir") + 1])
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        frame_record(
            "runtime_single_target",
            ClipSpec("v", 0, 3, 1),
            i,
            "tool",
            (0.5, 0.4),
            (0.6, 0.45),
        )
        for i in range(3)
    ]
    (output / "summary.json").write_text(
        json.dumps(
            {
                "metrics": summarize_frames(rows),
                "runtime_coverage": {
                    "frames": 3,
                    "annotated_empty_frames": 0,
                    "selected_on_annotated_empty": 0,
                },
            }
        )
    )
    with (output / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_failed_job_can_be_resumed_without_repeating_completed_jobs(
    tmp_path, monkeypatch
):
    run = fixture_run(tmp_path, monkeypatch)
    output = tmp_path / "output"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return SimpleNamespace(returncode=9)
        write_fake_evaluation(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(suite.subprocess, "run", fake_run)
    argv = [
        "suite",
        "--run-dir",
        str(run),
        "--output-dir",
        str(output),
        "--device",
        "cpu",
        "--lengths",
        "full",
        "--skip-oracle",
    ]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match="1"):
        suite.main()
    first_calls = len(calls)
    assert first_calls == 7
    monkeypatch.setattr("sys.argv", argv + ["--resume"])
    suite.main()
    assert len(calls) == first_calls + 1
    assert json.loads((output / "state.json").read_text())["status"] == "complete"
    assert (output / "comparisons.csv").is_file()
    monkeypatch.setattr("sys.argv", argv + ["--resume", "--image-size", "128"])
    with pytest.raises(ValueError, match="configuration"):
        suite.main()


def test_training_loader_uses_separate_fixed_validation_geometry(monkeypatch):
    from flowsis.cli.train import train_temporal

    frames = SimpleNamespace(
        frame_indices={"v": list(range(32))},
        sequence_ids={"v": {i: "run" for i in range(32)}},
        positive_indices={"v": list(range(32))},
    )
    monkeypatch.setattr(train_temporal, "TemporalClipDataset", lambda *a, **kw: frames)
    monkeypatch.setattr(train_temporal, "temporal_dataset_view", lambda rows: rows)
    args = SimpleNamespace(
        image_size=8,
        batch_size=1,
        batches_per_epoch=2,
        validation_batches=2,
        min_clip_length=4,
        max_clip_length=16,
        validation_clip_length=8,
        strides=(1,),
        stride_weights=(1,),
        seed=42,
        num_workers=0,
        device="cpu",
    )
    _, training = train_temporal.build_loader([], args, training=True)
    experimental = [geometry[0] for geometry in training.geometries]
    _, validation = train_temporal.build_loader([], args, training=False)
    assert 16 in experimental
    assert {geometry[0] for geometry in validation.geometries} == {8}


def test_report_rejects_unmatched_scored_frame_cohorts(tmp_path):
    from experiments.temporal_experiment_report import write_report

    jobs = suite.make_plan(
        options(tmp_path, lengths=["full"]),
        tmp_path / "base",
        tmp_path / "model.pt",
        tmp_path / "strong",
        8,
        7,
    )[:2]
    for job in jobs:
        write_fake_evaluation(["python", "--output-dir", job["output"]])
    path = Path(jobs[1]["output"]) / "frames.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["frame_index"] = "99"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Scored frames differ"):
        write_report(
            tmp_path / "output", jobs, {job["name"] for job in jobs}, {}, "running"
        )


def test_low_dice_runs_and_startup_delay_include_zero_score_frames():
    from experiments.temporal_experiment_report import summarize_failures

    rows = [
        {
            "video_id": "v",
            "clip_start": "0",
            "track_id": "t",
            "position": str(i),
            "temporal_dice": str(score),
        }
        for i, score in enumerate([0.0, 0.0, 0.6, 0.2])
    ]
    summary = summarize_failures(rows)
    assert summary["longest_low_dice_run_frames"] == 2
    assert summary["frames_dice_below_0_5"] == 3
    assert (
        summary["median_frames_from_first_scored_target_to_correct_successful_only"]
        == 2
    )
