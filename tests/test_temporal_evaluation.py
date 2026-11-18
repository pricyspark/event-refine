"""Temporal comparison contracts, prepared with Codex assistance."""

from types import SimpleNamespace

import pytest
import torch

from flowsis.cli.eval import evaluate_temporal as evaluation
from flowsis.data.temporal import ClipSpec
from flowsis.temporal import TemporalRefinementBranch


def clip_sample():
    masks = torch.zeros(3, 1, 8, 8)
    masks[:, :, 2:6, 2:6] = 1
    return {
        "frames": torch.zeros(3, 3, 8, 8),
        "masks": masks,
        "labels": torch.zeros(3, 1, dtype=torch.long),
        "mask_valid": torch.ones(3, 1, dtype=torch.bool),
        "visible": torch.ones(3, 1, dtype=torch.bool),
        "ignore": torch.zeros(3, 8, 8, dtype=torch.bool),
        "track_ids": ("tool",),
    }


def test_evaluation_chunks_never_cross_intervals_or_repeat_frames():
    dataset = SimpleNamespace(
        frame_indices={"v": list(range(20))},
        sequence_ids={"v": {i: "a" if i < 8 else "b" for i in range(20)}},
    )
    specs = evaluation.evaluation_clips(dataset, length=16, clips_per_video=6)
    assert [(spec.start, spec.length) for spec in specs] == [(0, 8), (8, 12)]
    frames = [frame for spec in specs for frame in spec.frame_indices]
    assert len(frames) == len(set(frames)) == 20


def test_mask_scores_ignore_pixels_and_count_missing_prediction_as_zero():
    target = torch.zeros(8, 8)
    target[2:6, 2:6] = 1
    logits = torch.where(target.bool(), 5.0, -5.0)
    logits[0] = 5
    ignore = torch.zeros(8, 8, dtype=torch.bool)
    ignore[0] = True
    assert evaluation.mask_metrics(logits, target, ignore, 0.5) == (1.0, 1.0)
    assert evaluation.mask_metrics(None, target, ignore, 0.5) == (0.0, 0.0)


def test_oracle_identity_branch_has_zero_paired_delta_at_every_position(monkeypatch):
    sample = clip_sample()
    batch = {
        key: value[None]
        for key, value in sample.items()
        if isinstance(value, torch.Tensor)
    }
    batch["boxes"] = torch.zeros(1, 3, 1, 4)
    batch["base_valid"] = batch["mask_valid"]
    monkeypatch.setattr(evaluation, "collate_temporal_clips", lambda _: batch)
    logits = torch.where(sample["masks"][0, 0].bool(), 5.0, -5.0)
    output = SimpleNamespace(pixel_features=torch.zeros(1, 1, 2, 8, 8))
    monkeypatch.setattr(evaluation, "_base_sequence", lambda *a, **kw: output)
    monkeypatch.setattr(
        evaluation,
        "select_training_observations",
        lambda *args: (
            logits.reshape(1, 1, 1, 1, 8, 8),
            torch.ones(1, 1, 1),
            torch.ones(1, 1, 1, 1),
            torch.ones(1, 1, 1, dtype=torch.bool),
        ),
    )
    branch = TemporalRefinementBranch(
        channels=(8,), feature_channels=2, context_channels=2, hidden_channels=2
    ).eval()
    base = SimpleNamespace(device=torch.device("cpu"), mask_threshold=0.5)
    flow = lambda a, b: torch.zeros(a.shape[0], 2, *a.shape[-2:])
    rows = evaluation.oracle_clip(base, flow, branch, sample, ClipSpec("v", 0, 3, 1))
    assert [row["position"] for row in rows] == [0, 1, 2]
    assert all(row["base_dice"] == 1 and row["delta_dice"] == 0 for row in rows)


def test_runtime_warmup_miss_and_reset_are_reported_separately():
    sample = clip_sample()
    logits = torch.where(sample["masks"][0, 0].bool(), 5.0, -5.0)
    instances = SimpleNamespace(
        scores=torch.tensor([0.9]), labels=torch.tensor([0]), mask_logits=logits[None]
    )

    class Runtime:
        device = torch.device("cpu")
        base = SimpleNamespace(
            mask_threshold=0.5,
            infer_frame=lambda frame: SimpleNamespace(instances=instances),
        )

        def reset(self):
            self.position = 0

        def infer_frame(self, frame):
            selected = (
                None if self.position == 0 else SimpleNamespace(label=0, track_id=7)
            )
            self.position += 1
            return SimpleNamespace(selected=selected, logits_by_track={7: logits})

    model = Runtime()
    for start in (0, 10):
        rows = evaluation.runtime_clip(model, sample, ClipSpec("v", start, 3, 1))
        assert [row["delta_dice"] for row in rows] == [-1, 0, 0]
    sample["labels"].fill_(1)
    rows = evaluation.runtime_clip(model, sample, ClipSpec("v", 0, 3, 1))
    assert all(row["base_dice"] == row["temporal_dice"] == 0 for row in rows)


def test_summary_retains_pairing_position_support_and_equal_video_mean():
    spec = ClipSpec("long", 0, 2, 1)
    rows = [
        evaluation.frame_record("oracle", spec, 0, "t", (0.5, 0.4), (0.6, 0.5))
        for _ in range(9)
    ]
    rows += [
        evaluation.frame_record(
            "oracle", ClipSpec("short", 0, 2, 1), 1, "t", (0.5, 0.4), (0.4, 0.3)
        )
    ]
    result = evaluation.summarize_frames(rows)["oracle"]
    assert result["overall"]["delta_dice"] == pytest.approx(0.08)
    assert result["equal_video_delta_dice"] == pytest.approx(0)
    assert result["by_position"]["1"]["count"] == 1
    assert result["subsequent_frames"]["delta_dice"] == pytest.approx(-0.1)


def test_cli_writes_position_plot_csv_and_summary(tmp_path, monkeypatch):
    sample = clip_sample()

    class Dataset:
        frame_indices = {"v": [0, 1, 2]}
        sequence_ids = {"v": {0: "a", 1: "a", 2: "a"}}

        def __getitem__(self, spec):
            return sample

    monkeypatch.setattr(
        evaluation, "load_base", lambda *a, **kw: SimpleNamespace(label_names=("tool",))
    )
    monkeypatch.setattr(evaluation, "load_local_dataset", lambda _: {"validation": []})
    monkeypatch.setattr(evaluation, "temporal_dataset_view", lambda x: x)
    monkeypatch.setattr(evaluation, "_label_names", lambda _: ("tool",))
    monkeypatch.setattr(evaluation, "TemporalClipDataset", lambda *a, **kw: Dataset())
    monkeypatch.setattr(
        evaluation, "load_flow_estimator", lambda *a: torch.nn.Identity()
    )
    monkeypatch.setattr(
        evaluation, "load_temporal_checkpoint", lambda *a, **kw: SimpleNamespace()
    )
    monkeypatch.setattr(evaluation, "FlowSISTemporal", lambda *a: None)
    monkeypatch.setattr(
        evaluation,
        "oracle_clip",
        lambda base, flow, temporal, sample, spec: [
            evaluation.frame_record(
                "oracle_matched", spec, i, "t", (0.5, 0.4), (0.5 + 0.05 * i, 0.4)
            )
            for i in range(3)
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--dataset-path",
            "unused",
            "--base-model",
            "unused",
            "--temporal-model",
            "unused",
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--mode",
            "oracle",
        ],
    )
    evaluation.main()
    assert (tmp_path / "frame_position.png").stat().st_size > 1000
    assert len((tmp_path / "frames.csv").read_text().splitlines()) == 4
    import json

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["metrics"]["oracle_matched"]["by_position"]["2"][
        "delta_dice"
    ] == pytest.approx(0.1)


@pytest.mark.parametrize("length", [2, 4, 8, 16, 32, 64, 0])
def test_exhaustive_length_sweep_preserves_identical_frames_including_tails(length):
    dataset = SimpleNamespace(
        frame_indices={"v": [*range(13), *range(20, 24)]},
        sequence_ids={
            "v": {i: "a" if i < 13 else "b" for i in [*range(13), *range(20, 24)]}
        },
    )
    clips = evaluation.evaluation_clips(
        dataset, length, clips_per_video=1, all_frames=True
    )
    frames = [i for spec in clips for i in spec.frame_indices]
    assert frames == [*range(13), *range(20, 24)]
    assert all(spec.frame_indices[-1] < 13 or spec.start >= 20 for spec in clips)
    if length == 4:
        assert any(spec.length == 1 for spec in clips)
