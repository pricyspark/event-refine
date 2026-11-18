"""Fresh-export guards and saved-model checks, with Codex assistance."""

import json

import pytest
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from experiments import overnight_support as support
from flowsis.data.prompts import validate_prompt_suite


def export(tmp_path, *, labels=("needle_holder",), video_length=0, leaked=False):
    features = Features(
        {
            "video_id": Value("string"),
            "video_path": Value("string"),
            "sample_id": Value("string"),
            "sequence_id": Value("string"),
            "frame_index": Value("int64"),
            "tool_ids": [Value("string")],
            "base_eligible": Value("bool"),
            "objects": [
                {
                    "category": ClassLabel(names=list(labels)),
                    "tool_class": Value("string"),
                    "tool_id": Value("string"),
                    "mask_valid": Value("bool"),
                    "visible": Value("bool"),
                }
            ],
        }
    )

    def rows(name):
        return [
            {
                "video_id": name if video_length else "",
                "video_path": "",
                "sample_id": name,
                "sequence_id": name,
                "frame_index": index,
                "tool_ids": ["held" if leaked else name],
                "base_eligible": True,
                "objects": [
                    {
                        "category": i,
                        "tool_class": label,
                        "tool_id": name,
                        "mask_valid": True,
                        "visible": True,
                    }
                    for i, label in enumerate(labels)
                ],
            }
            for index in range(max(1, video_length))
        ]

    data = {
        name: Dataset.from_list(rows(name), features=features)
        for name in ("train", "validation")
    }
    if leaked:
        data["test_unseen"] = Dataset.from_list(rows("test_unseen"), features=features)
    path = tmp_path / "dataset"
    DatasetDict(data).save_to_disk(path)
    if leaked:
        (path / "build_report.json").write_text(
            json.dumps({"physical_tool_holdout": {"unseen_tool_ids": ["held"]}})
        )
    return path


def test_still_export_skips_temporal_and_warns_about_missing_coverage(tmp_path):
    result = support.inspect_dataset(export(tmp_path))
    assert result["labels"] == ["needle_holder"]
    assert result["temporal_length"] == 0
    assert len(result["fingerprint"]) == 64
    assert any("empty scenes" in message for message in result["warnings"])


def test_short_clips_choose_supported_geometry(tmp_path):
    result = support.inspect_dataset(export(tmp_path, video_length=4))
    assert result["temporal_length"] == 4
    assert result["splits"]["validation"]["temporal_videos"]["8"] == 0


def test_semantic_label_duplicates_fail_before_training(tmp_path):
    with pytest.raises(ValueError, match="separators/case"):
        support.inspect_dataset(
            export(tmp_path, labels=("needle_holder", "needleholder"))
        )


def test_unseen_tool_leakage_fails(tmp_path):
    with pytest.raises(ValueError, match="leaked"):
        support.inspect_dataset(export(tmp_path, leaked=True))


def test_bootstrap_prompts_preserve_canonical_names():
    suite = support.prompt_suite(["needle_holder"])
    validate_prompt_suite(suite, {"needle_holder": "needle_holder"})
    assert "needle holder" in suite["needle_holder"][2]


def test_temporal_marker_and_finite_weights_are_required(tmp_path):
    path = tmp_path / "temporal.pt"
    torch.save({"state_dict": {"x": torch.ones(1)}}, path)
    with pytest.raises(ValueError, match="marker"):
        support.check_temporal_checkpoint(path)
    payload = {
        "training_preprocessing": "unit_rgb_to_uint8_detector_v1",
        "state_dict": {"x": torch.ones(1)},
    }
    torch.save(payload, path)
    support.check_temporal_checkpoint(path)
    payload["state_dict"]["x"].fill_(float("nan"))
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Nonfinite"):
        support.check_temporal_checkpoint(path)


def test_identity_ablation_restores_base_logits_from_a_nonidentity_refiner(tmp_path):
    from flowsis.temporal import TemporalRefinementBranch
    from flowsis.temporal_checkpoint import (
        save_temporal_checkpoint,
        load_temporal_checkpoint,
    )

    branch = TemporalRefinementBranch(
        channels=(8,), context_channels=2, hidden_channels=2
    ).eval()
    with torch.no_grad():
        branch.propagation_strength.fill_(2)
        branch.residual_strength.fill_(3)
        branch.residual_head.bias.fill_(1)
    source = save_temporal_checkpoint(branch, tmp_path / "source")
    output = tmp_path / "identity.pt"
    support.identity_checkpoint(source, output)
    identity = load_temporal_checkpoint(output)
    frames = torch.zeros(1, 3, 8, 8)
    logits = torch.zeros(1, 1, 8, 8)
    flow = torch.zeros(1, 2, 8, 8)
    with torch.no_grad():
        learned = branch(frames, frames, logits, logits + 2, flow)
        control = identity(frames, frames, logits, logits + 2, flow)
    assert not torch.allclose(learned.final_logits, logits)
    torch.testing.assert_close(control.final_logits, logits, rtol=0, atol=0)
    torch.testing.assert_close(
        identity.visibility_head.weight, branch.visibility_head.weight
    )


def test_real_export_dry_run_never_launches_jobs_or_writes_outputs(
    tmp_path, monkeypatch, capsys
):
    from experiments import run_overnight

    path = export(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "overnight",
            "--dataset-path",
            str(path),
            "--output-dir",
            str(output),
            "--dry-run",
        ],
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("Dry run must not launch subprocesses")

    monkeypatch.setattr(run_overnight.subprocess, "run", unexpected)
    run_overnight.main()
    printed = capsys.readouterr().out
    assert "[prompt-embeddings]" in printed and "[detector-train]" in printed
    assert "[temporal-main]" not in printed
    assert "--split test" not in printed
    assert not output.exists()


def test_required_temporal_fails_before_creating_run(tmp_path, monkeypatch):
    from experiments import run_overnight

    path = export(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "overnight",
            "--dataset-path",
            str(path),
            "--output-dir",
            str(output),
            "--require-temporal",
        ],
    )
    with pytest.raises(ValueError, match="usable clips"):
        run_overnight.main()
    assert not output.exists()


def test_failed_stage_preserves_report_and_stops_pipeline(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments import run_overnight

    path = export(tmp_path)
    output = tmp_path / "output"
    calls = []

    def failure(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(run_overnight.subprocess, "run", failure)
    monkeypatch.setattr(
        "sys.argv",
        [
            "overnight",
            "--dataset-path",
            str(path),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ],
    )
    with pytest.raises(RuntimeError, match="prompt-embeddings failed"):
        run_overnight.main()
    assert len(calls) == 1
    assert json.loads((output / "summary.json").read_text())["status"].startswith(
        "failed:"
    )
    assert (output / "REPORT.md").is_file()
    assert (output / "MODEL_MANIFEST.json").is_file()
