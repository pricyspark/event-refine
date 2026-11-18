"""Overnight orchestration safeguards, prepared with Codex assistance."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset, DatasetDict

from experiments import run_overnight as overnight


def detector_run(root: Path):
    root.mkdir(exist_ok=True)
    events = [
        {"event": "validation_epoch", "epoch": 0, "global_step": 10, "loss": 2.0},
        {"event": "validation_epoch", "epoch": 1, "global_step": 20, "loss": 1.0},
    ]
    (root / "training_log.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )
    for name in ("checkpoint-000010", "checkpoint-000020", "final"):
        path = root / name
        path.mkdir()
        (path / "flowsis_detector.json").write_text("{}")
        (path / "config.json").write_text(json.dumps({"id2label": {"0": "tool"}}))


def test_completion_requires_last_validation_and_final_save_marker(tmp_path):
    detector_run(tmp_path / "detector")
    root = tmp_path / "detector"
    assert not overnight.detector_finished(root, 20)
    assert overnight.detector_finished(root, 2)
    (root / "final/flowsis_detector.json").unlink()
    assert not overnight.detector_finished(root, 2)


def test_last_epoch_best_compares_runner_up_instead_of_duplicate_final(tmp_path):
    root = tmp_path / "detector"
    detector_run(root)
    assert [p.name for p in overnight.detector_candidates(root)] == [
        "checkpoint-000020",
        "checkpoint-000010",
    ]


def test_earlier_best_still_compares_final(tmp_path):
    root = tmp_path / "detector"
    detector_run(root)
    path = root / "training_log.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["loss"] = 0.5
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert [p.name for p in overnight.detector_candidates(root)] == [
        "checkpoint-000010",
        "final",
    ]


def test_detector_selection_ignores_partial_log_and_rejects_incomplete_save(tmp_path):
    root = tmp_path / "detector"
    detector_run(root)
    with (root / "training_log.jsonl").open("a") as handle:
        handle.write('{"event": "validation_epoch"')
    assert len(overnight.detector_history(root)) == 2
    assert overnight.detector_candidates(root)[0].name == "checkpoint-000020"
    (root / "checkpoint-000020/flowsis_detector.json").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete"):
        overnight.detector_candidates(root)


def test_macro_selection_excludes_unrepresented_classes(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "classes": {
                    "seen": {"ap": 0.8, "ground_truth": 5},
                    "unseen": {"ap": 0, "ground_truth": 0},
                }
            }
        )
    )
    assert overnight.represented_ap(path) == 0.8


@pytest.mark.parametrize(
    "control_fails,existing,temporal",
    [
        (False, True, True),
        (True, True, True),
        (False, False, True),
        (False, False, False),
    ],
)
def test_complete_orchestration_requires_temporal_before_optional_control(
    tmp_path, monkeypatch, control_fails, existing, temporal
):
    from flowsis.cli.data import embed_text_prompts
    from flowsis.data import prompts, loaders

    root = tmp_path / "detector"
    detector_run(root)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "build_report.json").write_text("{}")
    (root / "run_config.json").write_text(
        json.dumps(
            {
                "training_args": {
                    "dataset_path": str(dataset),
                    "max_steps": None,
                    "resume_from": None,
                    "save_every_epochs": 1,
                    "epochs": 2,
                }
            }
        )
    )
    monkeypatch.setattr(
        embed_text_prompts, "taxonomy_from_dataset", lambda _: {"tool": "tool"}
    )
    monkeypatch.setattr(prompts, "load_prompt_bank", lambda *args: torch.ones(1, 3, 4))
    monkeypatch.setattr(
        loaders,
        "load_local_dataset",
        lambda _: DatasetDict(
            validation=Dataset.from_list([{"video_id": ""}, {"video_id": "v"}])
        ),
    )
    from experiments import overnight_support

    split_audit = {
        "rows": 2,
        "base_rows": 2,
        "video_ids": ["v"],
        "samples": [],
        "tool_ids": ["1"],
        "classes": {"tool": 2},
        "multi_tool_rows": 0,
        "background_rows": 0,
        "temporal_videos": {"8": 1},
    }
    monkeypatch.setattr(
        overnight_support,
        "inspect_dataset",
        lambda _: {
            "labels": ["tool"],
            "splits": {
                "train": split_audit,
                "validation": split_audit,
                "test_known": split_audit,
                "test_unseen": split_audit,
            },
            "warnings": [],
            "temporal_length": 8 if temporal else 0,
            "fingerprint": "fixture",
        },
    )
    monkeypatch.setattr(overnight, "data_smoke", lambda *args: None)
    commands = []

    native_run = overnight.subprocess.run

    def fake_run(command, **kwargs):
        if len(command) < 3 or command[1] != "-m":
            return native_run(command, **kwargs)
        commands.append(command)
        module = command[2]

        def argument(flag):
            return Path(command[command.index(flag) + 1])

        if module.endswith("evaluate_base"):
            output = argument("--output-path")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        key: (0.8 if output.parent.name == "overlap-off" else 0.7)
                        for key in (
                            "matched_dice",
                            "oracle_dice",
                            "mask_ap50",
                            "box_ap50",
                        )
                    }
                )
            )
        else:
            output = argument("--output-dir")
            output.mkdir(parents=True, exist_ok=True)
            if module.endswith("train_detector"):
                detector_run(output)
            elif module.endswith("evaluate_detector"):
                (output / "metrics.json").write_text(
                    json.dumps(
                        {
                            "classes": {"tool": {"ap": 0.7, "ground_truth": 5}},
                            "iou_threshold": float(
                                command[command.index("--iou-threshold") + 1]
                                if "--iou-threshold" in command
                                else 0.5
                            ),
                            "micro_ap": 0.7,
                        }
                    )
                )
            elif module.endswith("train_base"):
                (output / "best").mkdir()
                (output / "best/base.pt").touch()
            elif module.endswith("train_temporal"):
                torch.save(
                    {
                        "training_preprocessing": "unit_rgb_to_uint8_detector_v1",
                        "state_dict": {
                            "propagation_strength": torch.zeros(()),
                            "residual_strength": torch.ones(()),
                        },
                    },
                    output / "temporal-best.pt",
                )
            elif module.endswith("evaluate_temporal"):
                (output / "summary.json").write_text(json.dumps({"metrics": {}}))
        return SimpleNamespace(
            returncode=(
                7
                if control_fails
                and module.endswith("train_base")
                and output.name == "overlap-off"
                else 0
            )
        )

    monkeypatch.setattr(overnight.subprocess, "run", fake_run)
    output = tmp_path / "output"
    monkeypatch.chdir(overnight.ROOT)
    monkeypatch.setattr(
        "sys.argv",
        [
            "overnight",
            "--detector-epochs",
            "2",
            "--device",
            "cpu",
            *(["--detector-run", str(root)] if existing else []),
            "--dataset-path",
            str(dataset),
            *(["--embeddings-dir", str(tmp_path)] if existing else []),
            "--evaluate-test",
            "--output-dir",
            str(output),
        ],
    )
    overnight.main()
    saved = json.loads((output / "summary.json").read_text())
    assert saved["status"] == "complete"
    assert saved["standalone_base"].endswith(
        "overlap25/best" if control_fails else "overlap-off/best"
    )
    assert saved["base"].endswith("overlap25/best")
    if temporal:
        assert saved["temporal"].endswith("temporal-best.pt")
    else:
        assert "temporal" not in saved
    stages = [
        json.loads(line)["stage"]
        for line in (output / "commands.jsonl").read_text().splitlines()
    ]
    if temporal:
        assert (
            stages.index("base-main")
            < stages.index("base-temporal-parity")
            < stages.index("temporal-main")
        )
        assert stages.index("temporal-main") < stages.index("base-overlap-control")
        assert "temporal-smoke8" in stages and "temporal-eval-identity" in stages
    else:
        assert not any(stage.startswith("temporal-") for stage in stages)
    assert stages.index("base-test_known") > stages.index("base-overlap-control")
    if not existing:
        assert (
            stages.index("prompt-embeddings")
            < stages.index("detector-train")
            < stages.index("base-main")
        )
    assert (output / "MODEL_MANIFEST.json").is_file()
    assert "0–45" not in (output / "REPORT.md").read_text()
    for command in commands:
        if command[2].endswith("train_temporal"):
            assert command[command.index("--flow-checkpoint") + 1] == "things"
