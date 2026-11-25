"""Complete model warm starts retain trained temporal and detector weights."""

import json
import sys

import pytest
import torch

from event_flow.checkpoint import load_checkpoint, save_checkpoint
from event_flow.cli import main
from event_flow.model import EventFlow
from event_flow.training import batch_clip_loss
from test_event_flow import tiny_detector, write_manifest


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("architecture", ["dfine", "rtdetrv2"])
def test_warm_start_preserves_weights_and_trains(tmp_path, monkeypatch, architecture):
    manifest = write_manifest(tmp_path)
    source = tmp_path / "previous" / "last"
    model = EventFlow(tiny_detector(architecture), hidden_dim=16)
    with torch.no_grad():
        model.temporal.class_delta.bias.fill_(0.25)
    save_checkpoint(model, source, classes=("drone",))
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    called = False

    def check_weights(loaded, clips, *args, **kwargs):
        nonlocal called
        if not called:
            assert loaded.training
            for name, value in loaded.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        called = True
        return batch_clip_loss(loaded, clips, *args, **kwargs)

    monkeypatch.setattr("event_flow.cli.batch_clip_loss", check_weights)
    monkeypatch.setattr(
        "event_flow.cli.load_detector",
        lambda *a, **kw: pytest.fail("Reinitialized detector"),
    )
    output = tmp_path / "continued"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "event-flow",
            "train",
            "--manifest",
            str(manifest),
            "--init-from",
            str(source),
            "--image-size",
            "64",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--clip-length",
            "2",
            "--epochs",
            "2",
            "--max-steps",
            "1",
            "--output",
            str(output),
        ],
    )
    main()
    assert called
    config = json.loads((output / "run_config.json").read_text())
    assert config["architecture"] == architecture
    assert config["hidden_dim"] == 16
    assert config["init_from"] == str(source)
    progress = [
        json.loads(line)
        for line in (output / "progress.jsonl").read_text().splitlines()
    ]
    assert (
        next(row for row in progress if row["event"] == "checkpoint_loaded")[
            "optimizer_restored"
        ]
        is False
    )
    summary = json.loads((output / "training.jsonl").read_text())
    assert summary["epoch"] == 1 and summary["steps"] == 1
    trained, _ = load_checkpoint(output / "last")
    assert not torch.equal(
        trained.temporal.class_delta.bias, expected["temporal.class_delta.bias"]
    )
    unchanged, _ = load_checkpoint(source)
    torch.testing.assert_close(
        unchanged.temporal.class_delta.bias,
        expected["temporal.class_delta.bias"],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "invalid",
    ["classes", "architecture", "image-size", "source-output", "detector-model"],
)
def test_incompatible_warm_start_is_rejected(tmp_path, monkeypatch, invalid):
    manifest = write_manifest(tmp_path)
    source = tmp_path / "previous" / "last"
    model = EventFlow(tiny_detector(), hidden_dim=16)
    save_checkpoint(model, source, classes=("drone",))
    output = tmp_path / "continued"
    extra = []
    message = "class names"
    image_size = "64"
    if invalid == "classes":
        document = json.loads(manifest.read_text())
        document["classes"] = ["bird"]
        manifest.write_text(json.dumps(document))
    elif invalid == "architecture":
        extra = ["--architecture", "rtdetrv2"]
        message = "Architecture"
    elif invalid == "image-size":
        image_size = "128"
        message = "Image size"
    elif invalid == "source-output":
        output = source.parent
        message = "new output"
    else:
        extra = ["--detector-model", "unused"]
        message = "either"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "event-flow",
            "train",
            "--manifest",
            str(manifest),
            "--init-from",
            str(source),
            "--image-size",
            image_size,
            "--device",
            "cpu",
            "--output",
            str(output),
            *extra,
        ],
    )
    with pytest.raises(ValueError, match=message):
        main()
