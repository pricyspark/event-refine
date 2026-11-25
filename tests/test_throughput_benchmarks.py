"""Numerical batching equivalence and isolated benchmark harness behavior."""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from event_flow.benchmark import batch_clip_loss
from event_flow.cli import _clip_loss
from event_flow.data import ClipDataset
from event_flow.model import EventFlow
from test_event_flow import tiny_detector, write_manifest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "throughput_matrix", ROOT / "scripts/benchmark_throughput.py"
)
MATRIX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATRIX
SPEC.loader.exec_module(MATRIX)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def clips(tmp_path):
    dataset = ClipDataset(write_manifest(tmp_path), split="train", clip_length=2)
    generator = torch.Generator().manual_seed(17)
    first = [
        replace(sample, rgb=torch.rand(sample.rgb.shape, generator=generator))
        for sample in dataset[0]
    ]
    second = [
        replace(
            sample,
            rgb=torch.rand(sample.rgb.shape, generator=generator) * 0.5,
            events=sample.events.roll(4, -1),
            boxes=sample.boxes + 2,
        )
        for sample in dataset[1]
    ]
    return [first, second]


def benchmark_model(architecture="dfine"):
    torch.manual_seed(42)
    return EventFlow(tiny_detector(architecture), hidden_dim=16).eval()


@pytest.mark.parametrize("architecture", ["dfine", "rtdetrv2"])
def test_batched_loss_and_temporal_gradients_match_independent_clips(
    tmp_path, architecture
):
    samples = clips(tmp_path)
    model = benchmark_model(architecture)
    batched = batch_clip_loss(model, samples, "cpu")
    batched.backward()
    gradients = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    separate = torch.stack(
        [_clip_loss(model, clip, "cpu", 0.5) for clip in samples]
    ).mean()
    torch.testing.assert_close(batched, separate, rtol=1e-4, atol=1e-5)
    separate.backward()
    for name, parameter in model.named_parameters():
        if name in gradients:
            assert torch.isfinite(gradients[name]).all()
            assert torch.isfinite(parameter.grad).all()
        # Check the new temporal branch's reduction explicitly. Native detector
        # derivatives at nonsmooth operations are not a bitwise-batching oracle;
        # training also intentionally changes batch-normalization statistics.
        if name in gradients and name.startswith("temporal."):
            torch.testing.assert_close(
                gradients[name], parameter.grad, rtol=2e-4, atol=1e-5
            )


def test_trusted_inputs_preserve_values_and_keep_structural_checks(tmp_path):
    samples = clips(tmp_path)
    model = benchmark_model()
    rgb = torch.stack([clip[0].rgb for clip in samples])
    events = torch.stack([clip[0].events for clip in samples])
    checked = model.prepare(rgb, events)
    trusted = model.prepare(rgb, events, validate_values=False)
    for left, right in zip(checked, trusted, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    with pytest.raises(ValueError, match="aligned"):
        model.prepare(rgb, events[:1], validate_values=False)
    with pytest.raises(ValueError, match="binary"):
        model.prepare(rgb, events + 0.25)
    with torch.no_grad():
        torch.testing.assert_close(
            batch_clip_loss(model, samples, "cpu"),
            batch_clip_loss(model, samples, "cpu", validate_values=False),
        )


def test_batching_does_not_mix_temporal_memory_when_order_changes(tmp_path):
    samples = clips(tmp_path)
    model = benchmark_model()
    torch.nn.init.normal_(model.temporal.class_delta.weight, std=0.1)
    torch.nn.init.normal_(model.temporal.box_delta.weight, std=0.1)
    with torch.no_grad():
        original = batch_clip_loss(model, samples, "cpu")
        reversed_batch = batch_clip_loss(model, samples[::-1], "cpu")
        separate = torch.stack(
            [_clip_loss(model, clip, "cpu", 0.5) for clip in samples]
        ).mean()
    torch.testing.assert_close(original, reversed_batch)
    torch.testing.assert_close(original, separate, rtol=1e-4, atol=1e-5)


def test_matrix_deduplicates_and_compares_compile_with_trusted_eager():
    cases = MATRIX.build_cases("all", [1, 2], [2, 1, 4, 8], 2)
    assert len(cases) == 18 and len(set(cases)) == 18
    compile_cases = MATRIX.build_cases("compile", [1], [2], 1)
    assert [case.mode for case in compile_cases] == ["trusted-inputs", "compile"]
    summary = MATRIX.summarize(
        [
            {
                "batch_size": 1,
                "threads": 2,
                "mode": "eager",
                "status": "ok",
                "clips_per_second": 2.0,
            },
            {
                "batch_size": 1,
                "threads": 2,
                "mode": "eager",
                "status": "ok",
                "clips_per_second": 4.0,
            },
            {
                "batch_size": 2,
                "threads": 2,
                "mode": "eager",
                "status": "oom",
                "clips_per_second": None,
            },
        ]
    )
    assert len(summary) == 1 and summary[0]["median_clips_per_second"] == 3.0


def test_worker_cpu_smoke(tmp_path):
    manifest = write_manifest(tmp_path)
    detector = tmp_path / "detector"
    tiny_detector().save_pretrained(detector)
    output = tmp_path / "result"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "event_flow.benchmark",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--detector-model",
            str(detector),
            "--device",
            "cpu",
            "--image-size",
            "64",
            "--hidden-dim",
            "16",
            "--clip-length",
            "2",
            "--warmup",
            "1",
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--threads",
            "1",
        ],
        cwd=ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metrics = json.loads((output / "result.json").read_text())
    assert metrics["status"] == "ok"
    assert metrics["measured_clips"] == 4 and metrics["measured_frames"] == 8
    assert metrics["clips_per_second"] > 0
    assert metrics["loss_and_gradient_checks"]
    assert not (output / "last").exists()


def test_runner_dry_run_does_not_create_output(tmp_path):
    output = tmp_path / "matrix"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/benchmark_throughput.py"),
            "--manifest",
            str(tmp_path / "unused.json"),
            "--output",
            str(output),
            "--suite",
            "batching",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert result.stdout.count("event_flow.benchmark") == 2
    assert not output.exists()


@pytest.mark.parametrize("interrupt", [False, True])
def test_runner_stops_worker_group_on_timeout_or_interrupt(
    tmp_path, monkeypatch, interrupt
):
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    if interrupt:
        original_wait = process.wait
        first = True

        def interrupt_once(*args, **kwargs):
            nonlocal first
            if first:
                first = False
                raise KeyboardInterrupt
            return original_wait(*args, **kwargs)

        monkeypatch.setattr(process, "wait", interrupt_once)
    expected = KeyboardInterrupt if interrupt else subprocess.TimeoutExpired
    try:
        with pytest.raises(expected):
            MATRIX._wait_for_case(process, timeout=0.05, log=tmp_path / "log")
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_runner_writes_real_cpu_case_summaries(tmp_path):
    manifest = write_manifest(tmp_path)
    detector = tmp_path / "detector"
    tiny_detector().save_pretrained(detector)
    output = tmp_path / "matrix"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/benchmark_throughput.py"),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--detector-model",
            str(detector),
            "--device",
            "cpu",
            "--image-size",
            "64",
            "--hidden-dim",
            "16",
            "--clip-length",
            "2",
            "--warmup",
            "1",
            "--steps",
            "1",
            "--threads",
            "1",
        ],
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (output / "results.csv").is_file()
    assert (output / "summary.csv").is_file()
    assert len(list(output.glob("*/result.json"))) == 2
    for path in output.glob("*/result.json"):
        assert json.loads(path.read_text())["status"] == "ok"


@pytest.mark.parametrize("architecture", ["dfine", "rtdetrv2"])
def test_batching_clips_with_different_image_sizes(tmp_path, architecture):
    samples = clips(tmp_path)
    samples[1] = [
        replace(sample, rgb=sample.rgb[:, :40, :48], events=sample.events[:, :40, :48])
        for sample in samples[1]
    ]
    model = benchmark_model(architecture)
    with torch.no_grad():
        batched = batch_clip_loss(model, samples, "cpu")
        separate = torch.stack(
            [_clip_loss(model, clip, "cpu", 0.5) for clip in samples]
        ).mean()
    torch.testing.assert_close(batched, separate, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("invalid", ["empty", "length", "dimensions"])
def test_invalid_clip_batches_are_rejected(tmp_path, invalid):
    samples = clips(tmp_path)
    if invalid == "empty":
        samples = []
    elif invalid == "length":
        samples[1] = samples[1][:1]
    else:
        samples[1][1] = replace(samples[1][1], rgb=samples[1][1].rgb[:, :40])
    with pytest.raises(ValueError):
        batch_clip_loss(benchmark_model(), samples, "cpu")
