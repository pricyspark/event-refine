"""One isolated throughput experiment; launched by benchmark_throughput.py."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import resource
import time
import traceback

import torch

from flowsis.pretrained.detector import BaseDetector, load_detector
from .data import ClipDataset
from .training import batch_clip_loss
from .model import EventFlow


def _synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def run_case(args: argparse.Namespace) -> dict[str, object]:
    if min(args.batch_size, args.threads, args.steps, args.warmup, args.pool_size) < 1:
        raise ValueError(
            "Batch size, threads, steps, warmup, and pool size must be positive."
        )
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dataset = ClipDataset(args.manifest, split="train", clip_length=args.clip_length)
    indices = random.Random(args.seed).sample(
        range(len(dataset)), min(len(dataset), args.pool_size)
    )
    # The selection pool is fixed across cases and cycled without preloading.
    # This bounds RAM use and includes real decoding/loading in measured time.
    position = 0
    classes = dict(enumerate(dataset.classes))
    detector = load_detector(
        args.detector_model,
        architecture=args.architecture,
        num_labels=len(classes),
        id2label=classes,
        label2id={label: index for index, label in classes.items()},
        image_size=args.image_size,
        device=args.device,
    )
    if not isinstance(detector, BaseDetector):
        raise TypeError("Expected a native detector adapter.")
    model = EventFlow(detector, hidden_dim=args.hidden_dim).to(args.device).train()
    if args.mode == "compile":
        model.compile(mode=args.compile_mode, dynamic=False)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.detector.parameters(), "lr": 1e-5},
            {"params": model.temporal.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )
    use_checks = args.mode == "eager"

    def step() -> tuple[float, float]:
        nonlocal position
        before = time.perf_counter()
        clips = [
            dataset[indices[(position + image) % len(indices)]]
            for image in range(args.batch_size)
        ]
        position += args.batch_size
        load_seconds = time.perf_counter() - before
        optimizer.zero_grad(set_to_none=True)
        if args.mode == "compile" and args.compile_mode == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()
        loss = batch_clip_loss(model, clips, args.device, validate_values=use_checks)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach()), load_seconds

    print(
        json.dumps(
            {
                "event": "warmup_start",
                "steps": args.warmup,
                "mode": args.mode,
                "batch_size": args.batch_size,
            }
        ),
        flush=True,
    )
    _synchronize(args.device)
    before = time.perf_counter()
    for _ in range(args.warmup):
        step()
    _synchronize(args.device)
    warmup_seconds = time.perf_counter() - before
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    print(
        json.dumps(
            {
                "event": "measurement_start",
                "warmup_seconds": warmup_seconds,
                "steps": args.steps,
            }
        ),
        flush=True,
    )
    started = time.perf_counter()
    loss_sum = load_seconds = 0.0
    for _ in range(args.steps):
        loss, loaded = step()
        loss_sum += loss
        load_seconds += loaded
    _synchronize(args.device)
    seconds = time.perf_counter() - started
    clips = args.steps * args.batch_size
    cuda = torch.device(args.device).type == "cuda"
    result = {
        "status": "ok",
        "mode": args.mode,
        "batch_size": args.batch_size,
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "steps": args.steps,
        "warmup_steps": args.warmup,
        "warmup_seconds": warmup_seconds,
        "measured_clips": clips,
        "measured_frames": clips * args.clip_length,
        "seconds": seconds,
        "clips_per_second": clips / seconds,
        "frames_per_second": clips * args.clip_length / seconds,
        "milliseconds_per_step": seconds * 1000 / args.steps,
        "loading_ms_per_clip": load_seconds * 1000 / clips,
        "mean_loss": loss_sum / args.steps,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(args.device) / 2**30 if cuda else None
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved(args.device) / 2**30 if cuda else None
        ),
        "peak_host_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "device": args.device,
        "device_name": torch.cuda.get_device_name(args.device) if cuda else "CPU",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "input_value_checks": use_checks,
        "loss_and_gradient_checks": True,
        "compile_mode": args.compile_mode if args.mode == "compile" else None,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "selection_pool_indices": indices,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector-model")
    parser.add_argument(
        "--architecture", choices=("dfine", "rtdetrv2"), default="dfine"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--mode", choices=("eager", "trusted-inputs", "compile"), default="eager"
    )
    parser.add_argument(
        "--compile-mode", choices=("default", "reduce-overhead"), default="default"
    )
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--clip-length", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        parser.error("Output already has a result; choose a new directory.")
    (args.output / "case.json").write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            indent=2,
        )
        + "\n"
    )
    try:
        result = run_case(args)
        code = 0
    except torch.cuda.OutOfMemoryError as error:
        result = {"status": "oom", "error": str(error)}
        traceback.print_exc()
        code = 2
    except Exception as error:
        result = {"status": "error", "error": str(error)}
        traceback.print_exc()
        code = 1
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
