from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import torch

from flowsis.artifacts import atomic_write_text
from flowsis.base_checkpoint import load_base
from flowsis.cli.deploy.common import center_square, resolve_video_source
from flowsis.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile end-to-end set-based FlowSIS inference."
    )
    parser.add_argument("--model-path", type=Path, default=Path("outputs/base/final"))
    parser.add_argument("--video-source", required=True)
    parser.add_argument("--device")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--detection-threshold", type=float)
    parser.add_argument("--max-detections", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--detector-amp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Percentile fraction must be between zero and one.")
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def summarize_timings(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence of timings.")
    median = statistics.median(values)
    return {
        "count": len(values),
        "mean_ms": round(statistics.mean(values), 4),
        "median_ms": round(median, 4),
        "p90_ms": round(percentile(values, 0.90), 4),
        "p95_ms": round(percentile(values, 0.95), 4),
        "min_ms": round(min(values), 4),
        "max_ms": round(max(values), 4),
        "fps_from_median": round(1000.0 / median, 3),
    }


def load_frames(source: str, count: int) -> list[object]:
    capture = cv2.VideoCapture(resolve_video_source(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")
    frames = []
    try:
        while len(frames) < count:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(center_square(frame))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames could be read from {source}.")
    return frames


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("Warmup must be non-negative and iterations must be positive.")
    device = torch.device(args.device) if args.device else get_device()
    model = load_base(
        args.model_path,
        device=device,
        use_amp=args.amp,
        detector_amp=args.detector_amp,
    )
    frames = load_frames(args.video_source, max(args.warmup, args.iterations))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index in range(args.warmup):
        model.infer_frame(
            frames[index % len(frames)],
            detection_threshold=args.detection_threshold,
            max_detections=args.max_detections,
        )
    timings = []
    instance_counts = []
    for index in range(args.iterations):
        synchronize(device)
        started = time.perf_counter()
        prediction = model.infer_frame(
            frames[index % len(frames)],
            detection_threshold=args.detection_threshold,
            max_detections=args.max_detections,
        )
        synchronize(device)
        timings.append((time.perf_counter() - started) * 1000.0)
        instance_counts.append(prediction.instances.scores.numel())
    median = statistics.median(timings)
    summary = {
        "device": str(device),
        "iterations": args.iterations,
        "mean_ms": statistics.mean(timings),
        "median_ms": median,
        "p90_ms": percentile(timings, 0.90),
        "p95_ms": percentile(timings, 0.95),
        "throughput_fps": 1000.0 / median,
        "mean_instances": statistics.mean(instance_counts),
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else None
        ),
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output_json is not None:
        atomic_write_text(args.output_json, text)


if __name__ == "__main__":
    main()
