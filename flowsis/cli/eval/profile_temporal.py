from __future__ import annotations

import argparse
import json
import time

import cv2
import torch

from flowsis.base_checkpoint import load_base
from flowsis.pretrained import load_flow_estimator
from flowsis.temporal_checkpoint import load_temporal_checkpoint
from flowsis.temporal_model import FlowSISTemporal
from flowsis.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile complete temporal inference FPS."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--temporal-model", required=True)
    parser.add_argument("--video-source", required=True)
    parser.add_argument("--flow-model", default="raft_small")
    parser.add_argument("--flow-checkpoint")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--device")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else get_device()
    base = load_base(args.base_model, device=device, use_amp=args.amp)
    flow = load_flow_estimator(args.flow_model, args.flow_checkpoint).to(device)
    temporal = load_temporal_checkpoint(args.temporal_model, device=device)
    model = FlowSISTemporal(base, flow, temporal)
    capture = cv2.VideoCapture(args.video_source)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video {args.video_source}.")
    durations = []
    selected_frames = 0
    try:
        for index in range(args.frames + args.warmup_frames):
            ok, frame = capture.read()
            if not ok:
                break
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction = model.infer_frame(frame)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if index >= args.warmup_frames:
                durations.append(time.perf_counter() - started)
                selected_frames += prediction.selected is not None
    finally:
        capture.release()
    elapsed = sum(durations)
    print(
        json.dumps(
            {
                "device": str(device),
                "frames": len(durations),
                "seconds": elapsed,
                "fps": len(durations) / elapsed if elapsed else 0.0,
                "selected_frame_rate": selected_frames / max(len(durations), 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
