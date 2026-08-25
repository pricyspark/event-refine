from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from flowsis.base_checkpoint import load_base
from flowsis.cli.deploy.common import center_square, resolve_video_source
from flowsis.set_mask import BasePrediction
from flowsis.utils import get_device

WINDOW_NAME = "FlowSIS Set Segmentation"
COLORS = (
    (0, 220, 0),
    (0, 180, 255),
    (255, 120, 0),
    (220, 0, 180),
    (180, 220, 0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run set-based FlowSIS inference on a video or camera feed."
    )
    parser.add_argument("--model-path", type=Path, default=Path("outputs/base/final"))
    parser.add_argument(
        "--video-source",
        default="live",
        help="Use 'live', a camera index, or a video file path.",
    )
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--detection-threshold", type=float)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int)
    parser.add_argument("--mask-alpha", type=float, default=0.45)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--detector-amp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--cpu-preprocess", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def render_result(
    frame: np.ndarray,
    prediction: BasePrediction,
    label_names: tuple[str, ...],
    *,
    mask_threshold: float,
    mask_alpha: float,
    elapsed_ms: float,
) -> np.ndarray:
    rendered = frame.copy()
    instances = prediction.instances
    for index in range(instances.scores.numel()):
        color = COLORS[index % len(COLORS)]
        mask = (
            (instances.mask_logits[index].sigmoid() >= mask_threshold)
            .to(device="cpu", dtype=torch.uint8)
            .numpy()
        )
        if cv2.countNonZero(mask):
            overlay = np.empty_like(rendered)
            overlay[:] = color
            blended = cv2.addWeighted(
                rendered, 1.0 - mask_alpha, overlay, mask_alpha, 0.0
            )
            cv2.copyTo(blended, mask, rendered)
        x1, y1, x2, y2 = (
            int(round(value)) for value in instances.boxes[index].cpu().tolist()
        )
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        label_index = int(instances.labels[index].item())
        label = (
            label_names[label_index]
            if 0 <= label_index < len(label_names)
            else f"class_{label_index}"
        )
        text = f"{label} {float(instances.scores[index]):.2f} q={int(instances.query_indices[index])}"
        cv2.putText(
            rendered,
            text,
            (max(x1, 0), max(y1 - 7, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        rendered,
        f"{instances.scores.numel()} instances | {elapsed_ms:.1f} ms | q/esc: quit",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return rendered


def open_writer(path: Path, *, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else 30.0, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open output video: {path}")
    return writer


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.mask_threshold <= 1.0:
        raise ValueError("--mask-threshold must be between zero and one.")
    if not 0.0 <= args.mask_alpha <= 1.0:
        raise ValueError("--mask-alpha must be between zero and one.")
    if args.no_display and args.output_path is None:
        raise ValueError("--no-display requires --output-path.")
    model = load_base(
        args.model_path,
        device=torch.device(args.device) if args.device else get_device(),
        use_amp=args.amp,
        detector_amp=args.detector_amp,
        device_preprocess=not args.cpu_preprocess,
    )
    capture = cv2.VideoCapture(resolve_video_source(args.video_source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.video_source}")
    writer: cv2.VideoWriter | None = None
    frame_count = 0
    try:
        while args.max_frames is None or frame_count < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame = center_square(frame)
            started = time.perf_counter()
            prediction = model.infer_frame(
                frame,
                detection_threshold=args.detection_threshold,
                max_detections=args.max_detections,
            )
            rendered = render_result(
                frame,
                prediction,
                model.label_names,
                mask_threshold=args.mask_threshold,
                mask_alpha=args.mask_alpha,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
            if args.output_path is not None:
                if writer is None:
                    writer = open_writer(
                        args.output_path,
                        fps=float(capture.get(cv2.CAP_PROP_FPS)),
                        size=(rendered.shape[1], rendered.shape[0]),
                    )
                writer.write(rendered)
            if not args.no_display:
                cv2.imshow(WINDOW_NAME, rendered)
                if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                    break
            frame_count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    print("live_base_complete", {"frames": frame_count, "output": args.output_path})


if __name__ == "__main__":
    main()
