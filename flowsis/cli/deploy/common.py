from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class FrameRecorder:
    """Save rendered frames to a separate MP4 for each recording session."""

    def __init__(self, root: Path, fps: float) -> None:
        self.root = root
        self.fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        self.path: Path | None = None
        self.writer: cv2.VideoWriter | None = None
        self.frame_count = 0

    def toggle(self) -> None:
        if self.path is not None:
            self.close()
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / datetime.now().strftime("%Y%m%d_%H%M%S_%f.mp4")
        self.frame_count = 0
        print(f"Recording started: {self.path}")

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.path is not None:
            print(f"Recording stopped: {self.frame_count} frames for {self.path}")
            self.path = None

    def write(self, frame: NDArray[np.uint8]) -> None:
        if self.path is None:
            return
        if self.writer is None:
            self.writer = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (frame.shape[1], frame.shape[0]),
            )
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise RuntimeError(f"Unable to open recording video: {self.path}")
        self.writer.write(frame)
        self.frame_count += 1

    def draw_status(self, rendered: NDArray[np.uint8]) -> None:
        recording = self.path is not None
        text = (
            f"REC {self.frame_count} | r: stop recording"
            if recording
            else "r: start recording"
        )
        cv2.putText(
            rendered,
            text,
            (10, rendered.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255) if recording else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def resolve_video_source(source: str) -> int | str:
    if source == "live":
        return 0
    if source.isdigit():
        return int(source)
    return source


def center_square(frame: NDArray) -> NDArray:
    height, width = frame.shape[:2]
    size = min(height, width)
    top = (height - size) // 2
    left = (width - size) // 2
    return np.ascontiguousarray(frame[top : top + size, left : left + size])
