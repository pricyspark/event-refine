from pathlib import Path

import cv2
import numpy as np
import pytest

from flowsis.cli.deploy.common import FrameRecorder


def test_recording_sessions_save_only_active_frames(tmp_path: Path) -> None:
    root = tmp_path / "recordings"
    recorder = FrameRecorder(root, fps=12.0)
    frame = np.full((32, 32, 3), (10, 20, 30), dtype=np.uint8)
    recorder.write(frame)
    assert not root.exists()

    recorder.toggle()
    first = recorder.path
    assert first is not None
    recorder.write(frame)
    recorder.write(frame)
    recorder.toggle()
    recorder.write(frame)
    capture = cv2.VideoCapture(str(first))
    try:
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(12.0)
        for _ in range(2):
            ok, decoded = capture.read()
            assert ok
            assert decoded.shape == frame.shape
            assert np.abs(decoded.astype(float) - frame).max() < 10
        assert not capture.read()[0]
    finally:
        capture.release()

    recorder.toggle()
    second = recorder.path
    assert second is not None and second != first
    recorder.write(frame)
    recorder.close()
    recorder.close()
    capture = cv2.VideoCapture(str(second))
    try:
        assert capture.isOpened()
        assert capture.read()[0]
        assert not capture.read()[0]
    finally:
        capture.release()
    assert len(list(root.glob("*.mp4"))) == 2


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf")])
def test_recording_falls_back_to_valid_fps(tmp_path: Path, fps: float) -> None:
    assert FrameRecorder(tmp_path, fps).fps == 30.0


def test_recording_reports_failed_open(tmp_path: Path, monkeypatch) -> None:
    recorder = FrameRecorder(tmp_path, fps=30.0)
    recorder.toggle()
    unopened = cv2.VideoWriter()
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: unopened)
    with pytest.raises(RuntimeError, match="Unable to open recording video"):
        recorder.write(np.zeros((32, 32, 3), dtype=np.uint8))
    assert recorder.frame_count == 0
    assert recorder.writer is None
    recorder.close()
