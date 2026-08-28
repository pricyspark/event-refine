import numpy as np

from flowsis.cli.deploy.live_base import resize_square_frame


def test_resize_square_frame_uses_requested_size() -> None:
    frame = np.zeros((480, 480, 3), dtype=np.uint8)

    resized = resize_square_frame(frame, 640)

    assert resized.shape == (640, 640, 3)


def test_resize_square_frame_preserves_frame_without_size() -> None:
    frame = np.zeros((480, 480, 3), dtype=np.uint8)

    resized = resize_square_frame(frame, None)

    assert resized is frame
