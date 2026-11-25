"""Explicit event representations and shared RGB/event geometry.

Developed with Codex assistance.
"""

from pathlib import Path
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image


def accumulate_events(
    x: NDArray[np.integer],
    y: NDArray[np.integer],
    timestamps: NDArray[np.number],
    *,
    start: float,
    end: float,
    height: int,
    width: int,
) -> NDArray[np.bool_]:
    """Binary occupancy in (start, end]; timestamps must share the same units.

    Coordinates must already be registered to the RGB/annotation coordinate
    system. Polarity is intentionally discarded for this four-channel baseline.
    """
    if not np.isfinite([start, end]).all() or start >= end:
        raise ValueError("Expected a finite interval with start < end.")
    if min(height, width) <= 0:
        raise ValueError("Event image dimensions must be positive.")
    if x.ndim != 1 or x.shape != y.shape or x.shape != timestamps.shape:
        raise ValueError("Event x, y, and timestamps must be aligned 1-D arrays.")
    if not np.issubdtype(x.dtype, np.integer) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Event coordinates must be integers.")
    if not np.isfinite(timestamps).all():
        raise ValueError("Event timestamps must be finite.")
    selected = (timestamps > start) & (timestamps <= end)
    event_x, event_y = x[selected], y[selected]
    if ((event_x < 0) | (event_x >= width) | (event_y < 0) | (event_y >= height)).any():
        raise ValueError("Event coordinates are outside the registered image.")
    occupancy = np.zeros((height, width), dtype=bool)
    occupancy[event_y, event_x] = True
    return occupancy


def save_event_map(path: str | Path, events: NDArray[np.bool_]) -> Path:
    """Atomically save boolean occupancy as NPY or bit-packed compressed NPZ."""
    path = Path(path)
    if path.suffix not in {".npy", ".npz"}:
        raise ValueError("Event maps must use .npy or .npz.")
    if events.dtype != np.bool_ or events.ndim != 2 or min(events.shape) <= 0:
        raise ValueError("Expected a nonempty boolean HxW event map.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            if path.suffix == ".npz":
                np.savez_compressed(
                    handle,
                    format_version=np.asarray(1, dtype=np.int64),
                    shape=np.asarray(events.shape, dtype=np.int64),
                    bitorder=np.asarray("big"),
                    packed=np.packbits(events.reshape(-1), bitorder="big"),
                )
            else:
                np.save(handle, events, allow_pickle=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def _load_packed_events(path: Path) -> NDArray[np.bool_]:
    with np.load(path, allow_pickle=False) as data:
        if not {"format_version", "shape", "bitorder", "packed"}.issubset(data.files):
            raise ValueError(f"Missing packed event metadata: {path}")
        version = data["format_version"]
        shape = data["shape"]
        bitorder = data["bitorder"]
        packed = data["packed"]
        if version.shape != () or version.dtype.kind not in "iu" or int(version) != 1:
            raise ValueError(f"Unsupported packed event format: {path}")
        if (
            shape.shape != (2,)
            or shape.dtype.kind not in "iu"
            or bool((shape <= 0).any())
        ):
            raise ValueError(f"Invalid packed event shape: {path}")
        if bitorder.shape != () or str(bitorder) != "big":
            raise ValueError(f"Unsupported packed event bit order: {path}")
        height, width = int(shape[0]), int(shape[1])
        count = height * width
        if packed.dtype != np.uint8 or packed.shape != ((count + 7) // 8,):
            raise ValueError(f"Packed event payload does not match shape: {path}")
        return (
            np.unpackbits(packed, count=count, bitorder="big")
            .reshape(height, width)
            .astype(np.bool_, copy=False)
        )


def load_event_map(
    path: Path,
    *,
    encoding: str = "binary",
    background: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Load [1,H,W] occupancy; rendered frames require explicit background RGB.

    A rendered visualization only recovers occupied pixels if the producer's
    encoding represents every event by a non-background pixel.
    """
    if encoding == "binary":
        if path.suffix == ".npz":
            array = _load_packed_events(path)
        elif path.suffix == ".npy":
            array = np.load(path, allow_pickle=False)
        else:
            with Image.open(path) as image:
                array = np.asarray(image).copy()
        if array.ndim != 2 or (
            array.dtype != np.bool_ and not np.isin(array, [0, 1, 255]).all()
        ):
            raise ValueError(f"Expected a binary HxW occupancy map: {path}")
        active = array if array.dtype == np.bool_ else array != 0
    elif encoding == "rendered":
        if (
            background is None
            or len(background) != 3
            or any(value < 0 or value > 255 for value in background)
        ):
            raise ValueError(
                "Rendered events require an explicit three-value RGB background."
            )
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
        active = (array != np.asarray(background)).any(axis=-1)
    else:
        raise ValueError(f"Unsupported event encoding: {encoding!r}")
    return torch.from_numpy(active.copy()).unsqueeze(0).float()


def prepare_inputs(
    rgb: torch.Tensor,
    events: torch.Tensor,
    *,
    image_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    rgb_scale: float = 1.0,
    validate_values: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize/pad aligned RGB [B,3,H,W] in [0,1] and binary [B,1,H,W].

    Downsample event occupancy with max pooling to retain sparse activity;
    upsample it with nearest-neighbor interpolation. Normalize RGB alone.
    Disable value checks only for canonical inputs validated by the data loader.
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("RGB must have shape [B,3,H,W].")
    if events.shape != (rgb.shape[0], 1, *rgb.shape[-2:]):
        raise ValueError("Events must have shape [B,1,H,W] aligned to RGB.")
    if rgb.device != events.device or not rgb.is_floating_point():
        raise ValueError("RGB must be floating point and share the event device.")
    if image_size <= 0 or mean.numel() != 3 or std.numel() != 3:
        raise ValueError("Invalid image size or RGB normalization.")
    if validate_values:
        if not torch.isfinite(rgb).all() or bool(((rgb < 0) | (rgb > 1)).any()):
            raise ValueError("RGB values must be finite and in [0,1].")
        if not bool(((events == 0) | (events == 1)).all()):
            raise ValueError("Events must be binary occupancy (0 or 1).")
        if bool((std <= 0).any()):
            raise ValueError("Invalid image size or RGB normalization.")
    height, width = rgb.shape[-2:]
    if min(height, width) <= 0:
        raise ValueError("Input spatial dimensions must be positive.")
    scale = image_size / max(height, width)
    size = (max(1, round(height * scale)), max(1, round(width * scale)))
    rgb = F.interpolate(
        rgb.float(), size=size, mode="bilinear", align_corners=False, antialias=True
    )
    events = events.float()
    if size[0] <= height and size[1] <= width:
        events = F.adaptive_max_pool2d(events, size)
    else:
        events = F.interpolate(events, size=size, mode="nearest")
    rgb = (rgb * rgb_scale - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    padding = (0, image_size - size[1], 0, image_size - size[0])
    pixels = F.pad(torch.cat((rgb, events), dim=1), padding)
    mask = F.pad(torch.ones_like(events[:, 0], dtype=torch.long), padding)
    return pixels, mask
