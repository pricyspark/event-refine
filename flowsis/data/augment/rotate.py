from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image

from ..masks import mask2xywh
from .common import mask_union
from flowsis.utils.common import init_rng


def sample_rotation_angle(
    width: int,
    height: int,
    rng: np.random.Generator,
    crop_loss_avoidance: float = 0.5,
    max_tries: int = 100,
    min_angle: float = -22.5,
    max_angle: float = 22.5,
    **kwargs,
) -> tuple[float, int, int, float, float]:
    if not 0 <= crop_loss_avoidance <= 1:
        raise ValueError("crop_loss_avoidance must be in [0, 1].")

    if not -360 <= min_angle <= max_angle <= 360:
        raise ValueError(
            "Rotation angles must satisfy -360 <= min_angle <= max_angle <= 360."
        )

    old_area = width * height
    for _ in range(max_tries):
        angle = rng.uniform(min_angle, max_angle)

        width_crop, height_crop, dx, dy = rotate_crop_bounds(
            width,
            height,
            math.radians(angle),
        )

        width_crop, height_crop = round(width_crop), round(height_crop)

        new_area = width_crop * height_crop
        kept_ratio = new_area / old_area

        accept_prob = (1.0 - crop_loss_avoidance) + crop_loss_avoidance * kept_ratio

        if rng.random() < accept_prob:
            return angle, width_crop, height_crop, dx, dy

    angle = min_angle
    width_crop, height_crop, dx, dy = rotate_crop_bounds(
        width,
        height,
        math.radians(angle),
    )

    width_crop, height_crop = round(width_crop), round(height_crop)
    return angle, width_crop, height_crop, dx, dy


def rotation_augment(example: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    rng = init_rng(kwargs.get("rng", None), kwargs.get("seed", None))

    img: Image.Image = example["image"]
    objects = example["objects"]
    width, height = img.size

    angle, width_crop, height_crop, dx, dy = sample_rotation_angle(
        width, height, rng, **kwargs
    )
    img_rot = img.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=0,
    )
    width_rot, height_rot = img_rot.size

    masks = (
        np.stack([obj["mask"] for obj in objects]).astype(np.bool_, copy=False)
        if objects
        else np.zeros((0, height, width), dtype=np.bool_)
    )
    offset = np.zeros(2, dtype=np.float64)
    if objects:
        m_union_img = Image.fromarray(mask_union(masks).astype(np.uint8, copy=False))
        m_union_rot = (
            np.asarray(
                m_union_img.rotate(
                    angle,
                    resample=Image.Resampling.NEAREST,
                    expand=True,
                    fillcolor=0,
                )
            )
            != 0
        )
        union_bbox = mask2xywh(m_union_rot)
        if union_bbox is not None:
            x, y, w, h = union_bbox
            bbox_center = np.array((x + w / 2, y + h / 2))
            frame_center = np.array((width_rot / 2, height_rot / 2))
            offset = bbox_center - frame_center

    offset_proj = capped_projection(offset, (dx, dy))
    offset_proj = np.trunc(offset_proj).astype(np.int32)

    width_diff = width_rot - width_crop
    height_diff = height_rot - height_crop
    col_start = width_diff // 2 + offset_proj[0]
    row_start = height_diff // 2 + offset_proj[1]
    col_end = col_start + width_crop
    row_end = row_start + height_crop
    if "pad" in kwargs:
        row_start += kwargs["pad"]
        col_start += kwargs["pad"]
        row_end -= kwargs["pad"]
        col_end -= kwargs["pad"]

    kept_objects = []
    # Rotate, crop, and update individual objects
    for obj, mask in zip(objects, masks):
        mask_uint8 = mask.astype(np.uint8, copy=False)
        mask_img = Image.fromarray(mask_uint8)
        mask_img_rot = mask_img.rotate(
            angle,
            resample=Image.Resampling.NEAREST,
            expand=True,
            fillcolor=0,
        )
        mask_img_crop = mask_img_rot.crop((col_start, row_start, col_end, row_end))
        mask_crop = np.asarray(mask_img_crop) != 0
        bbox = mask2xywh(mask_crop)
        if bbox is None:
            continue

        obj["mask"] = mask_crop
        obj["bbox"] = bbox
        obj["area"] = float(np.count_nonzero(mask_crop))
        obj["modified"] = True

        kept_objects.append(obj)

    img_crop = img_rot.crop((col_start, row_start, col_end, row_end))
    example["image"] = img_crop
    example["objects"] = kept_objects
    example["width"], example["height"] = img_crop.size
    example["modified"] = True

    return example


def rotate_crop_bounds(
    w: int, h: int, theta: float
) -> tuple[float, float, float, float]:
    theta = theta % math.pi
    reflect = False
    if theta >= math.pi / 2:
        theta = math.pi - theta
        reflect = True

    if w == h:
        side = w / (math.cos(theta) + math.sin(theta))
        return side, side, 0.0, 0.0
    if theta == 0:
        return float(w), float(h), 0.0, 0.0

    aspect_ratio = w / h
    c = math.cos(theta)
    s = math.sin(theta)
    if theta < math.pi / 3:
        bound = math.sin(2 * theta)
        if aspect_ratio < bound:
            x = w / (2 * c)
            y = w / (2 * s)

            dx = (h * s - x) / 2
            dy = (h * c - y) / 2
        elif aspect_ratio <= 1 / bound:
            denom = c * c - s * s
            x = (w * c - h * s) / denom
            y = (h * c - w * s) / denom
            dx, dy = 0, 0
        else:
            x = h / (2 * s)
            y = h / (2 * c)

            dx = (w * c - x) / 2
            dy = (w * s - y) / 2
    else:
        t = math.tan(theta / 2)
        k = s + c * t
        if aspect_ratio < 1 / k:
            x = w
            y = w * math.tan(theta / 2)

            dx = (h - w * k) * s / 2
            dy = (h - w * k) * c / 2
        elif aspect_ratio <= k:
            denom = c * c - s * s
            x = (w * c - h * s) / denom
            y = (h * c - w * s) / denom
            dx, dy = 0, 0
        else:
            x = h * math.tan(theta / 2)
            y = h

            dx = (w - h * k) * c / 2
            dy = (w - h * k) * s / 2

    dy *= -1 if reflect else 1
    return x, y, dx, dy


def capped_projection(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.all(b == 0):
        return np.zeros_like(b)

    ab = np.sum(a * b, axis=-1, keepdims=True)
    bb = np.sum(b * b, axis=-1, keepdims=True)
    coef = np.divide(
        ab,
        bb,
        out=np.zeros_like(bb),
        where=bb != 0,
    )
    coef = np.clip(coef, -1, 1)
    return coef * b
