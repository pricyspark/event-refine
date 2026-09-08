from __future__ import annotations

import copy
import random
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage

from ..masks import mask2xywh
from .common import mask_union

from flowsis.utils.common import init_rng

from ..classes import SampleContext


def _count_connected_components(
    mask: NDArray[np.bool_], connectivity: Literal[4, 8] = 8
) -> int:
    if connectivity == 4:
        structure = ndimage.generate_binary_structure(2, 1)
    elif connectivity == 8:
        structure = ndimage.generate_binary_structure(2, 2)
    else:
        raise ValueError("Connectivity must be 4 or 8")

    result = cast(
        tuple[NDArray[np.int_], int],
        ndimage.label(mask, structure=structure),
    )

    return result[1]


def _verify_overlap_example(
    example: dict[str, Any],
    min_visible_pixels: int,
    connectivity: Literal[4, 8] = 8,
    masks: NDArray[np.bool_] | None = None,
    visible_masks: NDArray[np.bool_] | None = None,
) -> bool:
    if min_visible_pixels <= 0:
        raise ValueError("min_visible_pixels must be positive.")
    objects = example["objects"]
    if masks is None:
        masks = np.array([obj["mask"] for obj in objects], dtype=np.bool_)
    if visible_masks is None:
        visible_masks = np.array(
            [obj["visible_mask"] for obj in objects], dtype=np.bool_
        )
    if len(objects) == 0:
        return True
    if len(objects) != len(masks) or len(objects) != len(visible_masks):
        raise ValueError("Objects, masks, and visible_masks must have the same length.")
    if masks.ndim != 3 or visible_masks.ndim != 3:
        raise ValueError(
            "Masks and visible_masks must be shaped as (num_objects, height, width)."
        )

    visible_area = np.sum(visible_masks, axis=(1, 2), dtype=np.int64)
    total_area = np.sum(masks, axis=(1, 2), dtype=np.int64)
    visible_ratios = np.divide(
        visible_area,
        total_area,
        out=np.zeros(len(masks), dtype=np.float32),
        where=total_area > 0,
    )

    kept_objects = []
    for i, (obj, visible_mask) in enumerate(zip(objects, visible_masks, strict=True)):
        if visible_area[i] < min_visible_pixels:
            continue
        bbox = mask2xywh(visible_mask)
        assert bbox is not None
        obj["bbox"] = bbox
        obj["area"] = bbox[2] * bbox[3]
        obj["mask"] = visible_mask
        obj["visible_fraction"] = float(visible_ratios[i])
        obj["visible_pixels"] = int(visible_area[i])
        obj["visible_components"] = _count_connected_components(
            visible_mask, connectivity
        )
        obj["synthetic_occlusion"] = True
        obj.pop("visible_mask", None)
        obj["modified"] = True

        kept_objects.append(obj)

    if not kept_objects:
        example["objects"] = []
        return False

    example["objects"] = kept_objects
    return True


def overlap_augment(
    example: dict[str, Any], *, context: SampleContext, **kwargs
) -> dict[str, Any]:
    rng = init_rng(kwargs.get("rng", None), kwargs.get("seed", None))

    if context is None:
        raise ValueError(
            "overlap_augment requires kwargs['augmentation_context']. "
            "Use it through TransformDataset/AugmentationPipeline with an indexable dataset."
        )

    img: Image.Image = example["image"]
    objects = example["objects"]
    original_image = img.copy()
    original_objects = copy.deepcopy(objects)
    height = example["height"]
    width = example["width"]

    base_masks = np.array([obj["mask"] for obj in objects], dtype=np.bool_)

    min_overlay = kwargs.get("min_overlays", 0)
    max_overlay = kwargs.get("max_overlays", min_overlay)
    p = kwargs.get("p", 0.5)
    if max_overlay < min_overlay:
        raise ValueError("max_overlays must be greater than or equal to min_overlays.")
    if not 0 <= p < 1:
        raise ValueError("p must satisfy 0 <= p < 1.")

    num_additional_samples = min(rng.geometric(1 - p) - 1, max_overlay - min_overlay)
    num_samples = min_overlay + num_additional_samples

    overlay_examples = context.sample_examples(num_samples, rng=rng)
    overlay_prepare = kwargs.get("overlay_prepare")

    currently_blocked_mask = np.zeros((height, width), dtype=np.bool_)

    all_objects: list[dict[str, Any]] = []
    all_masks: list[NDArray[np.bool_]] = []
    all_visible_masks: list[NDArray[np.bool_]] = []

    for overlay_index, overlay_example in enumerate(overlay_examples):
        if overlay_prepare is not None:
            prepare_kwargs: dict[str, Any] = {}
            if "seed" in kwargs and kwargs["seed"] is not None:
                prepare_kwargs["seed"] = int(kwargs["seed"]) * 1000 + overlay_index + 1
            prepared_overlay = overlay_prepare(overlay_example, **prepare_kwargs)
            if prepared_overlay is not None:
                overlay_example = prepared_overlay

        overlay_img = overlay_example["image"]
        overlay_objects = overlay_example["objects"]
        overlay_masks = np.array(
            [obj["mask"] for obj in overlay_objects], dtype=np.bool_
        )
        if overlay_img.width != width or overlay_img.height != height:
            raise ValueError(
                "Overlay example image size must match the target canvas size."
            )
        if overlay_masks.ndim != 3:
            raise ValueError(
                "Overlay masks must be shaped as (num_objects, height, width)."
            )
        if overlay_masks.shape[1:] != (height, width):
            raise ValueError("Overlay masks must match the target canvas size.")

        currently_visible_mask = ~currently_blocked_mask
        visible_overlay_masks = overlay_masks & currently_visible_mask

        for obj, full_mask, visible_mask in zip(
            overlay_objects, overlay_masks, visible_overlay_masks
        ):
            obj["visible_mask"] = visible_mask
            all_objects.append(obj)
            all_masks.append(full_mask)
            all_visible_masks.append(visible_mask)

        overlay_mask_union = mask_union(overlay_masks)
        currently_visible_overlay_mask = overlay_mask_union & currently_visible_mask
        mask_pil = Image.fromarray(
            currently_visible_overlay_mask.astype(np.uint8) * 255,
            mode="L",
        )
        img.paste(overlay_img, (0, 0), mask_pil)
        currently_blocked_mask |= overlay_mask_union

    currently_visible_mask = ~currently_blocked_mask
    visible_base_masks = base_masks & currently_visible_mask
    for obj, full_mask, visible_mask in zip(objects, base_masks, visible_base_masks):
        obj["visible_mask"] = visible_mask
        all_objects.append(obj)
        all_masks.append(full_mask)
        all_visible_masks.append(visible_mask)

    example["objects"] = all_objects

    min_visible_pixels = kwargs.get("min_visible_pixels", 16)
    connectivity = kwargs.get("connectivity", 8)

    is_valid = _verify_overlap_example(
        example,
        min_visible_pixels=min_visible_pixels,
        connectivity=connectivity,
        masks=np.asarray(all_masks, dtype=np.bool_),
        visible_masks=np.asarray(all_visible_masks, dtype=np.bool_),
    )
    if is_valid:
        severe_threshold = float(kwargs.get("severe_visible_fraction", 0.25))
        severe_probability = float(kwargs.get("severe_occlusion_probability", 0.2))
        if not 0.0 <= severe_threshold <= 1.0 or not 0.0 <= severe_probability <= 1.0:
            raise ValueError("Overlap severity controls must be probabilities.")
        is_severe = any(
            obj.get("visible_fraction", 1.0) < severe_threshold
            for obj in example["objects"]
        )
        if is_severe and rng.random() >= severe_probability:
            is_valid = False
    if not is_valid:
        example["image"] = original_image
        example["objects"] = original_objects

    return example
