from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from flowsis.pretrained.detector import BaseDetector, DetectorImage
from flowsis.set_mask import (
    BasePrediction,
    SetMaskHead,
    SetMaskOutput,
    postprocess_set_masks,
)
from flowsis.utils import build_autocast_context


class FlowSISBase(nn.Module):
    """Stateless detector and set-mask model."""

    def __init__(
        self,
        detector: BaseDetector,
        head: SetMaskHead,
        label_names: Sequence[str],
        *,
        head_config: dict[str, Any],
        detection_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        max_detections: int | None = None,
        use_amp: bool = True,
        detector_amp: bool = False,
        device_preprocess: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= detection_threshold <= 1.0:
            raise ValueError("Detection threshold must be between zero and one.")
        if not 0.0 <= mask_threshold <= 1.0:
            raise ValueError("Mask threshold must be between zero and one.")
        if max_detections is not None and max_detections <= 0:
            raise ValueError("Maximum detections must be positive when provided.")

        labels = tuple(label_names)
        detector_labels = tuple(
            detector.label_names[index] for index in sorted(detector.label_names)
        )
        if labels != detector_labels:
            raise ValueError(
                "Prompt-bank label order must exactly match detector labels: "
                f"{labels!r} != {detector_labels!r}."
            )
        if len(labels) != head.query_text_fusion.num_classes:
            raise ValueError(
                "Prompt-bank class count does not match the detector label count."
            )

        self.detector = detector
        self.head = head
        self.label_names = labels
        self.head_config = dict(head_config)
        self.detection_threshold = float(detection_threshold)
        self.mask_threshold = float(mask_threshold)
        self.max_detections = max_detections
        self.use_amp = bool(use_amp)
        self.detector_amp = bool(detector_amp)
        self.device_preprocess = bool(device_preprocess)

    @property
    def device(self) -> torch.device:
        return next(self.head.parameters()).device

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        pixel_mask: torch.Tensor | None = None,
        labels: list[dict[str, torch.Tensor]] | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> SetMaskOutput:
        with self.detector_autocast_context():
            detector_output = self.detector.forward_features(
                pixel_values,
                pixel_mask=pixel_mask,
                labels=labels,
            )
        return self.head(detector_output, class_weights=class_weights)

    def detector_autocast_context(self) -> AbstractContextManager[None]:
        """Use FP32 detector math by default, even inside an AMP mask-head pass."""
        if self.device.type != "cuda":
            return nullcontext()
        return torch.autocast("cuda", enabled=self.detector_amp)

    @staticmethod
    def _image_list(
        images: DetectorImage | Sequence[DetectorImage],
    ) -> list[DetectorImage]:
        if isinstance(images, torch.Tensor) and images.ndim == 4:
            return list(images)
        if isinstance(images, (list, tuple)):
            return list(images)
        return [images]

    @staticmethod
    def _image_size(image: DetectorImage) -> tuple[int, int]:
        if isinstance(image, torch.Tensor):
            return int(image.shape[-2]), int(image.shape[-1])
        if isinstance(image, bytes):
            raise TypeError(
                "Encoded image bytes are not supported by FlowSISBase.infer."
            )
        if isinstance(image, Image.Image):
            width, height = image.size
            return int(height), int(width)
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Unsupported image type: {type(image).__name__}.")
        return int(image.shape[0]), int(image.shape[1])

    @torch.inference_mode()
    def infer(
        self,
        images: DetectorImage | Sequence[DetectorImage],
        *,
        detection_threshold: float | None = None,
        max_detections: int | None = None,
    ) -> list[BasePrediction]:
        image_list = self._image_list(images)
        image_sizes = [self._image_size(image) for image in image_list]
        inputs = self.detector.preprocess(
            image_list,
            device_preprocess=self.device_preprocess,
        )
        pixel_values = inputs["pixel_values"]
        if not isinstance(pixel_values, torch.Tensor):
            raise RuntimeError("Detector preprocessing did not return pixel values.")
        pixel_values = pixel_values.to(self.device)
        pixel_mask = inputs.get("pixel_mask")
        if isinstance(pixel_mask, torch.Tensor):
            pixel_mask = pixel_mask.to(self.device)
        else:
            pixel_mask = None

        was_training = self.training
        self.eval()
        try:
            with build_autocast_context(enabled=self.use_amp, device=self.device):
                output = self(pixel_values, pixel_mask=pixel_mask)
        finally:
            self.train(was_training)
        return postprocess_set_masks(
            output,
            image_sizes,
            detection_threshold=(
                self.detection_threshold
                if detection_threshold is None
                else detection_threshold
            ),
            max_detections=(
                self.max_detections if max_detections is None else max_detections
            ),
        )

    @torch.inference_mode()
    def infer_frame(
        self,
        bgr_frame: NDArray,
        *,
        detection_threshold: float | None = None,
        max_detections: int | None = None,
    ) -> BasePrediction:
        if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
            raise ValueError(f"Expected an HWC BGR frame, got {bgr_frame.shape}.")
        rgb_frame = bgr_frame[..., ::-1].copy()
        return self.infer(
            rgb_frame,
            detection_threshold=detection_threshold,
            max_detections=max_detections,
        )[0]

    def binarize_logits(
        self,
        mask_logits: torch.Tensor,
        *,
        threshold: float | None = None,
    ) -> torch.Tensor:
        resolved_threshold = self.mask_threshold if threshold is None else threshold
        if not 0.0 <= resolved_threshold <= 1.0:
            raise ValueError("Mask threshold must be between zero and one.")
        return mask_logits.sigmoid() >= resolved_threshold

    @torch.inference_mode()
    def infer_union(
        self,
        bgr_frame: NDArray,
        *,
        detection_threshold: float | None = None,
        mask_threshold: float | None = None,
    ) -> torch.Tensor | None:
        prediction = self.infer_frame(
            bgr_frame,
            detection_threshold=detection_threshold,
        )
        if prediction.union_logits is None:
            return None
        return self.binarize_logits(prediction.union_logits, threshold=mask_threshold)


__all__ = ["FlowSISBase"]
