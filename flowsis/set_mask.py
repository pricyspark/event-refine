from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowsis.pretrained import DetectorFeatureOutput

TextConditioning = Literal["frozen", "learned", "none"]


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


class _RefinementBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.block(features)


class PixelDecoder(nn.Module):
    """Top-down convolutional pyramid producing shared mask pixel features."""

    def __init__(
        self,
        detail_channels: int,
        feature_channels: tuple[int, ...],
        *,
        mask_dim: int = 256,
        use_detail_features: bool = True,
    ) -> None:
        super().__init__()
        if detail_channels <= 0 or not feature_channels or mask_dim <= 0:
            raise ValueError("Pixel decoder channel dimensions must be positive.")
        self.mask_dim = int(mask_dim)
        self.use_detail_features = bool(use_detail_features)
        self.lateral = nn.ModuleList(
            nn.Conv2d(channels, mask_dim, kernel_size=1)
            for channels in feature_channels
        )
        self.refinement = nn.ModuleList(
            _RefinementBlock(mask_dim) for _ in feature_channels
        )
        self.detail_projection = nn.Conv2d(detail_channels, mask_dim, kernel_size=1)
        self.detail_refinement = _RefinementBlock(mask_dim)

    def forward(
        self,
        detail_features: torch.Tensor,
        feature_maps: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if len(feature_maps) != len(self.lateral):
            raise ValueError(
                f"Expected {len(self.lateral)} detector feature levels, "
                f"received {len(feature_maps)}."
            )
        projected = [
            projection(features)
            for projection, features in zip(self.lateral, feature_maps, strict=True)
        ]
        top_down: torch.Tensor | None = None
        for level in range(len(projected) - 1, -1, -1):
            current = projected[level]
            if top_down is not None:
                top_down = F.interpolate(
                    top_down,
                    size=current.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                current = current + top_down
            top_down = self.refinement[level](current)
        assert top_down is not None
        if not self.use_detail_features:
            return top_down
        semantic = F.interpolate(
            top_down,
            size=detail_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.detail_refinement(
            semantic + self.detail_projection(detail_features)
        )


class QueryTextFusion(nn.Module):
    """Turn detector queries and class prompt mixtures into mask embeddings."""

    def __init__(
        self,
        query_dim: int,
        prompt_embeddings: torch.Tensor,
        *,
        mask_dim: int = 256,
        conditioning: TextConditioning = "frozen",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if prompt_embeddings.ndim != 3:
            raise ValueError("Prompt embeddings must be shaped [C,P,D].")
        if prompt_embeddings.shape[0] <= 0 or prompt_embeddings.shape[1] <= 0:
            raise ValueError(
                "Prompt embeddings cannot have empty class or prompt axes."
            )
        if temperature <= 0.0:
            raise ValueError("Text temperature must be positive.")
        if conditioning not in {"frozen", "learned", "none"}:
            raise ValueError(f"Unsupported text conditioning mode {conditioning!r}.")

        self.query_dim = int(query_dim)
        self.mask_dim = int(mask_dim)
        self.text_dim = int(prompt_embeddings.shape[-1])
        self.num_classes = int(prompt_embeddings.shape[0])
        self.num_prompts = int(prompt_embeddings.shape[1])
        self.conditioning = conditioning
        self.temperature = float(temperature)
        self.register_buffer(
            "prompt_embeddings",
            prompt_embeddings.detach().float().clone(),
        )

        self.query_norm = nn.LayerNorm(query_dim)
        self.query_projection = nn.Linear(query_dim, mask_dim)
        self.text_norm = nn.LayerNorm(self.text_dim)
        self.text_projection = nn.Linear(self.text_dim, mask_dim)
        self.learned_classes = nn.Embedding(self.num_classes, mask_dim)
        self.text_gate = nn.Linear(mask_dim * 2, mask_dim)
        self.text_delta = nn.Linear(mask_dim * 2, mask_dim)
        self.fusion_norm = nn.LayerNorm(mask_dim)
        self.mask_mlp = nn.Sequential(
            nn.LayerNorm(mask_dim),
            nn.Linear(mask_dim, mask_dim * 4),
            nn.GELU(),
            nn.Linear(mask_dim * 4, mask_dim),
        )
        nn.init.zeros_(self.text_delta.weight)
        nn.init.zeros_(self.text_delta.bias)

    def predicted_class_weights(self, class_logits: torch.Tensor) -> torch.Tensor:
        if class_logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} class logits, "
                f"received {class_logits.shape[-1]}."
            )
        return torch.softmax(class_logits / self.temperature, dim=-1)

    def _frozen_text(
        self,
        query_features: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> torch.Tensor:
        mixed_prompts = torch.einsum(
            "bqc,cpd->bqpd",
            class_weights,
            self.prompt_embeddings.to(class_weights.dtype),
        )
        text_tokens = self.text_projection(self.text_norm(mixed_prompts))
        attention_logits = torch.einsum(
            "bqd,bqpd->bqp", query_features, text_tokens
        ) / math.sqrt(self.mask_dim)
        attention = torch.softmax(attention_logits, dim=-1)
        return torch.einsum("bqp,bqpd->bqd", attention, text_tokens)

    def _learned_text(self, class_weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "bqc,cd->bqd",
            class_weights,
            self.learned_classes.weight,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        class_logits: torch.Tensor,
        *,
        class_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_embeddings.ndim != 3 or query_embeddings.shape[-1] != self.query_dim:
            raise ValueError(
                f"Expected queries shaped [B,Q,{self.query_dim}], "
                f"got {tuple(query_embeddings.shape)}."
            )
        if class_weights is None:
            class_weights = self.predicted_class_weights(class_logits)
        if class_weights.shape != class_logits.shape:
            raise ValueError(
                "Class weights and detector logits must have equal shapes."
            )

        query_features = self.query_projection(self.query_norm(query_embeddings))
        if self.conditioning == "frozen":
            text_features = self._frozen_text(query_features, class_weights)
        elif self.conditioning == "learned":
            text_features = self._learned_text(class_weights)
        else:
            conditioned = self.fusion_norm(query_features)
            return conditioned + self.mask_mlp(conditioned), class_weights

        fusion_input = torch.cat((query_features, text_features), dim=-1)
        gate = torch.sigmoid(self.text_gate(fusion_input))
        conditioned = self.fusion_norm(
            query_features + gate * self.text_delta(fusion_input)
        )
        return conditioned + self.mask_mlp(conditioned), class_weights


@dataclass
class SetMaskOutput:
    mask_logits: torch.Tensor
    pixel_features: torch.Tensor
    mask_embeddings: torch.Tensor
    class_weights: torch.Tensor
    detector: DetectorFeatureOutput
    matches: torch.Tensor | None = None


class SetMaskHead(nn.Module):
    def __init__(
        self,
        detail_channels: int,
        feature_channels: tuple[int, ...],
        query_dim: int,
        prompt_embeddings: torch.Tensor,
        *,
        mask_dim: int = 256,
        text_conditioning: TextConditioning = "frozen",
        text_temperature: float = 1.0,
        use_detail_features: bool = True,
        use_box_prior: bool = True,
        box_expansion: float = 1.2,
        box_sharpness: float = 30.0,
        box_prior_initial_strength: float = 0.25,
        box_prior_max_strength: float = 1.0,
    ) -> None:
        super().__init__()
        if box_expansion <= 0.0 or box_sharpness <= 0.0:
            raise ValueError("Box expansion and sharpness must be positive.")
        if not 0.0 <= box_prior_initial_strength <= box_prior_max_strength:
            raise ValueError("Initial box-prior strength is outside its bounds.")
        self.pixel_decoder = PixelDecoder(
            detail_channels,
            feature_channels,
            mask_dim=mask_dim,
            use_detail_features=use_detail_features,
        )
        self.query_text_fusion = QueryTextFusion(
            query_dim,
            prompt_embeddings,
            mask_dim=mask_dim,
            conditioning=text_conditioning,
            temperature=text_temperature,
        )
        self.mask_dim = int(mask_dim)
        self.use_box_prior = bool(use_box_prior)
        self.box_expansion = float(box_expansion)
        self.box_sharpness = float(box_sharpness)
        self.box_prior_max_strength = float(box_prior_max_strength)
        ratio = box_prior_initial_strength / max(box_prior_max_strength, 1e-8)
        if ratio <= 0.0:
            initial_logit = -12.0
        elif ratio >= 1.0:
            initial_logit = 12.0
        else:
            initial_logit = math.log(ratio / (1.0 - ratio))
        self.box_prior_logit = nn.Parameter(torch.tensor(initial_logit))

    @property
    def prompt_embeddings(self) -> torch.Tensor:
        return self.query_text_fusion.prompt_embeddings

    @property
    def box_prior_strength(self) -> torch.Tensor:
        if not self.use_box_prior:
            return self.box_prior_logit.new_zeros(())
        return self.box_prior_max_strength * torch.sigmoid(self.box_prior_logit)

    @staticmethod
    def center_to_corners(boxes: torch.Tensor) -> torch.Tensor:
        center, size = boxes.split(2, dim=-1)
        half_size = 0.5 * size
        return torch.cat((center - half_size, center + half_size), dim=-1)

    def soft_box_gates(
        self,
        boxes: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        corners = self.center_to_corners(boxes)
        center = 0.5 * (corners[..., :2] + corners[..., 2:])
        size = (corners[..., 2:] - corners[..., :2]) * self.box_expansion
        corners = torch.cat((center - 0.5 * size, center + 0.5 * size), dim=-1).clamp(
            0.0, 1.0
        )
        x1, y1, x2, y2 = corners.unbind(dim=-1)
        y = (
            torch.arange(height, device=boxes.device, dtype=boxes.dtype) + 0.5
        ) / height
        x = (torch.arange(width, device=boxes.device, dtype=boxes.dtype) + 0.5) / width
        x = x.view(1, 1, 1, width)
        y = y.view(1, 1, height, 1)
        inside_x = torch.sigmoid(self.box_sharpness * (x - x1[..., None, None]))
        inside_x = inside_x * torch.sigmoid(
            self.box_sharpness * (x2[..., None, None] - x)
        )
        inside_y = torch.sigmoid(self.box_sharpness * (y - y1[..., None, None]))
        inside_y = inside_y * torch.sigmoid(
            self.box_sharpness * (y2[..., None, None] - y)
        )
        return inside_x * inside_y

    def forward(
        self,
        detector: DetectorFeatureOutput,
        *,
        class_weights: torch.Tensor | None = None,
    ) -> SetMaskOutput:
        pixel_features = self.pixel_decoder(
            detector.detail_features,
            detector.feature_maps,
        )
        mask_embeddings, resolved_weights = self.query_text_fusion(
            detector.query_embeddings,
            detector.query_logits,
            class_weights=class_weights,
        )
        mask_logits = torch.einsum("bqd,bdhw->bqhw", mask_embeddings, pixel_features)
        if self.use_box_prior:
            gates = self.soft_box_gates(
                detector.query_boxes,
                height=mask_logits.shape[-2],
                width=mask_logits.shape[-1],
            )
            prior = torch.log(gates.float().clamp_min(1e-4)).to(mask_logits.dtype)
            mask_logits = (
                mask_logits + self.box_prior_strength.to(mask_logits.dtype) * prior
            )
        return SetMaskOutput(
            mask_logits=mask_logits,
            pixel_features=pixel_features,
            mask_embeddings=mask_embeddings,
            class_weights=resolved_weights,
            detector=detector,
        )


@dataclass
class InstancePrediction:
    query_indices: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    boxes: torch.Tensor
    mask_logits: torch.Tensor


@dataclass
class BasePrediction:
    instances: InstancePrediction
    union_logits: torch.Tensor | None


def postprocess_set_masks(
    output: SetMaskOutput,
    original_sizes: list[tuple[int, int]],
    *,
    detection_threshold: float,
    max_detections: int | None = None,
) -> list[BasePrediction]:
    if len(original_sizes) != output.mask_logits.shape[0]:
        raise ValueError("original_sizes must contain one entry per image.")
    if not 0.0 <= detection_threshold <= 1.0:
        raise ValueError("Detection threshold must be between zero and one.")
    if max_detections is not None and max_detections <= 0:
        raise ValueError("max_detections must be positive when provided.")

    probabilities = output.detector.query_logits.sigmoid()
    query_scores, query_labels = probabilities.max(dim=-1)
    predictions: list[BasePrediction] = []
    for image_index, (height, width) in enumerate(original_sizes):
        accepted = torch.nonzero(
            query_scores[image_index] >= detection_threshold,
            as_tuple=False,
        ).flatten()
        if accepted.numel():
            order = torch.argsort(query_scores[image_index, accepted], descending=True)
            accepted = accepted[order]
            if max_detections is not None:
                accepted = accepted[:max_detections]

        scores = query_scores[image_index, accepted]
        labels = query_labels[image_index, accepted]
        corners = SetMaskHead.center_to_corners(
            output.detector.query_boxes[image_index, accepted]
        ).clamp(0.0, 1.0)
        scale = corners.new_tensor((width, height, width, height))
        boxes = corners * scale
        low_resolution_masks = output.mask_logits[image_index, accepted]
        if accepted.numel():
            mask_logits = F.interpolate(
                low_resolution_masks.unsqueeze(1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            union_logits: torch.Tensor | None = mask_logits.amax(dim=0)
        else:
            mask_logits = low_resolution_masks.new_empty((0, height, width))
            union_logits = None
        predictions.append(
            BasePrediction(
                instances=InstancePrediction(
                    query_indices=accepted,
                    scores=scores,
                    labels=labels,
                    boxes=boxes,
                    mask_logits=mask_logits,
                ),
                union_logits=union_logits,
            )
        )
    return predictions


__all__ = [
    "BasePrediction",
    "InstancePrediction",
    "PixelDecoder",
    "QueryTextFusion",
    "SetMaskHead",
    "SetMaskOutput",
    "TextConditioning",
    "postprocess_set_masks",
]
