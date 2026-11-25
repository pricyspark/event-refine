"""Four-channel DETR detection with causal recurrent query refinement.

Developed with Codex assistance. No masks or text prompts are required.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from flowsis.matching import center_to_corners
from flowsis.pretrained.detector import BaseDetector, DetectorFeatureOutput

from .events import prepare_inputs


@dataclass
class TemporalState:
    hidden: torch.Tensor
    boxes: torch.Tensor
    logits: torch.Tensor

    def detach(self) -> "TemporalState":
        return TemporalState(
            self.hidden.detach(), self.boxes.detach(), self.logits.detach()
        )


@dataclass
class EventFlowOutput:
    logits: torch.Tensor
    boxes: torch.Tensor
    base: DetectorFeatureOutput
    state: TemporalState


def expand_event_stem(detector: BaseDetector) -> None:
    """Preserve RGB weights and add zero-initialized event weights.

    Support the native Transformers ResNet and HGNetV2 backbones used by the
    inherited RT-DETRv2 and D-FINE adapters. Update runtime checks and serialized
    config together so expanded checkpoints reload without mismatched weights.
    """
    model = detector._model
    config = model.config.backbone_config
    if config.model_type not in {"resnet", "hgnet_v2"}:
        raise ValueError("Event-Flow currently supports ResNet and HGNetV2 backbones.")
    backbone = model.model.backbone.model
    name, conv = next(
        (name, layer)
        for name, layer in backbone.named_modules()
        if isinstance(layer, nn.Conv2d)
    )
    if conv.in_channels == 4:
        return
    if conv.in_channels != 3 or conv.groups != 1:
        raise ValueError("Expected an ungrouped three-channel backbone stem.")
    expanded = nn.Conv2d(
        4,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, :3].copy_(conv.weight)
        if expanded.bias is not None:
            expanded.bias.copy_(conv.bias)
    parent_path, _, child = name.rpartition(".")
    parent = backbone.get_submodule(parent_path) if parent_path else backbone
    setattr(parent, child, expanded)
    config.num_channels = 4
    backbone.config.num_channels = 4
    backbone.embedder.num_channels = 4
    if config.model_type == "hgnet_v2":
        config.stem_channels = [4, *config.stem_channels[1:]]
        backbone.config.stem_channels = list(config.stem_channels)


class TemporalDetectionHead(nn.Module):
    """Content/spatial attention associates queries across frames, then a GRU
    carries history. Query indices are never assumed to be stable identities.
    """

    def __init__(self, query_dim: int, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if min(query_dim, num_classes, hidden_dim) <= 0:
            raise ValueError("Head dimensions must be positive.")
        self.hidden_dim = hidden_dim
        self.event_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.geometry_projection = nn.Linear(4 + num_classes, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.box_delta = nn.Linear(hidden_dim, 4)
        self.class_delta = nn.Linear(hidden_dim, num_classes)
        for layer in (self.box_delta, self.class_delta):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        base: DetectorFeatureOutput,
        events: torch.Tensor,
        previous: TemporalState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, TemporalState]:
        batch, queries, _ = base.query_embeddings.shape
        # Pool each query's spatial event context on a 4x4 grid. Work at stride
        # four; max pooling retains isolated occupied pixels before convolution.
        pooled = F.max_pool2d(events, 4, stride=4, ceil_mode=True)
        event_features = self.event_encoder(pooled)
        bins = torch.linspace(-0.375, 0.375, 4, device=events.device)
        y, x = torch.meshgrid(bins, bins, indexing="ij")
        offsets = torch.stack((x, y), dim=-1).reshape(1, 1, 16, 2)
        grid = (
            base.query_boxes[:, :, None, :2]
            + offsets * base.query_boxes[:, :, None, 2:]
        )
        sampled = F.grid_sample(event_features, grid * 2 - 1, align_corners=False)
        event_context = sampled.mean(dim=-1).transpose(1, 2)
        current = self.norm(
            self.query_projection(base.query_embeddings)
            + event_context
            + self.geometry_projection(
                torch.cat((base.query_boxes, base.query_logits.sigmoid()), dim=-1)
            )
        )
        history = torch.zeros_like(current)
        if previous is not None:
            if (
                previous.hidden.shape != (batch, queries, self.hidden_dim)
                or previous.boxes.shape != base.query_boxes.shape
                or previous.logits.shape != base.query_logits.shape
            ):
                raise ValueError(
                    "Temporal state shape changed; reset state at sequence boundaries."
                )
            similarity = (
                current @ previous.hidden.transpose(1, 2) / math.sqrt(self.hidden_dim)
            )
            distance = torch.cdist(
                base.query_boxes[..., :2], previous.boxes[..., :2], p=1
            )
            confidence = previous.logits.sigmoid().amax(dim=-1).clamp_min(1e-4)
            attention = (
                similarity - 4.0 * distance + confidence[:, None].log()
            ).softmax(dim=-1)
            history = attention @ previous.hidden
        hidden = self.cell(
            current.reshape(-1, self.hidden_dim), history.reshape(-1, self.hidden_dim)
        ).reshape(batch, queries, self.hidden_dim)
        boxes = (base.query_boxes + 0.1 * self.box_delta(hidden).tanh()).clamp(
            1e-6, 1 - 1e-6
        )
        logits = base.query_logits + self.class_delta(hidden)
        return boxes, logits, TemporalState(hidden, boxes, logits)


class EventFlow(nn.Module):
    """Stateless training interface: pass/return explicit state for each clip.

    ``forward`` accepts prepared [B,4,H,W] tensors; ``prepare`` handles aligned
    raw RGB [0,1] and event occupancy. Discard state on sequence changes/gaps.
    """

    def __init__(self, detector: BaseDetector, *, hidden_dim: int = 128) -> None:
        super().__init__()
        if detector.image_size < 64 or detector.image_size % 32:
            raise ValueError(
                "Event-Flow image size must be a multiple of 32 and at least 64."
            )
        expand_event_stem(detector)
        self.detector = detector
        config = detector._model.config
        self.temporal = TemporalDetectionHead(
            config.d_model, config.num_labels, hidden_dim
        )
        self.temporal.to(detector.device)
        processor = detector._processor
        normalize = getattr(processor, "do_normalize", False)
        self.register_buffer(
            "rgb_mean",
            torch.tensor(
                processor.image_mean if normalize else [0.0, 0.0, 0.0],
                device=detector.device,
            ),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor(
                processor.image_std if normalize else [1.0, 1.0, 1.0],
                device=detector.device,
            ),
        )
        self.rgb_scale = (
            255 * float(processor.rescale_factor)
            if getattr(processor, "do_rescale", True)
            else 255.0
        )

    def prepare(
        self,
        rgb: torch.Tensor,
        events: torch.Tensor,
        *,
        validate_values: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return prepare_inputs(
            rgb,
            events,
            image_size=self.detector.image_size,
            mean=self.rgb_mean,
            std=self.rgb_std,
            rgb_scale=self.rgb_scale,
            validate_values=validate_values,
        )

    def forward(
        self,
        pixels: torch.Tensor,
        *,
        pixel_mask: torch.Tensor | None = None,
        state: TemporalState | None = None,
    ) -> EventFlowOutput:
        if pixels.ndim != 4 or pixels.shape[1] != 4:
            raise ValueError("Event-Flow expects [B,4,H,W] RGB/event input.")
        base = self.detector.forward_features(pixels, pixel_mask=pixel_mask)
        boxes, logits, state = self.temporal(base, pixels[:, 3:4], state)
        return EventFlowOutput(logits, boxes, base, state)

    @staticmethod
    def detections(
        output: EventFlowOutput,
        *,
        image_size: int,
        original_size: tuple[int, int],
        threshold: float = 0.25,
    ) -> list[dict[str, torch.Tensor]]:
        """Return pixel xyxy boxes in the original registered RGB coordinates."""
        if not 0 <= threshold <= 1:
            raise ValueError("Detection threshold must be in [0,1].")
        height, width = original_size
        scale = image_size / max(height, width)
        resized_height = max(1, round(height * scale))
        resized_width = max(1, round(width * scale))
        corners = center_to_corners(output.boxes)
        corners = corners * corners.new_tensor(
            (image_size * width / resized_width, image_size * height / resized_height)
            * 2
        )
        corners[..., 0::2].clamp_(0, width)
        corners[..., 1::2].clamp_(0, height)
        scores, labels = output.logits.sigmoid().max(dim=-1)
        result = []
        for boxes, image_scores, image_labels in zip(
            corners, scores, labels, strict=True
        ):
            valid = (image_scores >= threshold) & (boxes[:, 2:] > boxes[:, :2]).all(
                dim=-1
            )
            order = image_scores[valid].argsort(descending=True)
            result.append(
                {
                    "boxes": boxes[valid][order],
                    "scores": image_scores[valid][order],
                    "labels": image_labels[valid][order],
                }
            )
        return result
