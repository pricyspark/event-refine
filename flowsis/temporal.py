from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TemporalOutput:
    """Outputs and diagnostics produced for one frame pair."""

    final_logits: torch.Tensor
    propagated_logits: torch.Tensor
    propagation_gate: torch.Tensor
    residual_gate: torch.Tensor
    logit_residual: torch.Tensor
    warp_validity: torch.Tensor
    photometric_residual: torch.Tensor
    hidden_state: torch.Tensor | None = None
    visibility_logits: torch.Tensor | None = None


def warp_backward(
    values: torch.Tensor,
    backward_flow: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``values`` using current-to-previous flow.

    ``backward_flow[:, :, y, x]`` contains the pixel displacement from a
    location in the output/current frame to its source in ``values``. The
    returned validity tensor is one where the source pixel center lies inside
    the input image.
    """
    if values.ndim != 4:
        raise ValueError(
            "values must have shape [B, C, H, W], "
            f"but received {tuple(values.shape)}."
        )
    if backward_flow.ndim != 4 or backward_flow.shape[1] != 2:
        raise ValueError(
            "backward_flow must have shape [B, 2, H, W], "
            f"but received {tuple(backward_flow.shape)}."
        )
    if values.shape[0] != backward_flow.shape[0]:
        raise ValueError("values and backward_flow must have the same batch size.")
    if values.shape[-2:] != backward_flow.shape[-2:]:
        raise ValueError("values and backward_flow must have the same spatial size.")
    if values.device != backward_flow.device:
        raise ValueError("values and backward_flow must be on the same device.")

    batch_size, _, height, width = values.shape
    coordinate_dtype = backward_flow.dtype
    y, x = torch.meshgrid(
        torch.arange(height, device=values.device, dtype=coordinate_dtype),
        torch.arange(width, device=values.device, dtype=coordinate_dtype),
        indexing="ij",
    )
    source_x = x.unsqueeze(0) + backward_flow[:, 0]
    source_y = y.unsqueeze(0) + backward_flow[:, 1]

    grid_x = (2.0 * (source_x + 0.5) / width) - 1.0
    grid_y = (2.0 * (source_y + 0.5) / height) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).to(values.dtype)

    warped = F.grid_sample(
        values,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    validity = (
        (source_x >= 0.0)
        & (source_x <= width - 1)
        & (source_y >= 0.0)
        & (source_y <= height - 1)
    ).unsqueeze(1)
    return warped, validity.to(values.dtype)


def resize_flow(flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize pixel-valued flow while preserving its coordinate units."""
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape [B,2,H,W].")
    old_height, old_width = flow.shape[-2:]
    height, width = size
    if (old_height, old_width) == size:
        return flow
    resized = F.interpolate(flow, size=size, mode="bilinear", align_corners=False)
    scale = flow.new_tensor((width / old_width, height / old_height)).view(1, 2, 1, 1)
    return resized * scale


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class TemporalRefinementBranch(nn.Module):
    """Flow-guided corrective refinement for binary segmentation logits.

    Frames must be floating-point RGB tensors with the same normalization.
    ``backward_flow`` is expressed in pixels and maps current-frame positions
    into the previous frame. Gradient policy is controlled by the caller, so
    segmentation gradients can reach a trainable flow estimator.
    """

    def __init__(
        self,
        *,
        channels: tuple[int, ...] = (32, 64, 96),
        feature_channels: int = 0,
        context_channels: int = 16,
        hidden_channels: int = 16,
        residual_limit: float = 5.0,
        propagation_gate_bias: float = -3.0,
        residual_gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if not channels or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain positive integers.")
        if residual_limit <= 0 or min(context_channels, hidden_channels) <= 0:
            raise ValueError("residual_limit must be positive.")
        if feature_channels < 0:
            raise ValueError("feature_channels cannot be negative.")

        self.residual_limit = float(residual_limit)
        self.channels = tuple(channels)
        self.feature_channels = int(feature_channels)
        self.context_channels = int(context_channels)
        self.hidden_channels = int(hidden_channels)
        self.feature_projection = (
            nn.Conv2d(feature_channels, context_channels, kernel_size=1)
            if feature_channels
            else None
        )

        encoder: list[nn.Module] = []
        # RGB reliability (14), recurrent state, observation metadata (4), and
        # optional current/warped/difference base-feature context.
        in_channels = 14 + hidden_channels + 4
        if feature_channels:
            in_channels += 3 * context_channels
        for index, out_channels in enumerate(channels):
            encoder.append(
                _ConvBlock(
                    in_channels,
                    out_channels,
                    stride=1 if index == 0 else 2,
                )
            )
            in_channels = out_channels
        self.encoder = nn.ModuleList(encoder)

        decoder: list[nn.Module] = []
        for deep_channels, skip_channels in zip(
            reversed(channels[1:]),
            reversed(channels[:-1]),
            strict=True,
        ):
            decoder.append(_ConvBlock(deep_channels + skip_channels, skip_channels))
        self.decoder = nn.ModuleList(decoder)

        self.gate_head = nn.Conv2d(channels[0], 2, kernel_size=1)
        self.residual_head = nn.Conv2d(channels[0], 1, kernel_size=1)
        self.hidden_candidate = nn.Conv2d(channels[0], hidden_channels, kernel_size=1)
        self.hidden_gate = nn.Conv2d(
            channels[0] + hidden_channels, hidden_channels, kernel_size=1
        )
        self.visibility_head = nn.Linear(channels[0], 1)
        self.propagation_strength = nn.Parameter(torch.zeros(()))
        # The residual convolution is exactly zero at initialization. Its
        # multiplier starts at one so the zero-valued path can receive a
        # gradient immediately rather than deadlocking two zero factors.
        self.residual_strength = nn.Parameter(torch.ones(()))
        self.missing_base_logit = nn.Parameter(torch.tensor(-4.0))
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias[0], propagation_gate_bias)
        nn.init.constant_(self.gate_head.bias[1], residual_gate_bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @staticmethod
    def _validate_inputs(
        current_frame: torch.Tensor,
        previous_frame: torch.Tensor,
        current_base_logits: torch.Tensor,
        previous_logits: torch.Tensor,
        backward_flow: torch.Tensor,
    ) -> None:
        if current_frame.ndim != 4 or current_frame.shape[1] != 3:
            raise ValueError("current_frame must have shape [B, 3, H, W].")
        if previous_frame.shape != current_frame.shape:
            raise ValueError(
                "previous_frame must have the same shape as current_frame."
            )
        expected_logit_shape = (
            current_frame.shape[0],
            1,
            *current_base_logits.shape[-2:],
        )
        if current_base_logits.shape != expected_logit_shape:
            raise ValueError(
                "current_base_logits must have shape "
                f"{expected_logit_shape}, but received "
                f"{tuple(current_base_logits.shape)}."
            )
        if previous_logits.shape != expected_logit_shape:
            raise ValueError(
                "previous_logits must have shape "
                f"{expected_logit_shape}, but received {tuple(previous_logits.shape)}."
            )
        if backward_flow.ndim != 4 or backward_flow.shape[:2] != (
            current_frame.shape[0],
            2,
        ):
            raise ValueError(
                "backward_flow must have shape [B,2,H,W], "
                f"but received {tuple(backward_flow.shape)}."
            )
        tensors = (
            current_frame,
            previous_frame,
            current_base_logits,
            previous_logits,
            backward_flow,
        )
        if any(not tensor.is_floating_point() for tensor in tensors):
            raise ValueError("All temporal inputs must be floating-point tensors.")
        if any(tensor.device != current_frame.device for tensor in tensors[1:]):
            raise ValueError("All temporal inputs must be on the same device.")

    @staticmethod
    def _reliability_features(
        current_frame: torch.Tensor,
        warped_previous_frame: torch.Tensor,
        current_base_logits: torch.Tensor,
        propagated_logits: torch.Tensor,
        backward_flow: torch.Tensor,
        validity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = current_frame.shape[-2:]
        flow_scale = backward_flow.new_tensor((width, height)).view(1, 2, 1, 1)
        normalized_flow = backward_flow / flow_scale
        rgb_residual = (current_frame - warped_previous_frame).abs()
        photometric_residual = rgb_residual.mean(dim=1, keepdim=True)
        logit_difference = propagated_logits - current_base_logits

        features = torch.cat(
            (
                current_frame,
                warped_previous_frame,
                photometric_residual,
                normalized_flow,
                validity,
                torch.tanh(current_base_logits / 5.0),
                torch.tanh(propagated_logits / 5.0),
                torch.tanh(logit_difference / 5.0),
                torch.tanh(logit_difference.abs() / 5.0),
            ),
            dim=1,
        )
        return features, photometric_residual

    def forward(
        self,
        current_frame: torch.Tensor,
        previous_frame: torch.Tensor,
        current_base_logits: torch.Tensor,
        previous_logits: torch.Tensor,
        backward_flow: torch.Tensor,
        *,
        current_features: torch.Tensor | None = None,
        previous_features: torch.Tensor | None = None,
        previous_hidden: torch.Tensor | None = None,
        observation_valid: torch.Tensor | None = None,
        observation_score: torch.Tensor | None = None,
        class_agreement: torch.Tensor | None = None,
        frame_delta: torch.Tensor | None = None,
    ) -> TemporalOutput:
        self._validate_inputs(
            current_frame,
            previous_frame,
            current_base_logits,
            previous_logits,
            backward_flow,
        )
        output_size = current_base_logits.shape[-2:]
        flow = resize_flow(backward_flow, output_size)
        current_frame = F.interpolate(
            current_frame, size=output_size, mode="bilinear", align_corners=False
        )
        previous_frame = F.interpolate(
            previous_frame, size=output_size, mode="bilinear", align_corners=False
        )
        propagated_logits, validity = warp_backward(previous_logits, flow)
        warped_previous_frame, _ = warp_backward(previous_frame, flow)

        batch_size, _, height, width = current_base_logits.shape

        def metadata_map(value: torch.Tensor | None, default: float) -> torch.Tensor:
            if value is None:
                return current_base_logits.new_full(
                    (batch_size, 1, height, width), default
                )
            value = value.to(
                device=current_base_logits.device, dtype=current_base_logits.dtype
            )
            if value.ndim == 1:
                value = value[:, None]
            if value.shape != (batch_size, 1):
                raise ValueError("Temporal metadata must have shape [B] or [B,1].")
            return value[:, :, None, None].expand(-1, -1, height, width)

        observation_map = metadata_map(observation_valid, 1.0)
        score_map = metadata_map(observation_score, 1.0)
        agreement_map = metadata_map(class_agreement, 1.0)
        delta_map = metadata_map(frame_delta, 1.0) / 3.0
        effective_base = torch.where(
            observation_map > 0.5,
            current_base_logits,
            self.missing_base_logit.to(current_base_logits.dtype),
        )
        # Rebuild reliability channels with the explicit missing-observation prior.
        inputs, photometric_residual = self._reliability_features(
            current_frame,
            warped_previous_frame,
            effective_base,
            propagated_logits,
            flow,
            validity,
        )
        if previous_hidden is None:
            previous_hidden = current_base_logits.new_zeros(
                (batch_size, self.hidden_channels, height, width)
            )
        if previous_hidden.shape != (
            batch_size,
            self.hidden_channels,
            height,
            width,
        ):
            raise ValueError("previous_hidden has an incompatible shape.")
        warped_hidden, _ = warp_backward(previous_hidden, flow)
        parts = [
            inputs,
            warped_hidden,
            observation_map,
            score_map,
            agreement_map,
            delta_map,
        ]
        if self.feature_projection is not None:
            if current_features is None or previous_features is None:
                raise ValueError(
                    "current_features and previous_features are required when "
                    "feature_channels is non-zero."
                )
            expected_features = (
                batch_size,
                self.feature_channels,
                current_features.shape[-2],
                current_features.shape[-1],
            )
            if current_features.shape != expected_features:
                raise ValueError("current_features has an incompatible shape.")
            if previous_features.shape != current_features.shape:
                raise ValueError(
                    "previous_features must have the same shape as current_features."
                )
            if current_features.device != current_base_logits.device:
                raise ValueError("Base features must be on the temporal branch device.")
            current_context = self.feature_projection(
                F.interpolate(
                    current_features,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            previous_context = self.feature_projection(
                F.interpolate(
                    previous_features,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            warped_context, _ = warp_backward(previous_context, flow)
            parts.extend(
                [
                    current_context,
                    warped_context,
                    (current_context - warped_context).abs(),
                ]
            )

        skips: list[torch.Tensor] = []
        features = torch.cat(parts, dim=1)
        for block in self.encoder:
            features = block(features)
            skips.append(features)
        for block, skip in zip(self.decoder, reversed(skips[:-1]), strict=True):
            features = F.interpolate(
                features,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            features = block(torch.cat((features, skip), dim=1))

        gate_logits = self.gate_head(features)
        propagation_gate = gate_logits[:, :1].sigmoid() * validity
        residual_gate = gate_logits[:, 1:].sigmoid()
        logit_residual = self.residual_limit * torch.tanh(self.residual_head(features))
        propagation_scale = torch.tanh(self.propagation_strength)
        residual_scale = self.residual_strength
        final_logits = (
            effective_base
            + propagation_scale
            * propagation_gate
            * (propagated_logits - effective_base)
            + residual_scale * residual_gate * logit_residual
        )
        candidate_hidden = torch.tanh(self.hidden_candidate(features))
        hidden_gate = torch.sigmoid(
            self.hidden_gate(torch.cat((features, warped_hidden), dim=1))
        )
        hidden_state = (
            hidden_gate * candidate_hidden + (1.0 - hidden_gate) * warped_hidden
        )
        visibility_logits = self.visibility_head(features.mean(dim=(-2, -1)))
        return TemporalOutput(
            final_logits=final_logits,
            propagated_logits=propagated_logits,
            propagation_gate=propagation_gate,
            residual_gate=residual_gate,
            logit_residual=logit_residual,
            warp_validity=validity,
            photometric_residual=photometric_residual,
            hidden_state=hidden_state,
            visibility_logits=visibility_logits,
        )


__all__ = [
    "TemporalOutput",
    "TemporalRefinementBranch",
    "warp_backward",
    "resize_flow",
]
