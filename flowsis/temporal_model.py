from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from flowsis.base import FlowSISBase
from flowsis.pretrained.flow import PTLFlowEstimator
from flowsis.selection import (
    SelectionResult,
    SelectionTracker,
    Track,
    TrackDetection,
    TrackSelector,
)
from flowsis.temporal import TemporalRefinementBranch


@dataclass(frozen=True)
class TemporalFramePrediction:
    logits_by_track: dict[int, torch.Tensor]
    masks_by_track: dict[int, torch.Tensor]
    selected: SelectionResult | None
    tracks: tuple[Track, ...]

    @property
    def selected_mask(self) -> torch.Tensor | None:
        if self.selected is None or self.selected.track_id is None:
            return None
        return self.masks_by_track.get(self.selected.track_id)


@dataclass
class _RecurrentTrackState:
    logits: torch.Tensor
    hidden: torch.Tensor


class FlowSISTemporal:
    """Stateful inference pipeline with segmentation-independent selection."""

    def __init__(
        self,
        base: FlowSISBase,
        flow: PTLFlowEstimator,
        temporal: TemporalRefinementBranch,
        *,
        tracker: SelectionTracker | None = None,
        selector: TrackSelector | None = None,
    ) -> None:
        if temporal.feature_channels != base.head.mask_dim:
            raise ValueError(
                "Temporal feature channels must equal the base mask feature dimension."
            )
        self.base = base.eval()
        self.flow = flow.eval()
        self.temporal = temporal.eval()
        self.tracker = tracker or SelectionTracker()
        self.selector = selector or TrackSelector()
        self.previous_frame: torch.Tensor | None = None
        self.previous_features: torch.Tensor | None = None
        self.states: dict[int, _RecurrentTrackState] = {}

    @property
    def device(self) -> torch.device:
        return self.base.device

    def reset(self) -> None:
        self.previous_frame = None
        self.previous_features = None
        self.states.clear()
        self.tracker.reset()
        self.selector.reset()

    @torch.inference_mode()
    def infer_frame(self, bgr_frame: NDArray[np.uint8]) -> TemporalFramePrediction:
        if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
            raise ValueError(f"Expected an HWC BGR frame, got {bgr_frame.shape}.")
        rgb = np.ascontiguousarray(bgr_frame[..., ::-1])
        inputs = self.base.detector.preprocess([rgb], device_preprocess=True)
        pixels = inputs["pixel_values"].to(self.device)
        pixel_mask = inputs.get("pixel_mask")
        pixel_mask = (
            pixel_mask.to(self.device) if isinstance(pixel_mask, torch.Tensor) else None
        )
        output = self.base(pixels, pixel_mask=pixel_mask)
        frame = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)[None]
            .to(self.device, dtype=torch.float32)
            / 255.0
        )
        frame = F.interpolate(
            frame, size=pixels.shape[-2:], mode="bilinear", align_corners=False
        )
        height, width = output.mask_logits.shape[-2:]
        class_scores = output.detector.query_logits.sigmoid()
        scores, labels = class_scores.max(dim=-1)
        accepted = torch.nonzero(
            scores[0] >= self.base.detection_threshold, as_tuple=False
        ).flatten()
        if self.base.max_detections is not None and accepted.numel():
            order = torch.argsort(scores[0, accepted], descending=True)
            accepted = accepted[order[: self.base.max_detections]]
        corners = self.base.head.center_to_corners(output.detector.query_boxes[0])
        scale = corners.new_tensor((width, height, width, height))
        boxes = corners.clamp(0.0, 1.0) * scale
        detections = [
            TrackDetection(
                box=tuple(float(value) for value in boxes[query].tolist()),
                score=float(scores[0, query]),
                label=int(labels[0, query]),
                index=int(query),
                mask=(
                    output.mask_logits[0, query].sigmoid() >= self.base.mask_threshold
                )
                .cpu()
                .numpy(),
                class_distribution=output.class_weights[0, query].float().cpu().numpy(),
                query_embedding=output.detector.query_embeddings[0, query]
                .float()
                .cpu()
                .numpy(),
                frame_size=(height, width),
            )
            for query in accepted
        ]
        backward_flow = (
            None
            if self.previous_frame is None
            else self.flow(frame, self.previous_frame)
        )
        tracks = self.tracker.update(detections, backward_flow=backward_flow)
        active_ids = {track.track_id for track in tracks}
        self.states = {
            track_id: state
            for track_id, state in self.states.items()
            if track_id in active_ids
        }
        logits_by_track: dict[int, torch.Tensor] = {}
        masks_by_track: dict[int, torch.Tensor] = {}
        zero_flow = frame.new_zeros((1, 2, *frame.shape[-2:]))
        for track in tracks:
            observed = track.misses == 0 and track.last_detection_index is not None
            query = track.last_detection_index if observed else 0
            current_logits = (
                output.mask_logits[:, query : query + 1]
                if observed
                else output.mask_logits.new_zeros((1, 1, height, width))
            )
            previous = self.states.get(track.track_id)
            if (
                previous is None
                or self.previous_frame is None
                or self.previous_features is None
            ):
                previous_logits = current_logits
                previous_hidden = current_logits.new_zeros(
                    (1, self.temporal.hidden_channels, height, width)
                )
                previous_frame = frame
                previous_features = output.pixel_features
                track_flow = zero_flow
            else:
                previous_logits = previous.logits
                previous_hidden = previous.hidden
                previous_frame = self.previous_frame
                previous_features = self.previous_features
                assert backward_flow is not None
                track_flow = backward_flow
            refined = self.temporal(
                frame,
                previous_frame,
                current_logits,
                previous_logits,
                track_flow,
                current_features=output.pixel_features,
                previous_features=previous_features,
                previous_hidden=previous_hidden,
                observation_valid=current_logits.new_tensor([float(observed)]),
                observation_score=current_logits.new_tensor([track.score]),
                class_agreement=current_logits.new_tensor([track.class_stability]),
            )
            assert refined.hidden_state is not None
            self.states[track.track_id] = _RecurrentTrackState(
                refined.final_logits.detach(), refined.hidden_state.detach()
            )
            logits = F.interpolate(
                refined.final_logits,
                size=bgr_frame.shape[:2],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            logits_by_track[track.track_id] = logits
            masks_by_track[track.track_id] = (
                logits.sigmoid() >= self.base.mask_threshold
            )
        self.previous_frame = frame
        self.previous_features = output.pixel_features
        selected = self.selector.select(tracks)
        return TemporalFramePrediction(
            logits_by_track, masks_by_track, selected, tuple(tracks)
        )


__all__ = ["FlowSISTemporal", "TemporalFramePrediction"]
