from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .association import box_area
from .types import SelectionResult, Track


@dataclass(frozen=True)
class SelectionWeights:
    normalized_area: float = 0.12
    mask_stability: float = 0.16
    class_certainty: float = 0.12
    class_stability: float = 0.10
    confidence: float = 0.16
    hit_rate: float = 0.10
    age: float = 0.08
    misses: float = 0.06
    continuity: float = 0.10


def normalized_entropy(probabilities: np.ndarray | None) -> float:
    if probabilities is None or probabilities.size <= 1:
        return 0.0
    values = np.asarray(probabilities, dtype=np.float64).clip(1e-12)
    values /= values.sum()
    return float(-(values * np.log(values)).sum() / math.log(values.size))


def score_track(
    track: Track,
    *,
    selected_track_id: int | None,
    weights: SelectionWeights,
) -> tuple[float, dict[str, float]]:
    if track.frame_size is None:
        area = 0.0
    else:
        height, width = track.frame_size
        area = min(box_area(track.box) / max(height * width, 1), 1.0)
    breakdown = {
        "normalized_area": area,
        "mask_stability": track.mask_stability,
        "class_certainty": 1.0 - normalized_entropy(track.class_distribution),
        "class_stability": track.class_stability,
        "confidence": track.score_ema,
        "hit_rate": track.hit_rate,
        "age": min(track.age / 30.0, 1.0),
        "misses": 1.0 - min(track.misses / 5.0, 1.0),
        "continuity": float(track.track_id == selected_track_id),
    }
    total = sum(getattr(weights, key) * value for key, value in breakdown.items())
    return total, breakdown


class TrackSelector:
    """Select a confirmed persistent track with switch hysteresis."""

    def __init__(
        self,
        *,
        weights: SelectionWeights | None = None,
        switch_margin: float = 0.08,
        switch_frames: int = 3,
    ) -> None:
        if switch_margin < 0 or switch_frames <= 0:
            raise ValueError("Switch margin must be nonnegative and frames positive.")
        self.weights = weights or SelectionWeights()
        self.switch_margin = switch_margin
        self.switch_frames = switch_frames
        self.selected_track_id: int | None = None
        self._challenger_id: int | None = None
        self._challenger_frames = 0

    def reset(self) -> None:
        self.selected_track_id = None
        self._challenger_id = None
        self._challenger_frames = 0

    def select(self, tracks: Sequence[Track]) -> SelectionResult | None:
        candidates = [track for track in tracks if track.is_confirmed]
        if not candidates:
            return None
        scored = {
            track.track_id: score_track(
                track,
                selected_track_id=self.selected_track_id,
                weights=self.weights,
            )
            for track in candidates
        }
        by_id = {track.track_id: track for track in candidates}
        best = max(candidates, key=lambda track: scored[track.track_id][0])
        current = by_id.get(self.selected_track_id)
        if current is None:
            self.selected_track_id = best.track_id
            self._challenger_id = None
            self._challenger_frames = 0
        elif best.track_id == current.track_id:
            self._challenger_id = None
            self._challenger_frames = 0
        else:
            best_score = scored[best.track_id][0]
            current_score = scored[current.track_id][0]
            if best_score >= current_score + self.switch_margin:
                if self._challenger_id == best.track_id:
                    self._challenger_frames += 1
                else:
                    self._challenger_id = best.track_id
                    self._challenger_frames = 1
                if self._challenger_frames >= self.switch_frames:
                    self.selected_track_id = best.track_id
                    self._challenger_id = None
                    self._challenger_frames = 0
            else:
                self._challenger_id = None
                self._challenger_frames = 0
        selected = by_id[self.selected_track_id]
        selection_score, breakdown = scored[selected.track_id]
        return SelectionResult(
            index=(
                selected.last_detection_index
                if selected.last_detection_index is not None
                else -1
            ),
            box=selected.box,
            score=selected.score,
            label=selected.label,
            selection_score=selection_score,
            track_length=selected.age,
            matched_frames=selected.hits,
            score_breakdown=breakdown,
            track_id=selected.track_id,
        )


def weights_from_mapping(values: Mapping[str, float]) -> SelectionWeights:
    unknown = set(values) - set(SelectionWeights.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown selector weights: {sorted(unknown)}")
    return SelectionWeights(**values)


def normalize_box(
    selection: SelectionResult, *, width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = selection.box
    return (
        max(0.0, min(x1 / width, 1.0)),
        max(0.0, min(y1 / height, 1.0)),
        max(0.0, min(x2 / width, 1.0)),
        max(0.0, min(y2 / height, 1.0)),
    )


def select_detection(
    detections_log: Sequence[Mapping[str, object]],
    previous_selection: SelectionResult | None = None,
) -> SelectionResult | None:
    """Compatibility wrapper around persistent tracking and the new policy."""
    from .tracker import SelectionTracker, TrackerConfig

    tracker = SelectionTracker(TrackerConfig(min_hits=1))
    tracks = []
    for detections in detections_log[-30:]:
        tracks = tracker.update(detections)
    selector = TrackSelector(switch_frames=1)
    if previous_selection is not None and previous_selection.track_id is not None:
        selector.selected_track_id = previous_selection.track_id
    return selector.select(tracks)


def select_first_detection(
    detections_log: Sequence[Mapping[str, object]],
) -> SelectionResult:
    result = select_detection(detections_log)
    if result is None:
        raise ValueError("No current detection can be selected.")
    return result


def select_recurrent_detection(
    detections_log: Sequence[Mapping[str, object]],
    previous_selection: SelectionResult | None = None,
) -> SelectionResult:
    result = select_detection(detections_log, previous_selection)
    if result is None:
        raise ValueError("No current detection can be selected.")
    return result


select_recurrant_detection = select_recurrent_detection


__all__ = [
    "SelectionWeights",
    "TrackSelector",
    "normalize_box",
    "normalized_entropy",
    "score_track",
    "select_detection",
    "select_first_detection",
    "select_recurrent_detection",
    "select_recurrant_detection",
    "weights_from_mapping",
]
