from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class SelectionResult:
    index: int
    box: Box
    score: float
    label: int
    selection_score: float
    track_length: int
    matched_frames: int
    score_breakdown: dict[str, float]
    track_id: int | None = None


@dataclass(frozen=True)
class TrackDetection:
    box: Box
    score: float
    label: int
    index: int | None = None
    mask: np.ndarray | None = None
    class_distribution: np.ndarray | None = None
    query_embedding: np.ndarray | None = None
    frame_size: tuple[int, int] | None = None


@dataclass
class TrackState:
    track_id: int
    box: Box
    score: float
    label: int
    velocity: tuple[float, float] = (0.0, 0.0)
    age: int = 1
    hits: int = 1
    misses: int = 0
    stationary_frames: int = 0
    score_ema: float = 0.0
    last_detection_index: int | None = None
    history: list[TrackDetection] = field(default_factory=list)
    mask: np.ndarray | None = None
    warped_mask: np.ndarray | None = None
    class_distribution: np.ndarray | None = None
    query_embedding: np.ndarray | None = None
    mask_stability_ema: float = 0.0
    class_stability_ema: float = 0.0
    frame_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class Track:
    track_id: int
    box: Box
    score: float
    label: int
    age: int
    hits: int
    misses: int
    stationary_frames: int
    is_confirmed: bool
    last_detection_index: int | None
    score_ema: float
    history: tuple[TrackDetection, ...]
    mask: np.ndarray | None = None
    class_distribution: np.ndarray | None = None
    query_embedding: np.ndarray | None = None
    mask_stability: float = 0.0
    class_stability: float = 0.0
    hit_rate: float = 0.0
    frame_size: tuple[int, int] | None = None
