import numpy as np

from flowsis.selection import (
    SelectionTracker,
    SelectionWeights,
    TrackDetection,
    TrackSelector,
    TrackerConfig,
)


def detection(index: int, x: int, embedding: tuple[float, float]) -> TrackDetection:
    mask = np.zeros((20, 20), dtype=np.bool_)
    mask[5:12, x : x + 5] = True
    return TrackDetection(
        box=(x, 5, x + 5, 12),
        score=0.9,
        label=0,
        index=index,
        mask=mask,
        class_distribution=np.array([0.9, 0.1]),
        query_embedding=np.array(embedding),
        frame_size=(20, 20),
    )


def test_tracker_survives_miss_and_keeps_crossing_identities() -> None:
    tracker = SelectionTracker(
        TrackerConfig(
            min_hits=1,
            min_iou=0.0,
            max_motion=1.0,
            max_area_change=1.0,
            embedding_weight=2.0,
        )
    )
    first = tracker.update([detection(0, 2, (1, 0)), detection(1, 12, (0, 1))])
    first_ids = [track.track_id for track in first]
    tracker.update([])
    crossed = tracker.update([detection(0, 10, (1, 0)), detection(1, 4, (0, 1))])

    by_embedding = {
        tuple(track.query_embedding.tolist()): track.track_id for track in crossed
    }
    assert by_embedding[(1, 0)] == first_ids[0]
    assert by_embedding[(0, 1)] == first_ids[1]


def test_selector_requires_hysteresis_before_switch() -> None:
    tracker = SelectionTracker(TrackerConfig(min_hits=1))
    tracks = tracker.update([detection(0, 2, (1, 0)), detection(1, 12, (0, 1))])
    selector = TrackSelector(
        weights=SelectionWeights(
            normalized_area=0.0,
            mask_stability=0.0,
            class_certainty=0.0,
            class_stability=0.0,
            confidence=1.0,
            hit_rate=0.0,
            age=0.0,
            misses=0.0,
            continuity=0.0,
        ),
        switch_margin=0.0,
        switch_frames=2,
    )
    selected = selector.select(tracks)
    assert selected is not None
    original = selected.track_id

    stronger = [
        track
        for track in tracker.update(
            [
                TrackDetection(**{**detection(0, 2, (1, 0)).__dict__, "score": 0.1}),
                detection(1, 12, (0, 1)),
            ]
        )
    ]
    assert selector.select(stronger).track_id == original
    assert selector.select(stronger).track_id != original
