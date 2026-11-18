"""Inference-policy behavior without GT inputs, with Codex assistance."""

from types import SimpleNamespace

import numpy as np
import torch

from flowsis.selection import SelectionTracker, TrackerConfig
from flowsis.set_mask import SetMaskHead
from flowsis.temporal_model import FlowSISTemporal


class Base:
    device = torch.device("cpu")
    mask_threshold = detection_threshold = 0.5
    max_detections = None
    head = SimpleNamespace(mask_dim=1, center_to_corners=SetMaskHead.center_to_corners)
    detector = SimpleNamespace(
        preprocess=lambda *a, **kw: {"pixel_values": torch.zeros(1, 3, 8, 8)}
    )
    confidence = 4.0

    def eval(self):
        return self

    def __call__(self, *args, **kwargs):
        return SimpleNamespace(
            mask_logits=torch.full((1, 1, 8, 8), -1.0),
            pixel_features=torch.zeros(1, 1, 8, 8),
            class_weights=torch.ones(1, 1, 1),
            detector=SimpleNamespace(
                query_logits=torch.tensor([[[self.confidence]]]),
                query_boxes=torch.tensor([[[0.5, 0.5, 0.5, 0.5]]]),
                query_embeddings=torch.ones(1, 1, 2),
            ),
        )


class Refiner(torch.nn.Module):
    feature_channels = hidden_channels = 1

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, current, previous, logits, *args, previous_hidden, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            final_logits=logits + 2, hidden_state=previous_hidden + 1
        )


class Flow(torch.nn.Module):
    def forward(self, a, b):
        return torch.zeros(1, 2, *a.shape[-2:])


def test_default_retains_confirmation_delay_and_original_refinement():
    refiner = Refiner()
    model = FlowSISTemporal(Base(), Flow(), refiner)
    first = model.infer_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    assert first.selected is None and first.selected_mask is None
    assert refiner.calls == 1
    second = model.infer_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    assert second.selected is not None and second.selected_mask.all()


def test_fallback_returns_raw_base_without_promoting_tentative_track():
    model = FlowSISTemporal(
        Base(),
        Flow(),
        Refiner(),
        base_fallback=True,
        tracker=SelectionTracker(TrackerConfig(min_hits=3)),
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for _ in range(2):
        result = model.infer_frame(frame)
        assert result.used_base_fallback
        assert not result.selected_mask.any()
        assert not result.tracks[0].is_confirmed
        assert model.selector.selected_track_id is None
    third = model.infer_frame(frame)
    assert not third.used_base_fallback and third.selected_mask.all()


def test_valid_history_policy_seeds_state_and_resets_on_new_sequence():
    refiner = Refiner()
    model = FlowSISTemporal(
        Base(), Flow(), refiner, base_fallback=True, skip_initial_refinement=True
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    first = model.infer_frame(frame)
    assert refiner.calls == 0
    state = model.states[first.selected.track_id]
    assert (state.logits == -1).all() and (state.hidden == 0).all()
    model.infer_frame(frame)
    assert refiner.calls == 1
    model.reset()
    again = model.infer_frame(frame)
    assert refiner.calls == 1 and again.used_base_fallback
    assert not again.selected_mask.any()


def test_empty_detections_do_not_fabricate_fallback():
    base = Base()
    base.confidence = -4
    model = FlowSISTemporal(
        base, Flow(), Refiner(), base_fallback=True, skip_initial_refinement=True
    )
    result = model.infer_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result.selected is None and not result.used_base_fallback
