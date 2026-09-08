import pytest
import torch
from types import SimpleNamespace

from flowsis.cli.train.train_temporal import temporal_objective
from flowsis.temporal import (
    TemporalRefinementBranch,
    TemporalState,
    warp_backward,
)
from flowsis.temporal_checkpoint import (
    load_temporal_checkpoint,
    save_temporal_checkpoint,
)


def test_zero_flow_is_identity() -> None:
    values = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    flow = torch.zeros(1, 2, 3, 4)

    warped, validity = warp_backward(values, flow)

    torch.testing.assert_close(warped, values)
    assert validity.shape == (1, 1, 3, 4)
    assert validity.all()


def test_backward_flow_samples_source_coordinates_and_marks_bounds() -> None:
    values = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]])
    flow = torch.zeros(1, 2, 1, 4)
    flow[:, 0] = 1.0

    warped, validity = warp_backward(values, flow)

    torch.testing.assert_close(warped, torch.tensor([[[[1.0, 2.0, 3.0, 0.0]]]]))
    torch.testing.assert_close(
        validity,
        torch.tensor([[[[1.0, 1.0, 1.0, 0.0]]]]),
    )


def test_temporal_branch_returns_full_resolution_diagnostics() -> None:
    branch = TemporalRefinementBranch(channels=(8, 16))
    current_frame = torch.rand(2, 3, 16, 20)
    previous_frame = torch.rand(2, 3, 16, 20)
    base_logits = torch.randn(2, 1, 16, 20, requires_grad=True)
    previous_logits = torch.randn(2, 1, 16, 20, requires_grad=True)
    flow = torch.zeros(2, 2, 16, 20, requires_grad=True)

    output = branch(
        current_frame,
        previous_frame,
        base_logits,
        previous_logits,
        flow,
    )

    for tensor in (
        output.final_logits,
        output.propagated_logits,
        output.propagation_gate,
        output.residual_gate,
        output.logit_residual,
        output.warp_validity,
        output.photometric_residual,
    ):
        assert tensor.shape == (2, 1, 16, 20)

    output.final_logits.mean().backward()
    assert base_logits.grad is not None
    assert previous_logits.grad is not None
    assert flow.grad is not None
    assert branch.gate_head.weight.grad is not None
    assert branch.residual_head.weight.grad is not None


def test_invalid_warp_falls_back_to_base_at_initialization() -> None:
    branch = TemporalRefinementBranch(channels=(8,))
    current_frame = torch.rand(1, 3, 8, 8)
    previous_frame = torch.rand(1, 3, 8, 8)
    base_logits = torch.randn(1, 1, 8, 8)
    previous_logits = torch.randn(1, 1, 8, 8)
    flow = torch.full((1, 2, 8, 8), 100.0)

    output = branch(
        current_frame,
        previous_frame,
        base_logits,
        previous_logits,
        flow,
    )

    assert not output.warp_validity.any()
    torch.testing.assert_close(output.final_logits, base_logits)


def test_initial_refiner_is_exact_base_identity_with_features_and_hidden() -> None:
    branch = TemporalRefinementBranch(
        channels=(8, 16), feature_channels=6, context_channels=4, hidden_channels=5
    )
    base_logits = torch.randn(2, 1, 5, 7)
    output = branch(
        torch.rand(2, 3, 20, 28),
        torch.rand(2, 3, 20, 28),
        base_logits,
        torch.randn_like(base_logits),
        torch.randn(2, 2, 20, 28),
        current_features=torch.rand(2, 6, 5, 7),
        previous_features=torch.rand(2, 6, 5, 7),
        previous_hidden=torch.rand(2, 5, 5, 7),
    )

    torch.testing.assert_close(output.final_logits, base_logits, rtol=0, atol=0)
    assert output.hidden_state is not None
    assert output.hidden_state.shape == (2, 5, 5, 7)
    assert output.visibility_logits is not None
    assert output.visibility_logits.shape == (2, 1)


def test_temporal_state_detaches_recurrent_tensors() -> None:
    branch = TemporalRefinementBranch(channels=(8,))
    frame = torch.rand(1, 3, 8, 8, requires_grad=True)
    base_logits = torch.randn(1, 1, 8, 8, requires_grad=True)
    output = branch(
        frame,
        torch.rand_like(frame),
        base_logits,
        torch.randn_like(base_logits),
        torch.zeros(1, 2, 8, 8),
    )

    state = TemporalState.from_output(frame, output, identity=7)

    assert not state.frame.requires_grad
    assert not state.logits.requires_grad
    assert state.identity == 7


def test_temporal_branch_rejects_incompatible_shapes() -> None:
    branch = TemporalRefinementBranch(channels=(8,))

    with pytest.raises(ValueError, match="current_base_logits"):
        branch(
            torch.rand(1, 3, 8, 8),
            torch.rand(1, 3, 8, 8),
            torch.rand(1, 8, 8),
            torch.rand(1, 1, 8, 8),
            torch.zeros(1, 2, 8, 8),
        )


def test_temporal_checkpoint_round_trip(tmp_path) -> None:
    branch = TemporalRefinementBranch(
        channels=(8,), feature_channels=6, context_channels=4, hidden_channels=5
    )
    path = save_temporal_checkpoint(branch, tmp_path)

    loaded = load_temporal_checkpoint(path)

    assert loaded.channels == (8,)
    assert loaded.feature_channels == 6
    for expected, actual in zip(
        branch.state_dict().values(), loaded.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_recurrent_objective_backpropagates_through_time() -> None:
    batch_size, length, tracks, queries = 1, 3, 1, 2
    branch = TemporalRefinementBranch(
        channels=(8,), feature_channels=3, context_channels=4, hidden_channels=5
    )
    base_output = SimpleNamespace(
        mask_logits=torch.randn(batch_size, length, queries, 4, 4),
        pixel_features=torch.randn(batch_size, length, 3, 4, 4),
        class_weights=torch.softmax(
            torch.randn(batch_size, length, queries, 2), dim=-1
        ),
        detector=SimpleNamespace(
            query_logits=torch.tensor([[[[8.0, -8.0], [-8.0, 8.0]]] * length]),
            query_boxes=torch.tensor(
                [[[[0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]]] * length]
            ),
        ),
    )
    batch = {
        "labels": torch.zeros(batch_size, length, tracks, dtype=torch.long),
        "boxes": torch.tensor([[[[0.25, 0.25, 0.75, 0.75]]] * length]),
        "base_valid": torch.ones(batch_size, length, tracks, dtype=torch.bool),
        "mask_valid": torch.ones(batch_size, length, tracks, dtype=torch.bool),
        "visible": torch.ones(batch_size, length, tracks, dtype=torch.bool),
        "masks": torch.ones(batch_size, length, tracks, 16, 16),
        "annotation_confidence": torch.ones(batch_size, length, tracks),
        "preferred_track": torch.zeros(batch_size, length, tracks, dtype=torch.bool),
        "ignore": torch.zeros(batch_size, length, 16, 16, dtype=torch.bool),
        "frame_delta": torch.ones(batch_size, length, dtype=torch.long),
    }

    class ZeroFlow(torch.nn.Module):
        def forward(self, current, previous):
            return current.new_zeros((current.shape[0], 2, *current.shape[-2:]))

    result = temporal_objective(
        branch,
        ZeroFlow(),
        torch.rand(batch_size, length, 3, 16, 16),
        base_output,
        batch,
        visibility_weight=0.1,
        preferred_track_weight=1.0,
        use_annotation_confidence=True,
    )
    result["loss"].backward()

    assert torch.isfinite(result["loss"])
    assert branch.residual_head.weight.grad is not None
    assert branch.hidden_candidate.weight.grad is not None
