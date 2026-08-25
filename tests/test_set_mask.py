from __future__ import annotations

import pytest
import torch

from flowsis.cli.train.train_base import configure_stage, teacher_ratio
from flowsis.matching import match_object_queries, segmentation_objective
from flowsis.pretrained import DetectorFeatureOutput
from flowsis.set_mask import QueryTextFusion, SetMaskHead, postprocess_set_masks


def detector_output(*, batch_size: int = 2, queries: int = 4) -> DetectorFeatureOutput:
    return DetectorFeatureOutput(
        loss=None,
        loss_dict={},
        detail_features=torch.randn(batch_size, 12, 16, 16, requires_grad=True),
        feature_maps=(
            torch.randn(batch_size, 16, 8, 8, requires_grad=True),
            torch.randn(batch_size, 16, 4, 4, requires_grad=True),
            torch.randn(batch_size, 16, 2, 2, requires_grad=True),
        ),
        query_embeddings=torch.randn(batch_size, queries, 10, requires_grad=True),
        query_logits=torch.randn(batch_size, queries, 3, requires_grad=True),
        query_boxes=torch.rand(batch_size, queries, 4, requires_grad=True),
    )


def build_head(**kwargs: object) -> SetMaskHead:
    return SetMaskHead(
        12,
        (16, 16, 16),
        10,
        torch.randn(3, 2, 7),
        mask_dim=8,
        **kwargs,
    )


def test_set_head_shapes_and_gradient_paths() -> None:
    detector = detector_output()
    head = build_head()
    output = head(detector)
    assert output.mask_logits.shape == (2, 4, 16, 16)
    output.mask_logits[:, :2].mean().backward()
    assert detector.detail_features.grad is not None
    assert detector.feature_maps[0].grad is not None
    assert detector.query_embeddings.grad is not None
    assert detector.query_boxes.grad is not None
    assert head.prompt_embeddings.grad is None


def test_text_delta_starts_as_query_only_and_none_bypasses_delta() -> None:
    prompts = torch.randn(3, 2, 7)
    fusion = QueryTextFusion(10, prompts, mask_dim=8, conditioning="frozen")
    queries = torch.randn(1, 4, 10)
    logits = torch.randn(1, 4, 3)
    frozen, _ = fusion(queries, logits)
    fusion.conditioning = "none"
    no_text, _ = fusion(queries, logits)
    torch.testing.assert_close(frozen, no_text)
    with torch.no_grad():
        fusion.text_delta.weight.fill_(0.1)
    fusion.conditioning = "frozen"
    frozen_after_training, _ = fusion(queries, logits)
    fusion.conditioning = "none"
    no_text_after_training, _ = fusion(queries, logits)
    assert not torch.allclose(frozen_after_training, no_text_after_training)


def test_soft_box_prior_is_stable_expanded_and_differentiable() -> None:
    head = build_head(box_expansion=1.2)
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]], requires_grad=True)
    gates = head.soft_box_gates(boxes, height=32, width=32)
    assert torch.isfinite(gates).all()
    assert gates[0, 0, 16, 16] > gates[0, 0, 0, 0]
    unexpanded = build_head(box_expansion=1.0).soft_box_gates(
        boxes.detach(), height=32, width=32
    )
    assert gates[0, 0, 16, 19] > unexpanded[0, 0, 16, 19]
    gates.sum().backward()
    assert boxes.grad is not None
    assert torch.isfinite(boxes.grad).all()


def test_hungarian_matching_is_one_to_one_for_same_class() -> None:
    logits = torch.tensor([[[8.0, -8.0], [7.0, -8.0], [-8.0, 8.0]]])
    boxes = torch.tensor(
        [[[0.25, 0.5, 0.2, 0.2], [0.75, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]]
    )
    matched = match_object_queries(
        logits,
        boxes,
        torch.tensor([0, 0]),
        torch.tensor([0, 0]),
        torch.tensor([[0.15, 0.4, 0.35, 0.6], [0.65, 0.4, 0.85, 0.6]]),
    )
    assert matched.tolist() == [0, 1]
    assert matched.unique().numel() == 2


def test_all_empty_mask_objective_is_differentiable() -> None:
    logits = torch.empty(0, 8, 8, requires_grad=True)
    result = segmentation_objective(
        logits,
        torch.empty_like(logits),
        torch.empty(0, dtype=torch.long),
        batch_size=2,
    )
    result["loss"].backward()
    assert logits.grad is not None


def test_postprocessing_uses_unique_queries_and_max_union() -> None:
    detector = detector_output(batch_size=1, queries=3)
    detector.query_logits = torch.tensor(
        [[[5.0, 4.0, -5.0], [3.0, -5.0, -5.0], [-5.0, -5.0, -5.0]]]
    )
    output = build_head(use_box_prior=False)(detector)
    predictions = postprocess_set_masks(
        output, [(32, 40)], detection_threshold=0.5, max_detections=2
    )
    instances = predictions[0].instances
    assert instances.query_indices.tolist() == [0, 1]
    assert predictions[0].union_logits is not None
    torch.testing.assert_close(
        predictions[0].union_logits, instances.mask_logits.amax(dim=0)
    )


@pytest.mark.parametrize(
    ("stage", "epoch", "epochs", "expected"),
    [
        ("head", 0, 5, 1.0),
        ("neck", 0, 3, 1.0),
        ("neck", 1, 3, 0.5),
        ("neck", 2, 3, 0.0),
        ("all", 0, 2, 0.0),
    ],
)
def test_text_curriculum_endpoints(
    stage: str, epoch: int, epochs: int, expected: float
) -> None:
    assert teacher_ratio(stage, epoch, epochs) == expected


class _PhaseDetector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.neck = torch.nn.Linear(2, 2)


class _PhaseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.detector = _PhaseDetector()
        self.head = torch.nn.Linear(2, 2)


def test_training_phase_gradient_boundaries() -> None:
    model = _PhaseModel()
    backbone = list(model.detector.backbone.parameters())
    configure_stage(model, "head", backbone)
    assert not any(parameter.requires_grad for parameter in model.detector.parameters())
    assert not model.detector.training
    assert all(parameter.requires_grad for parameter in model.head.parameters())

    configure_stage(model, "neck", backbone)
    assert not any(parameter.requires_grad for parameter in backbone)
    assert all(
        parameter.requires_grad for parameter in model.detector.neck.parameters()
    )

    configure_stage(model, "all", backbone)
    assert all(parameter.requires_grad for parameter in model.detector.parameters())


def test_two_object_mask_head_overfits_tiny_example() -> None:
    torch.manual_seed(4)
    detector = detector_output(batch_size=1, queries=2)
    head = build_head(text_conditioning="none", use_box_prior=False)
    targets = torch.zeros(2, 16, 16)
    targets[0, 3:13, 1:7] = 1.0
    targets[1, 3:13, 9:15] = 1.0
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=0.0)

    losses = []
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        logits = head(detector).mask_logits[0]
        result = segmentation_objective(
            logits,
            targets,
            torch.tensor([0, 0]),
            batch_size=1,
        )
        result["loss"].backward()
        optimizer.step()
        losses.append(float(result["loss"].detach()))

    assert losses[-1] < 0.5 * losses[0]
