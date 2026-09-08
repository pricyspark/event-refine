from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature

from flowsis.base import FlowSISBase
from flowsis.base_checkpoint import load_base_checkpoint, save_base_checkpoint
from flowsis.pretrained import DetectorFeatureOutput
from flowsis.set_mask import SetMaskHead


class FakeDetector(nn.Module):
    architecture = "dfine"
    source = "fake"
    image_size = 16
    label_names = {0: "tool", 1: "other"}
    model_config = {"model_type": "d_fine", "num_queries": 2}

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def preprocess(self, images: Any, **_: Any) -> BatchFeature:
        batch_size = len(images) if isinstance(images, list) else 1
        return BatchFeature(
            {
                "pixel_values": self.anchor.new_zeros((batch_size, 3, 16, 16)),
                "pixel_mask": torch.ones(batch_size, 16, 16, dtype=torch.long),
            }
        )

    def forward_features(
        self,
        pixel_values: torch.Tensor,
        *,
        pixel_mask: torch.Tensor | None = None,
        labels: list[dict[str, torch.Tensor]] | None = None,
    ) -> DetectorFeatureOutput:
        batch_size = pixel_values.shape[0]
        logits = self.anchor.new_tensor([[[4.0, -4.0], [-4.0, -4.0]]]).expand(
            batch_size, -1, -1
        )
        boxes = self.anchor.new_tensor(
            [[[0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.1, 0.1]]]
        ).expand(batch_size, -1, -1)
        return DetectorFeatureOutput(
            loss=None,
            loss_dict={},
            detail_features=self.anchor.new_ones((batch_size, 4, 4, 4)),
            feature_maps=(
                self.anchor.new_ones((batch_size, 8, 2, 2)),
                self.anchor.new_ones((batch_size, 8, 1, 1)),
                self.anchor.new_ones((batch_size, 8, 1, 1)),
            ),
            query_embeddings=self.anchor.new_ones((batch_size, 2, 6)),
            query_logits=logits,
            query_boxes=boxes,
        )

    def save_pretrained(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        (output / "fake_detector").write_text("fake\n")


def build_model() -> FlowSISBase:
    prompt_bank = torch.randn(2, 3, 5)
    config = {
        "detail_channels": 4,
        "feature_channels": (8, 8, 8),
        "query_dim": 6,
        "mask_dim": 8,
    }
    return FlowSISBase(
        FakeDetector(),
        SetMaskHead(prompt_embeddings=prompt_bank, **config),
        ("tool", "other"),
        head_config=config,
        detection_threshold=0.5,
        use_amp=False,
    )


def test_forward_preserves_all_queries() -> None:
    model = build_model()
    output = model(torch.zeros(2, 3, 16, 16))
    assert output.mask_logits.shape == (2, 2, 4, 4)
    assert output.detector.query_logits.shape[:2] == output.mask_logits.shape[:2]


def test_infer_returns_unique_instances_and_union() -> None:
    model = build_model()
    prediction = model.infer_frame(np.zeros((16, 16, 3), dtype=np.uint8))
    assert prediction.instances.query_indices.tolist() == [0]
    assert prediction.instances.mask_logits.shape == (1, 16, 16)
    assert prediction.union_logits is not None
    torch.testing.assert_close(
        prediction.union_logits, prediction.instances.mask_logits[0]
    )


def test_empty_inference_has_no_union() -> None:
    model = build_model()
    prediction = model.infer_frame(
        np.zeros((16, 16, 3), dtype=np.uint8), detection_threshold=0.999
    )
    assert prediction.instances.query_indices.numel() == 0
    assert prediction.union_logits is None


def test_checkpoint_embeds_prompts(tmp_path: Path) -> None:
    model = build_model()
    path = save_base_checkpoint(model, tmp_path)
    checkpoint, resolved = load_base_checkpoint(path)
    assert resolved == path
    assert checkpoint.label_names == ("tool", "other")
    assert checkpoint.metadata["num_queries"] == 2
    assert checkpoint.metadata["mask_stride"] == 4
    torch.testing.assert_close(
        checkpoint.prompt_embeddings, model.head.prompt_embeddings.cpu()
    )


@pytest.mark.parametrize(
    ("argument", "value"),
    [("detection_threshold", -0.1), ("mask_threshold", 1.1), ("max_detections", 0)],
)
def test_invalid_configuration_is_rejected(argument: str, value: float) -> None:
    detector = FakeDetector()
    head = SetMaskHead(4, (8, 8, 8), 6, torch.randn(2, 3, 5), mask_dim=8)
    with pytest.raises(ValueError):
        FlowSISBase(
            detector,
            head,
            ("tool", "other"),
            head_config={},
            **{argument: value},
        )
