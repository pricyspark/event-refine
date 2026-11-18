"""Pixel-range/aliasing regressions, prepared with Codex assistance."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from flowsis.cli.train.train_temporal import _base_sequence
from flowsis.pretrained.image_processing import preprocess_detr_images


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_detector_preprocessing_does_not_mutate_float_input(batched, normalize):
    processor = SimpleNamespace(
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=normalize,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    original = image.clone()
    pixels, _, _ = preprocess_detr_images(
        processor,
        image if batched else [image[0]],
        image_size=8,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(image, original, rtol=0, atol=0)
    expected = original / 255
    if normalize:
        expected = (expected - 0.5) / 0.5
    torch.testing.assert_close(pixels, expected)


def test_temporal_detector_inputs_match_runtime_and_preserve_flow_frames():
    processor = SimpleNamespace(
        do_rescale=True, rescale_factor=1 / 255, do_normalize=False
    )
    raw = (
        torch.arange(2 * 3 * 3 * 8 * 8)
        .remainder(256)
        .to(torch.uint8)
        .reshape(2, 3, 3, 8, 8)
    )
    frames = raw.float() / 255
    original = frames.clone()
    captured = []

    class Detector:
        def preprocess(self, images, **kwargs):
            pixels, mask, _ = preprocess_detr_images(
                processor, images, image_size=8, device=torch.device("cpu")
            )
            return {"pixel_values": pixels, "pixel_mask": mask}

        def forward_features(self, pixels, **kwargs):
            captured.append(pixels.clone())
            return SimpleNamespace(
                query_logits=torch.ones(len(pixels), 1, 1),
                query_boxes=torch.zeros(len(pixels), 1, 4),
            )

    def head(features):
        count = features.query_logits.shape[0]
        return SimpleNamespace(
            mask_logits=torch.zeros(count, 1, 8, 8),
            pixel_features=torch.zeros(count, 2, 8, 8),
            class_weights=torch.ones(count, 1, 1),
            detector=features,
        )

    base = SimpleNamespace(
        detector=Detector(),
        head=head,
        device=torch.device("cpu"),
        detector_autocast_context=nullcontext,
    )
    output = _base_sequence(base, frames, train_head=False)
    expected, _, _ = preprocess_detr_images(
        processor, raw.flatten(0, 1), image_size=8, device=torch.device("cpu")
    )
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(frames, original, rtol=0, atol=0)
    assert output.mask_logits.shape == (2, 3, 1, 8, 8)
