import torch
from transformers import (
    RTDetrImageProcessor,
    RTDetrV2Config,
    RTDetrV2ForObjectDetection,
    ResNetConfig,
)

from flowsis.pretrained.rtdetrv2 import RTDetrV2Detector


def build_tiny_rtdetrv2() -> RTDetrV2Detector:
    backbone = ResNetConfig(
        depths=[1, 1, 1, 1],
        hidden_sizes=[16, 32, 64, 128],
        embedding_size=8,
        out_features=["stage2", "stage3", "stage4"],
    )
    config = RTDetrV2Config(
        backbone_config=backbone,
        encoder_in_channels=(32, 64, 128),
        encoder_hidden_dim=32,
        encoder_ffn_dim=64,
        encoder_attention_heads=4,
        d_model=32,
        decoder_in_channels=(32, 32, 32),
        decoder_ffn_dim=64,
        decoder_attention_heads=4,
        decoder_layers=2,
        num_queries=10,
        num_denoising=0,
        num_labels=3,
    )
    return RTDetrV2Detector(
        RTDetrImageProcessor(),
        RTDetrV2ForObjectDetection(config),
        source="tiny-rtdetrv2",
        image_size=64,
    )


def test_rtdetrv2_differentiable_feature_contract() -> None:
    detector = build_tiny_rtdetrv2()
    pixels = torch.zeros(1, 3, 64, 64, requires_grad=True)
    output = detector.forward_features(
        pixels, pixel_mask=torch.ones(1, 64, 64, dtype=torch.long)
    )
    assert output.detail_features.shape == (1, 16, 16, 16)
    assert [feature.shape for feature in output.feature_maps] == [
        (1, 32, 8, 8),
        (1, 32, 4, 4),
        (1, 32, 2, 2),
    ]
    assert output.query_embeddings.shape == (1, 10, 32)
    output.query_embeddings.mean().backward()
    assert pixels.grad is not None
