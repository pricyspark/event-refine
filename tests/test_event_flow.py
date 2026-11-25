"""Behavioral coverage for Event-Flow, developed with Codex assistance."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from transformers import (
    DFineConfig,
    DFineForObjectDetection,
    HGNetV2Config,
    ResNetConfig,
    RTDetrImageProcessor,
    RTDetrV2Config,
    RTDetrV2ForObjectDetection,
)

from event_flow.checkpoint import load_checkpoint, save_checkpoint
from event_flow.cli import predict, train
from event_flow.data import ClipDataset, import_fred, load_manifest
from event_flow.events import accumulate_events, load_event_map, prepare_inputs
from event_flow.loss import detection_loss, normalized_targets
from event_flow.metrics import DetectionRecord, detection_metrics
from event_flow.model import EventFlow, TemporalState
from flowsis.pretrained.dfine import DFineDetector
from flowsis.pretrained.rtdetrv2 import RTDetrV2Detector


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_detector(architecture="dfine", backbone="resnet"):
    if backbone == "resnet":
        backbone_config = ResNetConfig(
            depths=[1, 1, 1, 1],
            hidden_sizes=[16, 32, 64, 128],
            embedding_size=8,
            out_features=["stage2", "stage3", "stage4"],
        )
    else:
        backbone_config = HGNetV2Config(
            stem_channels=[3, 8, 16],
            hidden_sizes=[16, 32, 64, 128],
            stage_in_channels=[16, 16, 32, 64],
            stage_mid_channels=[8, 8, 16, 32],
            stage_out_channels=[16, 32, 64, 128],
            stage_num_blocks=[1, 1, 1, 1],
            stage_numb_of_layers=[1, 1, 1, 1],
            stage_light_block=[False] * 4,
            stage_kernel_size=[3] * 4,
            out_features=["stage2", "stage3", "stage4"],
        )
    config_type = DFineConfig if architecture == "dfine" else RTDetrV2Config
    config = config_type(
        backbone_config=backbone_config,
        encoder_in_channels=(32, 64, 128),
        encoder_hidden_dim=32,
        encoder_ffn_dim=64,
        encoder_attention_heads=4,
        d_model=32,
        decoder_in_channels=(32, 32, 32),
        decoder_ffn_dim=64,
        decoder_attention_heads=4,
        decoder_layers=2,
        num_queries=5,
        num_denoising=0,
        num_labels=1,
        id2label={0: "drone"},
        label2id={"drone": 0},
    )
    if architecture == "dfine":
        config.max_num_bins = 8
        config.lqe_hidden_dim = 16
        return DFineDetector(
            RTDetrImageProcessor(),
            DFineForObjectDetection(config),
            source="tiny",
            image_size=64,
        )
    return RTDetrV2Detector(
        RTDetrImageProcessor(),
        RTDetrV2ForObjectDetection(config),
        source="tiny",
        image_size=64,
    )


@pytest.mark.parametrize(
    "architecture,backbone",
    [("dfine", "resnet"), ("dfine", "hgnet_v2"), ("rtdetrv2", "resnet")],
)
def test_rgb_initialization_and_expanded_checkpoint_roundtrip(
    tmp_path, architecture, backbone
):
    detector = tiny_detector(architecture, backbone).eval()
    rgb = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        baseline = detector.forward_features(rgb)
    model = EventFlow(detector, hidden_dim=16).eval()
    with torch.no_grad():
        output = model(torch.cat((rgb, torch.ones(1, 1, 64, 64)), 1))
    torch.testing.assert_close(output.logits, baseline.query_logits)
    torch.testing.assert_close(output.boxes, baseline.query_boxes)
    assert detector._model.config.backbone_config.num_channels == 4
    save_checkpoint(model, tmp_path / "model", classes=("drone",))
    restored, classes = load_checkpoint(tmp_path / "model")
    assert classes == ("drone",)
    with torch.no_grad():
        actual = restored(torch.cat((rgb, torch.ones(1, 1, 64, 64)), 1))
    torch.testing.assert_close(output.logits, actual.logits)
    torch.testing.assert_close(output.boxes, actual.boxes)


def test_event_stem_and_box_class_heads_receive_training_gradients():
    model = EventFlow(tiny_detector(), hidden_dim=16).train()
    pixels = torch.rand(1, 4, 64, 64)
    pixels[:, 3] = (pixels[:, 3] > 0.8).float()
    output = model(pixels)
    target_boxes = torch.tensor([[0.25, 0.25, 0.55, 0.55]])
    loss = detection_loss(output.logits, output.boxes, target_boxes, torch.tensor([0]))[
        "loss"
    ]
    loss.backward()
    stem = next(
        layer
        for layer in model.detector._model.model.backbone.model.modules()
        if isinstance(layer, torch.nn.Conv2d)
    )
    assert stem.weight.grad[:, 3].abs().sum() > 0
    assert model.temporal.box_delta.weight.grad.abs().sum() > 0
    assert model.temporal.class_delta.weight.grad.abs().sum() > 0
    assert torch.isfinite(loss)


def test_temporal_history_changes_predictions_and_ignores_query_order():
    model = EventFlow(tiny_detector(), hidden_dim=16).eval()
    torch.nn.init.normal_(model.temporal.class_delta.weight, std=0.1)
    torch.nn.init.normal_(model.temporal.box_delta.weight, std=0.1)
    pixels = torch.rand(1, 4, 64, 64)
    with torch.no_grad():
        first = model(pixels)
        second = model(pixels, state=first.state)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        reordered = TemporalState(
            first.state.hidden[:, permutation],
            first.state.boxes[:, permutation],
            first.state.logits[:, permutation],
        )
        permuted = model(pixels, state=reordered)
        reset = model(pixels)
    assert not torch.allclose(first.logits, second.logits)
    assert not torch.allclose(first.boxes, second.boxes)
    torch.testing.assert_close(second.logits, permuted.logits)
    torch.testing.assert_close(second.boxes, permuted.boxes)
    torch.testing.assert_close(first.logits, reset.logits)
    assert not first.state.detach().hidden.requires_grad


def test_gradient_reaches_previous_frame_through_recurrent_memory():
    model = EventFlow(tiny_detector(), hidden_dim=16).eval()
    torch.nn.init.normal_(model.temporal.class_delta.weight, std=0.1)
    first_pixels = torch.rand(1, 4, 64, 64, requires_grad=True)
    first = model(first_pixels)
    second = model(torch.rand(1, 4, 64, 64), state=first.state)
    second.logits.sum().backward()
    assert first_pixels.grad is not None
    assert first_pixels.grad.abs().sum() > 0


def test_event_pooling_normalization_and_input_ownership():
    rgb = torch.full((1, 3, 4, 8), 0.5)
    events = torch.zeros(1, 1, 4, 8)
    events[0, 0, 1, 1] = 1
    original_rgb, original_events = rgb.clone(), events.clone()
    pixels, mask = prepare_inputs(
        rgb,
        events,
        image_size=4,
        mean=torch.tensor([0.5] * 3),
        std=torch.tensor([0.25] * 3),
    )
    assert pixels.shape == (1, 4, 4, 4)
    assert pixels[:, :3].count_nonzero() == 0
    assert pixels[:, 3].sum() == 1
    assert mask.sum() == 8
    torch.testing.assert_close(rgb, original_rgb)
    torch.testing.assert_close(events, original_events)
    with pytest.raises(ValueError, match="binary"):
        prepare_inputs(
            rgb, events + 0.5, image_size=4, mean=torch.zeros(3), std=torch.ones(3)
        )


def test_raw_event_window_is_causal_and_binary():
    result = accumulate_events(
        np.array([0, 1, 1, 2, 2]),
        np.array([0, 1, 1, 2, 2]),
        np.array([0.0, 0.5, 0.75, 1.0, 1.1]),
        start=0.0,
        end=1.0,
        height=3,
        width=3,
    )
    assert result.dtype == np.bool_
    assert result.sum() == 2
    assert not result[0, 0]
    assert result[1, 1] and result[2, 2]


def test_rendered_events_require_explicit_background(tmp_path):
    array = np.full((3, 3, 3), 128, dtype=np.uint8)
    array[1, 1] = [255, 0, 0]
    path = tmp_path / "events.png"
    Image.fromarray(array).save(path)
    with pytest.raises(ValueError, match="explicit"):
        load_event_map(path, encoding="rendered")
    occupancy = load_event_map(path, encoding="rendered", background=(128, 128, 128))
    assert occupancy.sum() == 1
    assert occupancy[0, 1, 1] == 1


def write_manifest(tmp_path: Path, *, missing=False, gap=False):
    frames = []
    for index in range(4):
        Image.fromarray(np.full((48, 64, 3), 127, dtype=np.uint8)).save(
            tmp_path / f"rgb{index}.png"
        )
        event_map = np.zeros((48, 64), dtype=np.uint8)
        event_map[10:20, 10] = 1
        np.save(tmp_path / f"event{index}.npy", event_map)
        frames.append(
            {
                "rgb": f"rgb{index}.png",
                "events": f"event{index}.npy",
                "timestamp": index * 0.03 + (1 if gap and index >= 2 else 0),
                "annotated": not (missing and index == 1),
                "boxes": [[8, 8, 24, 24]],
                "labels": [0],
            }
        )
    manifest = {
        "format_version": 1,
        "classes": ["drone"],
        "sequences": [{"id": "sequence", "split": "train", "frames": frames}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.mark.parametrize("missing,gap", [(True, False), (False, True)])
def test_clips_do_not_cross_missing_supervision_or_timestamp_gaps(
    tmp_path, missing, gap
):
    path = write_manifest(tmp_path, missing=missing, gap=gap)
    with pytest.raises(ValueError, match="No contiguous"):
        ClipDataset(path, split="train", clip_length=3)
    classes, sequences = load_manifest(path)
    assert classes == ("drone",)
    assert len(sequences[0].frames) == 4


def test_unlabeled_frames_are_not_implicitly_background(tmp_path):
    path = write_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    del manifest["sequences"][0]["frames"][0]["boxes"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="explicitly supply"):
        load_manifest(path)


def test_fred_import_preserves_missing_annotation_frames(tmp_path):
    folder = tmp_path / "FRED" / "train" / "0"
    (folder / "RGB").mkdir(parents=True)
    (folder / "Event" / "Frames").mkdir(parents=True)
    for index in range(3):
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(
            folder / "RGB" / f"{index}.jpg"
        )
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(
            folder / "Event" / "Frames" / f"{index}.png"
        )
    (folder / "coordinates.txt").write_text(
        "0.033333: 1, 1, 10, 10, 0, drone\n0.099999: 2, 2, 11, 11, 0, drone\n"
    )
    path = tmp_path / "fred.json"
    import_fred(tmp_path / "FRED", path, event_background=(0, 0, 0))
    _, sequences = load_manifest(path)
    assert [frame.annotated for frame in sequences[0].frames] == [True, False, True]


def test_empty_frame_loss_penalizes_false_positives():
    logits = torch.tensor([[[2.0], [1.0]]], requires_grad=True)
    boxes = torch.rand(1, 2, 4, requires_grad=True)
    result = detection_loss(
        logits, boxes, torch.empty(0, 4), torch.empty(0, dtype=torch.long)
    )
    result["loss"].backward()
    assert result["loss"] > 0
    assert bool((logits.grad > 0).all())
    assert result["bbox"] == 0


def test_box_geometry_roundtrip():
    boxes = torch.tensor([[8.0, 4.0, 40.0, 20.0]])
    normalized = normalized_targets(boxes, original_size=(24, 64), image_size=64)
    torch.testing.assert_close(normalized, boxes / 64)


def test_box_ap_counts_duplicate_false_positives_and_empty_frames():
    targets = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    perfect = DetectionRecord(
        targets, torch.tensor([0.9]), torch.tensor([0]), targets, torch.tensor([0])
    )
    assert detection_metrics([perfect], ("drone",))["AP50_95"] == pytest.approx(1)
    empty = DetectionRecord(
        targets,
        torch.tensor([0.99]),
        torch.tensor([0]),
        torch.empty(0, 4),
        torch.empty(0, dtype=torch.long),
    )
    assert detection_metrics([empty, perfect], ("drone",))["AP50"] == pytest.approx(0.5)


def test_training_and_evaluation_end_to_end_without_network(tmp_path):
    path = write_manifest(tmp_path)
    detector_dir = tmp_path / "detector"
    tiny_detector().save_pretrained(detector_dir)
    output = tmp_path / "trained"
    train(
        argparse.Namespace(
            manifest=path,
            split="train",
            validation_split=None,
            clip_length=2,
            max_gap=0.05,
            epochs=1,
            max_steps=2,
            learning_rate=1e-4,
            temporal_learning_rate=1e-3,
            base_loss_weight=0.5,
            seed=42,
            detector_model=str(detector_dir),
            architecture="dfine",
            image_size=64,
            device="cpu",
            hidden_dim=16,
            output=output,
            command="train",
        )
    )
    assert (output / "last" / "event_flow.pt").is_file()
    assert json.loads((output / "training.jsonl").read_text())["steps"] == 2
    predict(
        argparse.Namespace(
            manifest=path,
            model=output / "last",
            device="cpu",
            split="train",
            max_gap=0.05,
            max_detections=100,
            threshold=0.001,
            output=tmp_path / "evaluated",
            command="evaluate",
        )
    )
    assert (tmp_path / "evaluated" / "metrics.json").is_file()
    records = (tmp_path / "evaluated" / "predictions.jsonl").read_text().splitlines()
    assert len(records) == 4


def test_learned_temporal_head_uses_event_occupancy():
    model = EventFlow(tiny_detector(), hidden_dim=16).eval()
    torch.nn.init.normal_(model.temporal.class_delta.weight, std=0.1)
    rgb = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        empty = model(torch.cat((rgb, torch.zeros(1, 1, 64, 64)), dim=1))
        active = model(torch.cat((rgb, torch.ones(1, 1, 64, 64)), dim=1))
    # The detector stem's event weights are still zero here: the difference
    # comes from the explicit event context in the temporal detection head.
    torch.testing.assert_close(empty.base.query_logits, active.base.query_logits)
    assert not torch.allclose(empty.logits, active.logits)


def test_predictions_restore_non_square_padded_box_coordinates():
    from flowsis.matching import corners_to_center

    model = EventFlow(tiny_detector(), hidden_dim=16).eval()
    original = torch.tensor([[8.0, 4.0, 40.0, 20.0]])
    size = (35, 73)
    normalized = normalized_targets(original, original_size=size, image_size=64)
    with torch.no_grad():
        output = model(torch.zeros(1, 4, 64, 64))
    output.boxes = corners_to_center(normalized)[None]
    output.logits = torch.full((1, 1, 1), 10.0)
    result = model.detections(output, image_size=64, original_size=size)[0]
    torch.testing.assert_close(result["boxes"], original)
