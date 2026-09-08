from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import __version__ as transformers_version

from flowsis.artifacts import atomic_torch_save
from flowsis.base import FlowSISBase
from flowsis.pretrained import DetectorArchitecture, load_detector
from flowsis.set_mask import SetMaskHead

BASE_CHECKPOINT_FILE = "base.pt"
BASE_CHECKPOINT_VERSION = 2
BASE_ARCHITECTURE = "set_mask_base"


@dataclass(frozen=True)
class BaseCheckpoint:
    head_config: dict[str, Any]
    head_state_dict: Mapping[str, torch.Tensor]
    prompt_embeddings: torch.Tensor
    label_names: tuple[str, ...]
    detection_threshold: float
    mask_threshold: float
    max_detections: int | None
    metadata: dict[str, Any]


def save_base_checkpoint(model: FlowSISBase, directory: str | Path) -> Path:
    """Save a self-contained detector directory and set-mask bundle."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.detector.save_pretrained(output_dir)
    path = output_dir / BASE_CHECKPOINT_FILE
    atomic_torch_save(
        {
            "format_version": BASE_CHECKPOINT_VERSION,
            "architecture": BASE_ARCHITECTURE,
            "head_config": dict(model.head_config),
            "head_state_dict": model.head.state_dict(),
            "prompt_embeddings": model.head.prompt_embeddings.detach().cpu(),
            "label_names": list(model.label_names),
            "detection_threshold": model.detection_threshold,
            "mask_threshold": model.mask_threshold,
            "max_detections": model.max_detections,
            "metadata": {
                "detector_architecture": model.detector.architecture,
                "detector_model_type": model.detector.model_config.get("model_type"),
                "num_queries": model.detector.model_config.get("num_queries"),
                "transformers_version": transformers_version,
                "mask_stride": 4 if model.head.pixel_decoder.use_detail_features else 8,
            },
        },
        path,
    )
    return path


def resolve_base_checkpoint(path: str | Path) -> Path:
    input_path = Path(path)
    if input_path.is_file():
        return input_path

    candidates = [
        input_path / BASE_CHECKPOINT_FILE,
        input_path / "final" / BASE_CHECKPOINT_FILE,
    ]
    last_checkpoint = input_path / "last_checkpoint"
    if last_checkpoint.exists():
        checkpoint_dir = Path(last_checkpoint.read_text().strip())
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = input_path / checkpoint_dir
        candidates.append(checkpoint_dir / BASE_CHECKPOINT_FILE)
    checkpoints = sorted(
        (
            checkpoint
            for checkpoint in input_path.glob("checkpoint-*")
            if checkpoint.is_dir()
            and checkpoint.name.removeprefix("checkpoint-").isdigit()
        ),
        key=lambda checkpoint: int(checkpoint.name.removeprefix("checkpoint-")),
        reverse=True,
    )
    candidates.extend(checkpoint / BASE_CHECKPOINT_FILE for checkpoint in checkpoints)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {BASE_CHECKPOINT_FILE} under {input_path}."
    )


def load_base_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[BaseCheckpoint, Path]:
    checkpoint_path = resolve_base_checkpoint(path)
    bundle = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected a base checkpoint mapping at {checkpoint_path}.")
    if bundle.get("format_version") != BASE_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported base checkpoint version at {checkpoint_path}: "
            f"{bundle.get('format_version')!r}."
        )
    if bundle.get("architecture") != BASE_ARCHITECTURE:
        raise ValueError(
            f"Checkpoint {checkpoint_path} contains architecture "
            f"{bundle.get('architecture')!r}, not {BASE_ARCHITECTURE!r}."
        )
    config = bundle.get("head_config")
    state_dict = bundle.get("head_state_dict")
    prompts = bundle.get("prompt_embeddings")
    labels = bundle.get("label_names")
    metadata = bundle.get("metadata")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise TypeError("Base checkpoint is missing head configuration or weights.")
    if not isinstance(prompts, torch.Tensor) or prompts.ndim != 3:
        raise TypeError("Base checkpoint prompt bank must be a [C,P,D] tensor.")
    if not isinstance(labels, list) or not all(isinstance(v, str) for v in labels):
        raise TypeError("Base checkpoint label_names must be an ordered string list.")
    if any(not isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise TypeError("Base checkpoint contains a non-tensor model parameter.")
    if not isinstance(metadata, dict):
        raise TypeError("Base checkpoint is missing architecture metadata.")
    checkpoint = BaseCheckpoint(
        head_config=config,
        head_state_dict=state_dict,
        prompt_embeddings=prompts.float(),
        label_names=tuple(labels),
        detection_threshold=float(bundle.get("detection_threshold", 0.5)),
        mask_threshold=float(bundle.get("mask_threshold", 0.5)),
        max_detections=bundle.get("max_detections"),
        metadata=metadata,
    )
    return checkpoint, checkpoint_path


def load_base(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    detector_architecture: DetectorArchitecture | None = None,
    use_amp: bool = True,
    detector_amp: bool = False,
    device_preprocess: bool = True,
) -> FlowSISBase:
    checkpoint, checkpoint_path = load_base_checkpoint(path)
    model_dir = checkpoint_path.parent
    detector = load_detector(
        str(model_dir),
        architecture=detector_architecture,
        device=device,
    )
    head = SetMaskHead(
        prompt_embeddings=checkpoint.prompt_embeddings,
        **checkpoint.head_config,
    )
    head.load_state_dict(checkpoint.head_state_dict, strict=True)
    model = FlowSISBase(
        detector,
        head,
        checkpoint.label_names,
        head_config=checkpoint.head_config,
        detection_threshold=checkpoint.detection_threshold,
        mask_threshold=checkpoint.mask_threshold,
        max_detections=checkpoint.max_detections,
        use_amp=use_amp,
        detector_amp=detector_amp,
        device_preprocess=device_preprocess,
    )
    model.to(device)
    model.eval()
    return model


__all__ = [
    "BASE_CHECKPOINT_FILE",
    "BaseCheckpoint",
    "load_base",
    "load_base_checkpoint",
    "resolve_base_checkpoint",
    "save_base_checkpoint",
]
