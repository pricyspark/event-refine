"""Self-contained Event-Flow checkpoints, developed with Codex assistance."""

from pathlib import Path

import torch

from flowsis.artifacts import atomic_torch_save
from flowsis.pretrained.detector import load_detector, BaseDetector

from .model import EventFlow


def save_checkpoint(
    model: EventFlow, path: str | Path, *, classes: tuple[str, ...]
) -> None:
    if classes != tuple(
        model.detector.label_names[index]
        for index in sorted(model.detector.label_names)
    ):
        raise ValueError("Class names do not match the model.")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.detector.save_pretrained(path / "detector")
    atomic_torch_save(
        {
            "format_version": 1,
            "classes": classes,
            "hidden_dim": model.temporal.hidden_dim,
            "temporal": model.temporal.state_dict(),
        },
        path / "event_flow.pt",
    )


def load_checkpoint(
    path: str | Path, *, device: str = "cpu"
) -> tuple[EventFlow, tuple[str, ...]]:
    path = Path(path)
    saved = torch.load(path / "event_flow.pt", map_location="cpu", weights_only=True)
    if saved["format_version"] != 1:
        raise ValueError("Unsupported Event-Flow checkpoint version.")
    detector = load_detector(path / "detector", device=device)
    if not isinstance(detector, BaseDetector):
        raise TypeError("Expected a native detector adapter.")
    model = EventFlow(detector, hidden_dim=saved["hidden_dim"])
    model.temporal.load_state_dict(saved["temporal"], strict=True)
    classes = tuple(saved["classes"])
    if classes != tuple(
        model.detector.label_names[index]
        for index in sorted(model.detector.label_names)
    ):
        raise ValueError("Checkpoint classes do not match the detector.")
    return model.to(device).eval(), classes
