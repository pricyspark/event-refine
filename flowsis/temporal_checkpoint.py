from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from flowsis.artifacts import atomic_torch_save
from flowsis.temporal import TemporalRefinementBranch

TEMPORAL_CHECKPOINT_FILE = "temporal.pt"
TEMPORAL_CHECKPOINT_VERSION = 2


def save_temporal_checkpoint(
    model: TemporalRefinementBranch,
    directory: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    return atomic_torch_save(
        {
            "format_version": TEMPORAL_CHECKPOINT_VERSION,
            "architecture": "residual_temporal_refiner",
            "state_dict": model.state_dict(),
            "config": {
                "channels": model.channels,
                "feature_channels": model.feature_channels,
                "context_channels": model.context_channels,
                "hidden_channels": model.hidden_channels,
                "residual_limit": model.residual_limit,
            },
            "metadata": dict(metadata or {}),
        },
        Path(directory) / TEMPORAL_CHECKPOINT_FILE,
    )


def load_temporal_checkpoint(
    source: str | Path, *, device: torch.device | str = "cpu"
) -> TemporalRefinementBranch:
    source = Path(source)
    path = source / TEMPORAL_CHECKPOINT_FILE if source.is_dir() else source
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if int(payload.get("format_version", -1)) != TEMPORAL_CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported temporal checkpoint format: {path}")
    if payload.get("architecture") != "residual_temporal_refiner":
        raise ValueError(f"Unexpected temporal architecture in {path}.")
    config = dict(payload["config"])
    model = TemporalRefinementBranch(**config)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


__all__ = [
    "TEMPORAL_CHECKPOINT_FILE",
    "TEMPORAL_CHECKPOINT_VERSION",
    "load_temporal_checkpoint",
    "save_temporal_checkpoint",
]
