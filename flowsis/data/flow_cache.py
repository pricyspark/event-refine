from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from flowsis.artifacts import atomic_torch_save, atomic_write_text


def _tensor_digest(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tensor.shape, tensor.dtype)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FrozenFlowCache:
    """Persistent backward flows for exact, ordered input frame pairs.

    Call from the training process, with a frozen estimator in eval mode.
    Content keys distinguish stride, crop, resize, and flip without relying on
    filenames or annotation metadata. Entries retain the estimator's precision.
    """

    def __init__(
        self, directory: Path, estimator: torch.nn.Module, *, batch_size: int = 2
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Flow cache batch size must be positive.")
        if estimator.training or any(
            parameter.requires_grad for parameter in estimator.parameters()
        ):
            raise ValueError("Flow caching requires a frozen estimator in eval mode.")
        self.estimator = estimator
        self.batch_size = batch_size
        try:
            ptlflow_version = version("ptlflow")
        except PackageNotFoundError:
            ptlflow_version = "unavailable"
        metadata = {
            "format_version": 1,
            "torch_version": torch.__version__,
            "ptlflow_version": ptlflow_version,
            "modules": [
                (
                    name,
                    f"{type(module).__module__}.{type(module).__qualname__}",
                    repr(module),
                    repr(getattr(module, "hparams", {})),
                )
                for name, module in estimator.named_modules()
            ],
            "state": {
                name: _tensor_digest(tensor)
                for name, tensor in estimator.state_dict().items()
            },
        }
        signature = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()
        self.directory = directory / signature
        self.directory.mkdir(parents=True, exist_ok=True)
        # Human-readable provenance alongside the content-addressed entries.
        atomic_write_text(
            self.directory / "metadata.json", json.dumps(metadata, indent=2) + "\n"
        )

    @torch.no_grad()
    def __call__(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        if current.shape != previous.shape or current.ndim != 4:
            raise ValueError("Flow cache inputs must have matching [B, 3, H, W] shapes.")
        if current.shape[1] != 3 or current.shape[0] == 0:
            raise ValueError("Flow cache inputs require a nonempty RGB batch.")
        if self.estimator.training or any(
            parameter.requires_grad for parameter in self.estimator.parameters()
        ):
            raise ValueError("Flow caching requires a frozen estimator in eval mode.")
        device_type = current.device.type
        execution = (
            device_type,
            torch.is_autocast_enabled(device_type),
            str(torch.get_autocast_dtype(device_type)),
            torch.get_float32_matmul_precision(),
            torch.backends.cudnn.allow_tf32,
        )
        current_cpu = current.detach().cpu()
        previous_cpu = previous.detach().cpu()
        paths = []
        missing: dict[Path, int] = {}
        flows: dict[Path, torch.Tensor] = {}
        for index in range(len(current)):
            key = hashlib.sha256(
                repr(
                    (
                        execution,
                        _tensor_digest(current_cpu[index]),
                        _tensor_digest(previous_cpu[index]),
                    )
                ).encode()
            ).hexdigest()
            path = self.directory / key[:2] / f"{key}.pt"
            paths.append(path)
            if path in flows or path in missing:
                continue
            if path.is_file():
                flow = torch.load(path, map_location="cpu", weights_only=True)
                if (
                    not isinstance(flow, torch.Tensor)
                    or not flow.is_floating_point()
                    or flow.shape != (2, *current.shape[-2:])
                ):
                    raise ValueError(f"Invalid cached flow at {path}.")
                flows[path] = flow
            else:
                missing[path] = index

        pending = list(missing.items())
        for start in range(0, len(pending), self.batch_size):
            chunk = pending[start : start + self.batch_size]
            indices = [index for _, index in chunk]
            predictions = self.estimator(current[indices], previous[indices])
            if predictions.shape != (len(chunk), 2, *current.shape[-2:]):
                raise ValueError("Flow estimator returned an unexpected shape.")
            for (path, _), flow in zip(chunk, predictions.detach().cpu(), strict=True):
                atomic_torch_save(flow.clone(), path)
                flows[path] = flow
        return torch.stack([flows[path] for path in paths]).to(current.device)
