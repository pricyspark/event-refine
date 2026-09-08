from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from flowsis.pretrained.siglip2 import SigLIP2


def siglip2_pipeline_prompt(label: str) -> str:
    """Return the exact template used by Transformers' SigLIP2 pipeline."""
    return f"This is a photo of {label}."


def build_prompt_suite(
    taxonomy: Mapping[str, str], descriptions: Mapping[str, str]
) -> dict[str, list[str]]:
    """Build the three prompt types used for every canonical tool label."""
    missing = set(taxonomy) - set(descriptions)
    extra = set(descriptions) - set(taxonomy)
    if missing or extra:
        raise ValueError(
            "Description labels do not match the dataset taxonomy: "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )
    suites = {
        label: [
            label,
            siglip2_pipeline_prompt(label),
            " ".join(descriptions[label].strip().split()),
        ]
        for label in taxonomy
    }
    validate_prompt_suite(suites, taxonomy)
    return suites


def validate_prompt_suite(
    suites: Mapping[str, Sequence[str]], taxonomy: Mapping[str, str]
) -> None:
    missing = set(taxonomy) - set(suites)
    extra = set(suites) - set(taxonomy)
    if missing or extra:
        raise ValueError(
            "Prompt-suite labels do not match the dataset taxonomy: "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )
    for label, prompts in suites.items():
        if len(prompts) != 3:
            raise ValueError(
                f"Prompt suite for {label!r} must contain exactly three prompts: "
                "canonical label, SigLIP2 pipeline template, and visual description."
            )
        expected = [label, siglip2_pipeline_prompt(label)]
        if list(prompts[:2]) != expected:
            raise ValueError(
                f"Prompt suite for {label!r} must begin with {expected!r}."
            )
        description = prompts[2].strip()
        if not description or description.casefold() in {
            label.casefold(),
            expected[1].casefold(),
        }:
            raise ValueError(
                f"Prompt suite for {label!r} needs a distinct visual description."
            )


class LabelPrompts:
    def __init__(
        self,
        *,
        model_name_or_path: str = "google/siglip2-base-patch16-224",
        device: str | torch.device | None = None,
    ) -> None:
        self.prompts: dict[str, list[str]] = {}
        self.embeddings: dict[str, torch.Tensor] = {}
        self.model_name_or_path = model_name_or_path
        self.device = device
        self._siglip2: SigLIP2 | None = None

    @property
    def siglip2(self) -> SigLIP2:
        """Load the text encoder only when embeddings are requested."""
        if self._siglip2 is None:
            from flowsis.pretrained.siglip2 import SigLIP2

            self._siglip2 = SigLIP2.from_pretrained(
                self.model_name_or_path,
                return_mode="pooled",
                normalize=True,
                device=self.device,
            )
        return self._siglip2

    @classmethod
    def from_mapping(
        cls,
        prompts: Mapping[str, Sequence[str]],
        *,
        model_name_or_path: str = "google/siglip2-base-patch16-224",
        device: str | torch.device | None = None,
    ) -> "LabelPrompts":
        result = cls(model_name_or_path=model_name_or_path, device=device)
        for label, values in prompts.items():
            if (
                not label
                or not values
                or not all(isinstance(value, str) for value in values)
            ):
                raise ValueError(
                    "Each prompt-suite entry must map a nonempty label to "
                    "a nonempty sequence of strings."
                )
            result.prompts[label] = list(values)
        return result

    @classmethod
    def load(
        cls,
        input_path: PathLike[str],
        *,
        model_name_or_path: str = "google/siglip2-base-patch16-224",
        device: str | torch.device | None = None,
    ) -> "LabelPrompts":
        input_path = Path(input_path)
        with input_path.open(encoding="utf-8") as file:
            prompts = json.load(file)

        if not isinstance(prompts, dict):
            raise ValueError(f"Expected a JSON object in {input_path}")

        if not all(
            isinstance(label, str) and isinstance(values, list)
            for label, values in prompts.items()
        ):
            raise ValueError(
                f"Expected each entry in {input_path} to map a label to a list "
                "of prompts."
            )
        return cls.from_mapping(
            prompts,
            model_name_or_path=model_name_or_path,
            device=device,
        )

    @staticmethod
    def load_embeddings(
        label: str,
        directory: Path,
        cache: dict[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if label not in cache:
            path = directory / f"{label}.pt"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing prompt embeddings for detector label {label!r}: {path}"
                )
            embeddings = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
                shape = getattr(embeddings, "shape", None)
                raise ValueError(
                    f"Expected prompt embeddings shaped [P,D] at {path}, got {shape}."
                )
            cache[label] = embeddings.float()
        # Tensor.to is a no-op when the cached tensor is already on the target
        # device, and migrates it if the caller moved the model since caching.
        cache[label] = cache[label].to(device)
        return cache[label]

    def embed_all(self) -> dict[str, torch.Tensor]:
        """Create one normalized, pooled semantic vector per prompt."""
        self.embeddings = {
            label: self.siglip2(texts=prompts)
            for label, prompts in self.prompts.items()
        }
        return self.embeddings

    def save_embeddings(self, embedding_dir: PathLike[str]) -> None:
        embedding_dir = Path(embedding_dir)
        embedding_dir.mkdir(parents=True, exist_ok=True)
        for label, embedding in self.embeddings.items():
            output_path = embedding_dir / f"{label}.pt"
            if output_path.parent != embedding_dir:
                raise ValueError(
                    f"Label {label!r} cannot be used as an embedding filename"
                )
            torch.save(embedding.cpu(), output_path)


def load_prompt_bank(
    label_names: list[str] | tuple[str, ...],
    directory: PathLike[str],
    *,
    expected_dimension: int = 768,
) -> torch.Tensor:
    """Load an ordered, rectangular [classes,prompts,dimension] prompt bank."""
    embedding_dir = Path(directory)
    cache: dict[str, torch.Tensor] = {}
    embeddings = [
        LabelPrompts.load_embeddings(
            label,
            embedding_dir,
            cache,
            device=torch.device("cpu"),
        )
        for label in label_names
    ]
    if not embeddings:
        raise ValueError("At least one detector label is required for prompt loading.")
    expected_shape = embeddings[0].shape
    mismatches = {
        label: tuple(embedding.shape)
        for label, embedding in zip(label_names, embeddings, strict=True)
        if embedding.shape != expected_shape
    }
    if mismatches:
        raise ValueError(
            "Every detector class must have the same prompt count and dimension; "
            f"expected {tuple(expected_shape)}, got {mismatches}."
        )
    if expected_shape[1] != expected_dimension:
        raise ValueError(
            f"Expected {expected_dimension}-dimensional pooled prompt embeddings, "
            f"received {expected_shape[1]}."
        )
    return torch.stack(embeddings)
