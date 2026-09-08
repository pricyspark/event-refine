import json
import torch
import warnings
from os import PathLike
from pathlib import Path

from flowsis.pretrained.siglip2 import SigLIP2
from flowsis.utils import get_device


def apply_template(label: str) -> str:
    return f"This is a photo of {label}."


class LabelPrompts:
    def __init__(self) -> None:
        self.prompts: dict[str, list[str]] = {}
        self.embeddings: dict[str, torch.Tensor] = {}
        self._siglip2: SigLIP2 | None = None

    @property
    def siglip2(self) -> SigLIP2:
        if self._siglip2 is None:
            self._siglip2 = SigLIP2.from_pretrained(device=get_device())
        assert self._siglip2 is not None
        return self._siglip2

    def add(self, label: str, prompt: str) -> None:
        if label in self.prompts:
            warnings.warn(f"Label {label} already has prompts. Overwriting.")

        self.prompts[label] = [
            label,
            apply_template(label),
            prompt,
        ]
        self.embeddings[label] = self.siglip2(texts=self.prompts[label])

    @classmethod
    def load(cls, input_path: PathLike) -> "LabelPrompts":
        input_path = Path(input_path)
        with input_path.open() as fp:
            prompts = json.load(fp)

        if not isinstance(prompts, dict):
            raise ValueError(f"Expected a JSON object in {input_path}")

        label_prompts = cls()
        for label, prompt_list in prompts.items():
            if not isinstance(label, str) or not isinstance(prompt_list, list):
                raise ValueError(
                    f"Expected each entry in {input_path} to map a label to a list of prompts"
                )
            if not prompt_list or not all(
                isinstance(prompt, str) for prompt in prompt_list
            ):
                raise ValueError(
                    f"Prompts for {label!r} must be a non-empty list of strings"
                )
            label_prompts.prompts[label] = prompt_list

        return label_prompts

    def embed_all(self) -> None:
        self.embeddings = {
            label: self.siglip2(texts=prompts)
            for label, prompts in self.prompts.items()
        }

    def dump(
        self, output_path: PathLike, *, indent=None, separators=None, sort_keys=False
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as fp:
            json.dump(
                self.prompts,
                fp,
                indent=indent,
                separators=separators,
                sort_keys=sort_keys,
            )

    def dumps(self, *, indent=None, separators=None, sort_keys=False) -> str:
        return json.dumps(
            self.prompts, indent=indent, separators=separators, sort_keys=sort_keys
        )

    def save_embeddings(self, embedding_dir: Path) -> None:
        embedding_dir.mkdir(parents=True, exist_ok=True)
        for label, embedding in self.embeddings.items():
            torch.save(embedding, embedding_dir / f"{label}.pt")


"""


[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "flowsis"
version = "0.1.0"
description = "FlowSIS training and inference code"
readme = "README.md"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
include = ["flowsis*"]

[project.scripts]
flowsis-annotate-keyframes = "flowsis.cli.data.annotate_keyframes:main"
flowsis-auto-text-prompts = "flowsis.cli.data.auto_text_prompts:main"
flowsis-build-hf-detection-dataset = "flowsis.cli.data.build_hf_detection_dataset:main"
flowsis-build-hf-segmentation-dataset = "flowsis.cli.data.build_hf_segmentation_dataset:main"
flowsis-create-classes = "flowsis.cli.data.create_classes:main"
flowsis-embed-text-prompts = "flowsis.cli.data.embed_text_prompts:main"
flowsis-live-rtdetrv2 = "flowsis.cli.deploy.live_rtdetrv2:main"
flowsis-manual-text-prompts = "flowsis.cli.data.manual_text_prompts:main"
flowsis-reset = "flowsis.cli.data.reset:main"
flowsis-sample-keyframes = "flowsis.cli.data.sample_keyframes:main"
flowsis-train-base = "flowsis.cli.train.train_base:main"
flowsis-train-rtdetrv2 = "flowsis.cli.train.train_rtdetrv2:main"
flowsis-visualize-detection-dataset = "flowsis.cli.data.visualize_hf_detection_dataset:main"
flowsis-visualize-rtdetrv2-augmentations = "flowsis.cli.train.visualize_rtdetrv2_augmentations:main"

"""
