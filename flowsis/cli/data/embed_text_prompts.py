import argparse
from collections import defaultdict
from pathlib import Path

import torch
from datasets import ClassLabel, DatasetDict, load_from_disk

from flowsis.data import LabelPrompts, validate_prompt_suite
from flowsis.data.object_records import get_object_feature_schema, get_object_records
from flowsis.utils import get_device

PROMPT_TYPES = ("label", "template", "description")


def select_prompt_types(
    suites: dict[str, list[str]], selected: list[str] | None
) -> dict[str, list[str]]:
    if not selected:
        return suites
    selected_set = set(selected)
    indices = [
        index
        for index, prompt_type in enumerate(PROMPT_TYPES)
        if prompt_type in selected_set
    ]
    return {
        label: [prompts[index] for index in indices]
        for label, prompts in suites.items()
    }


def taxonomy_from_dataset(dataset_path: Path) -> dict[str, str]:
    dataset = load_from_disk(str(dataset_path))
    if not isinstance(dataset, DatasetDict) or not dataset:
        raise TypeError("Expected --dataset-path to contain a nonempty DatasetDict.")
    first_split = next(iter(dataset.values()))
    schema = get_object_feature_schema(first_split.features["objects"])
    category = getattr(schema["category"], "feature", schema["category"])
    if not isinstance(category, ClassLabel):
        raise TypeError("objects.category must be a datasets.ClassLabel.")

    coarse_by_name: defaultdict[str, set[str]] = defaultdict(set)
    for split in dataset.values():
        split_schema = get_object_feature_schema(split.features["objects"])
        split_category = getattr(
            split_schema["category"], "feature", split_schema["category"]
        )
        if (
            not isinstance(split_category, ClassLabel)
            or split_category.names != category.names
        ):
            raise ValueError(
                "All dataset splits must use the same ordered class labels."
            )
        for example in split:
            for obj in get_object_records(example):
                coarse = " ".join(str(obj.get("tool_class", "")).split())
                label = category.int2str(int(obj["category"]))
                # Exported categories are authoritative: label fields may be
                # configured independently of the preserved tool metadata.
                if coarse:
                    coarse_by_name[label].add(coarse)

    taxonomy = {}
    for name in category.names:
        coarse_names = coarse_by_name[name]
        if len(coarse_names) > 1:
            raise ValueError(
                f"Canonical class {name!r} has conflicting coarse classes: "
                f"{sorted(coarse_names)}."
            )
        taxonomy[name] = next(iter(coarse_names), "")
    return taxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a reviewed prompt suite against the Arrow taxonomy and "
            "cache one pooled SigLIP2 embedding per prompt."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/new/segmentation-dataset"),
    )
    parser.add_argument(
        "--prompt-suite",
        type=Path,
        default=Path("data/new/text_prompts.json"),
        help="Reviewed JSON created by flowsis-author-text-prompts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/new/text-embeddings"),
    )
    parser.add_argument(
        "--prompt-type",
        action="append",
        choices=PROMPT_TYPES,
        help=(
            "Embed only this prompt type for an ablation. Repeat to select more "
            "than one; the default embeds all three."
        ),
    )
    parser.add_argument(
        "--model",
        dest="model_source",
        default="google/siglip2-base-patch16-224",
        metavar="MODEL",
        help="Hugging Face Hub model ID or local model path.",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device is not None else get_device()
    taxonomy = taxonomy_from_dataset(args.dataset_path)
    label_prompts = LabelPrompts.load(
        args.prompt_suite,
        model_name_or_path=args.model_source,
        device=device,
    )
    validate_prompt_suite(label_prompts.prompts, taxonomy)
    label_prompts.prompts = select_prompt_types(label_prompts.prompts, args.prompt_type)
    embeddings = label_prompts.embed_all()
    label_prompts.save_embeddings(args.output_dir)
    for label, embedding in embeddings.items():
        print(label, tuple(embedding.shape))


if __name__ == "__main__":
    main()
