from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from flowsis.artifacts import atomic_write_text
from flowsis.cli.data.embed_text_prompts import taxonomy_from_dataset
from flowsis.data import siglip2_pipeline_prompt


def prompt_draft(
    taxonomy: Mapping[str, str], descriptions: Mapping[str, str]
) -> dict[str, list[str]]:
    """Create an editable suite; incomplete descriptions remain empty."""
    return {
        label: [
            label,
            siglip2_pipeline_prompt(label),
            " ".join(descriptions.get(label, "").strip().split()),
        ]
        for label in taxonomy
    }


def existing_descriptions(
    path: Path, taxonomy: Mapping[str, str]
) -> dict[str, str]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    descriptions = {}
    for label in taxonomy:
        prompts = raw.get(label)
        if prompts is None:
            continue
        if not isinstance(prompts, list) or len(prompts) != 3:
            raise ValueError(
                f"Existing prompt suite for {label!r} must contain three strings."
            )
        expected = [label, siglip2_pipeline_prompt(label)]
        if prompts[:2] != expected or not all(
            isinstance(prompt, str) for prompt in prompts
        ):
            raise ValueError(
                f"Existing prompt suite for {label!r} must begin with {expected!r}."
            )
        descriptions[label] = prompts[2]
    return descriptions


def save_draft(
    path: Path,
    taxonomy: Mapping[str, str],
    descriptions: Mapping[str, str],
) -> Path:
    return atomic_write_text(
        path,
        json.dumps(
            prompt_draft(taxonomy, descriptions),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )


def author_interactively(
    path: Path,
    taxonomy: Mapping[str, str],
    descriptions: dict[str, str],
) -> None:
    save_draft(path, taxonomy, descriptions)
    print(
        "Describe stable, visible geometry that distinguishes each tool. "
        "Avoid use, scene, anatomy, and transient color."
    )
    for label, coarse in taxonomy.items():
        current = " ".join(descriptions.get(label, "").strip().split())
        while True:
            print(f"\n{label}  [coarse class: {coarse or '-'}]")
            if current:
                print(f"current: {current}")
                entered = input("description (Enter keeps current): ").strip()
                description = " ".join((entered or current).split())
            else:
                description = " ".join(input("description: ").strip().split())
            if not description:
                print("A visual description is required.")
                continue
            if not description.casefold().startswith(label.casefold()):
                print(f"Start the description with the exact label {label!r}.")
                continue
            descriptions[label] = description
            save_draft(path, taxonomy, descriptions)
            break
    print(f"\nsaved_prompt_suite {{'path': '{path}', 'classes': {len(taxonomy)}}}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Author the visual-description prompt for every canonical Arrow "
            "class without an external API."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/segmentation-dataset"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/text_prompts.json"),
    )
    parser.add_argument(
        "--scaffold",
        action="store_true",
        help="Write an editable JSON draft with empty description fields.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --scaffold to replace an existing prompt suite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = taxonomy_from_dataset(args.dataset_path)
    if args.scaffold:
        if args.output_path.exists() and not args.force:
            raise FileExistsError(
                f"Prompt suite already exists: {args.output_path}. Edit it directly "
                "or pass --force to replace it with a blank scaffold."
            )
        save_draft(args.output_path, taxonomy, {})
        print(
            f"saved_prompt_scaffold {{'path': '{args.output_path}', "
            f"'classes': {len(taxonomy)}}}"
        )
        return

    descriptions = existing_descriptions(args.output_path, taxonomy)
    author_interactively(args.output_path, taxonomy, descriptions)


if __name__ == "__main__":
    main()
