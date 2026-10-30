from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol

from flowsis.artifacts import atomic_write_text
from flowsis.cli.data.embed_text_prompts import taxonomy_from_dataset
from flowsis.data import build_prompt_suite

DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"

INSTRUCTIONS = """You write visual descriptions for surgical-instrument recognition.

For each supplied canonical label, produce one concise noun phrase that:
- starts with the exact canonical label;
- adds stable, visible features that distinguish it from the other supplied labels;
- prioritizes shape, shaft, jaws, blades, tips, teeth, serrations, handles,
  fenestrations, and other geometry visible in a photograph;
- describes only appearance, not use, procedure, anatomy, scene, hands, tissue,
  blood, labels, masks, or embeddings;
- does not add generic features merely to make the phrase longer;
- does not invent a feature when the label does not imply one.

Consider the complete taxonomy together. Return exactly one entry for every label.
"""


class Responses(Protocol):
    def create(self, **kwargs: object) -> object: ...


def _response_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "instruments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["label", "description"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["instruments"],
        "additionalProperties": False,
    }


def generate_descriptions(
    taxonomy: dict[str, str], responses: Responses, *, model: str
) -> tuple[dict[str, str], dict[str, object]]:
    response = responses.create(
        model=model,
        instructions=INSTRUCTIONS,
        input=json.dumps(
            [
                {"canonical_label": label, "coarse_class": coarse}
                for label, coarse in taxonomy.items()
            ],
            indent=2,
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "instrument_descriptions",
                "strict": True,
                "schema": _response_schema(),
            }
        },
        store=False,
    )
    output_text = getattr(response, "output_text", "")
    if not output_text:
        raise RuntimeError("The description model returned no text output.")
    entries = json.loads(output_text)["instruments"]
    descriptions: dict[str, str] = {}
    for entry in entries:
        label = " ".join(str(entry["label"]).strip().split())
        description = " ".join(str(entry["description"]).strip().split())
        if label in descriptions:
            raise ValueError(
                f"The description model returned duplicate label {label!r}."
            )
        descriptions[label] = description

    missing = set(taxonomy) - set(descriptions)
    extra = set(descriptions) - set(taxonomy)
    if missing or extra:
        raise ValueError(
            "Generated description labels do not match the Arrow taxonomy: "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )
    for label, description in descriptions.items():
        if not description.casefold().startswith(label.casefold()):
            raise ValueError(
                f"Description for {label!r} must start with its canonical label: "
                f"{description!r}."
            )

    provenance = {
        "schema_version": 1,
        "method": "openai_responses_api",
        "requested_model": model,
        "response_model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "store": False,
        "instructions": INSTRUCTIONS,
        "taxonomy": taxonomy,
        "review_required": True,
    }
    return descriptions, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optionally generate visual-description prompt drafts with the "
            "OpenAI API. Manual authoring remains fully supported."
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
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing prompt suite with a new generated draft.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_path.exists() and not args.force:
        raise FileExistsError(
            f"Prompt suite already exists: {args.output_path}. Review it manually "
            "or pass --force to regenerate the complete draft."
        )
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "Install the optional generator with "
            "`pip install -e '.[prompt-generation]'`."
        ) from error

    taxonomy = taxonomy_from_dataset(args.dataset_path)
    descriptions, provenance = generate_descriptions(
        taxonomy, OpenAI().responses, model=args.model
    )
    suites = build_prompt_suite(taxonomy, descriptions)
    atomic_write_text(
        args.output_path,
        json.dumps(suites, indent=2, ensure_ascii=False) + "\n",
    )
    provenance_path = args.output_path.with_name(
        f"{args.output_path.stem}.provenance.json"
    )
    atomic_write_text(
        provenance_path,
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
    )
    print(
        "saved_prompt_suite",
        {
            "path": str(args.output_path),
            "provenance": str(provenance_path),
            "classes": len(suites),
            "review_required": True,
        },
    )


if __name__ == "__main__":
    main()
