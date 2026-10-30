import json
from pathlib import Path

import pytest
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from flowsis.cli.data.author_text_prompts import existing_descriptions, prompt_draft
from flowsis.cli.data.embed_text_prompts import (
    select_prompt_types,
    taxonomy_from_dataset,
)
from flowsis.cli.data.generate_text_prompts import generate_descriptions
from flowsis.data.prompts import (
    LabelPrompts,
    build_prompt_suite,
    load_prompt_bank,
    siglip2_pipeline_prompt,
    validate_prompt_suite,
)


def test_load_embeddings_moves_cached_tensor_to_requested_device(
    tmp_path: Path,
) -> None:
    torch.save(torch.ones(2, 3), tmp_path / "object.pt")
    cache: dict[str, torch.Tensor] = {}

    cpu_embeddings = LabelPrompts.load_embeddings(
        "object",
        tmp_path,
        cache,
        device=torch.device("cpu"),
    )
    meta_embeddings = LabelPrompts.load_embeddings(
        "object",
        tmp_path,
        cache,
        device=torch.device("meta"),
    )

    assert cpu_embeddings.device.type == "cpu"
    assert meta_embeddings.device.type == "meta"
    assert cache["object"] is meta_embeddings


def test_prompt_bank_preserves_label_order_and_requires_rectangular_shape(
    tmp_path: Path,
) -> None:
    torch.save(torch.full((2, 768), 1.0), tmp_path / "first.pt")
    torch.save(torch.full((2, 768), 2.0), tmp_path / "second.pt")
    bank = load_prompt_bank(("second", "first"), tmp_path)
    assert bank.shape == (2, 2, 768)
    assert torch.equal(bank[0], torch.full((2, 768), 2.0))

    torch.save(torch.ones(1, 768), tmp_path / "second.pt")
    with pytest.raises(ValueError, match="same prompt count"):
        load_prompt_bank(("first", "second"), tmp_path)


def test_prompt_suite_uses_full_label_exact_template_and_description() -> None:
    taxonomy = {"iris scissors": "scissors", "mallet": "mallet"}
    descriptions = {
        "iris scissors": "iris scissors with short fine blades and ring handles",
        "mallet": "mallet with a broad cylindrical striking head",
    }
    suites = build_prompt_suite(taxonomy, descriptions)

    assert suites["iris scissors"] == [
        "iris scissors",
        "This is a photo of iris scissors.",
        "iris scissors with short fine blades and ring handles",
    ]
    assert len(suites["mallet"]) == len(suites["iris scissors"]) == 3
    assert siglip2_pipeline_prompt("mayo scissors") == (
        "This is a photo of mayo scissors."
    )


def test_custom_prompt_suite_must_match_taxonomy_and_be_rectangular() -> None:
    taxonomy = {"iris scissors": "scissors", "mallet": "mallet"}
    with pytest.raises(ValueError, match="missing"):
        validate_prompt_suite(
            {
                "iris scissors": [
                    "iris scissors",
                    "This is a photo of iris scissors.",
                    "iris scissors with short fine blades",
                ]
            },
            taxonomy,
        )
    with pytest.raises(ValueError, match="exactly three"):
        validate_prompt_suite(
            {
                "iris scissors": [
                    "iris scissors",
                    "This is a photo of iris scissors.",
                    "iris scissors with short fine blades",
                ],
                "mallet": ["mallet", "This is a photo of mallet."],
            },
            taxonomy,
        )


def test_manual_prompt_draft_is_resumable(tmp_path: Path) -> None:
    taxonomy = {"iris scissors": "scissors"}
    draft = prompt_draft(taxonomy, {})
    assert draft == {
        "iris scissors": [
            "iris scissors",
            "This is a photo of iris scissors.",
            "",
        ]
    }

    path = tmp_path / "prompts.json"
    draft["iris scissors"][2] = "iris scissors with short fine blades"
    path.write_text(json.dumps(draft), encoding="utf-8")

    assert existing_descriptions(path, taxonomy) == {
        "iris scissors": "iris scissors with short fine blades"
    }


def test_prompt_type_selection_supports_clean_ablations() -> None:
    suites = {
        "iris scissors": [
            "iris scissors",
            "This is a photo of iris scissors.",
            "iris scissors with short fine blades",
        ]
    }

    assert select_prompt_types(suites, ["description"]) == {
        "iris scissors": ["iris scissors with short fine blades"]
    }
    assert select_prompt_types(suites, ["description", "label"]) == {
        "iris scissors": [
            "iris scissors",
            "iris scissors with short fine blades",
        ]
    }


@pytest.mark.parametrize("label", ["scissors", "iris scissors", "scissors iris"])
def test_prompt_taxonomy_uses_exported_labels(tmp_path: Path, label: str) -> None:
    features = Features(
        {
            "objects": [
                {
                    "category": ClassLabel(names=[label]),
                    "tool_name": Value("string"),
                    "tool_class": Value("string"),
                }
            ]
        }
    )
    split = Dataset.from_list(
        [
            {
                "objects": [
                    {"category": 0, "tool_name": "iris", "tool_class": "scissors"}
                ]
            },
            {
                "objects": [
                    {"category": 0, "tool_name": "mayo", "tool_class": "scissors"}
                ]
            },
        ],
        features=features,
    )
    DatasetDict(train=split, validation=split).save_to_disk(tmp_path / "dataset")
    assert taxonomy_from_dataset(tmp_path / "dataset") == {label: "scissors"}


def test_prompt_taxonomy_rejects_different_split_label_order(tmp_path: Path) -> None:
    splits = {}
    for name, labels in (
        ("train", ["scissors", "forceps"]),
        ("validation", ["forceps", "scissors"]),
    ):
        features = Features(
            {
                "objects": [
                    {
                        "category": ClassLabel(names=labels),
                        "tool_class": Value("string"),
                    }
                ]
            }
        )
        splits[name] = Dataset.from_list(
            [{"objects": [{"category": 0, "tool_class": labels[0]}]}],
            features=features,
        )
    DatasetDict(splits).save_to_disk(tmp_path / "dataset")
    with pytest.raises(ValueError, match="same ordered class labels"):
        taxonomy_from_dataset(tmp_path / "dataset")


def test_optional_generator_records_reproducibility_metadata() -> None:
    class Response:
        id = "response-test"
        model = "test-model-snapshot"
        output_text = """{
          "instruments": [
            {
              "label": "iris scissors",
              "description": "iris scissors with short fine blades"
            }
          ]
        }"""

    class Responses:
        def create(self, **kwargs: object) -> Response:
            assert kwargs["store"] is False
            return Response()

    descriptions, provenance = generate_descriptions(
        {"iris scissors": "scissors"}, Responses(), model="requested-model"
    )

    assert descriptions == {"iris scissors": "iris scissors with short fine blades"}
    assert provenance["requested_model"] == "requested-model"
    assert provenance["response_model"] == "test-model-snapshot"
    assert provenance["review_required"] is True
