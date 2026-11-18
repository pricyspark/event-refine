"""Generation-comparison safeguards, prepared with Codex assistance."""

import hashlib

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
import pytest

from experiments.benchmark_generations import common_validation, remap_categories


def test_category_mapping_uses_names_when_new_labels_shift_indices():
    data = Dataset.from_list([{"objects": [{"category": 2}]}])
    mapped = remap_categories(
        data, ("forceps", "needleholder", "osteotome"), ("forceps", "osteotome")
    )
    assert mapped[0]["objects"][0]["category"] == 1
    assert data[0]["objects"][0]["category"] == 2
    with pytest.raises(ValueError, match="absent"):
        remap_categories(data, ("forceps", "needleholder", "osteotome"), ("forceps",))


def cohorts(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"same original image")
    mask = tmp_path / "mask.npz"
    mask.write_bytes(b"mask hash placeholder")
    features = Features(
        {
            "image_id": Value("string"),
            "image_path": Value("string"),
            "sample_id": Value("string"),
            "video_id": Value("string"),
            "objects": [
                {
                    "category": ClassLabel(names=["forceps"]),
                    "mask_path": Value("string"),
                }
            ],
        }
    )
    shared = {
        "image_id": "shared",
        "image_path": str(image),
        "video_id": "",
        "objects": [{"category": 0, "mask_path": str(mask)}],
    }
    # The current held-out ID happens to equal a PRIOR training ID after renumbering.
    prior = DatasetDict(
        train=Dataset.from_list(
            [{**shared, "image_id": "old-train", "sample_id": "new-validation"}],
            features=features,
        ),
        validation=Dataset.from_list(
            [
                {
                    **shared,
                    "sample_id": "old-validation",
                    "image_path": "old/moved-image.jpg",
                }
            ],
            features=features,
        ),
    )
    current = DatasetDict(
        train=Dataset.from_list(
            [{**shared, "image_id": "new-train", "sample_id": "new-training"}],
            features=features,
        ),
        validation=Dataset.from_list(
            [{**shared, "sample_id": "new-validation"}], features=features
        ),
    )
    return (
        prior,
        current,
        {"old/moved-image.jpg": hashlib.sha256(image.read_bytes()).hexdigest()},
    )


def test_common_cohort_checks_groups_in_each_export_and_hashes_media(tmp_path):
    prior, current, hashes = cohorts(tmp_path)
    selected, metadata = common_validation(prior, current, hashes)
    assert len(selected) == metadata["images"] == 1
    assert metadata["frames"][0]["prior_sample_id"] == "old-validation"
    assert metadata["frames"][0]["current_sample_id"] == "new-validation"


def test_changed_image_bytes_are_excluded(tmp_path):
    prior, current, _ = cohorts(tmp_path)
    with pytest.raises(ValueError, match="hash-verified"):
        common_validation(prior, current, {"old/moved-image.jpg": "different hash"})


def test_training_exposure_is_rejected(tmp_path):
    prior, current, hashes = cohorts(tmp_path)
    prior["train"] = prior["validation"]
    with pytest.raises(ValueError, match="Training exposure"):
        common_validation(prior, current, hashes)
