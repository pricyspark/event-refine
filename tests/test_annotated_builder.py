import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from flowsis.cli.data.build_annotated_dataset import build_dataset
from flowsis.data.masks import save_mask_bundle


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_image_builder_preserves_hierarchy_and_external_mask_indices(
    tmp_path: Path,
) -> None:
    image_manifest = tmp_path / "images.csv"
    object_manifest = tmp_path / "objects.csv"
    review_log = tmp_path / "review.jsonl"
    object_fields = [
        "Image ID",
        "Instance ID",
        "Tool ID",
        "Tool Class",
        "Tool Name",
        "BBox X",
        "BBox Y",
        "BBox Width",
        "BBox Height",
        "Mask Path",
        "Mask Index",
        "Visibility",
        "Reviewed",
        "Annotation Complete",
        "Class Valid",
        "Base Valid",
        "Mask Valid",
        "Visible",
        "Annotation Confidence",
    ]
    image_rows = []
    object_rows = []
    decisions = []
    for index in range(3):
        image_id = f"image-{index}"
        image_path = tmp_path / f"{index}.png"
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
        mask = np.zeros((1, 8, 8), dtype=np.bool_)
        mask[0, 2:6, 2:6] = True
        mask_path = tmp_path / f"{index}.npz"
        save_mask_bundle(mask_path, mask, [f"instance-{index}"])
        image_rows.append(
            {
                "Filepath": image_path.name,
                "Tool ID": str(index),
                "Sample ID": "state",
                "Image ID": image_id,
                "Capture Session ID": f"session-{index}",
            }
        )
        object_rows.append(
            {
                "Image ID": image_id,
                "Instance ID": f"instance-{index}",
                "Tool ID": str(index),
                "Tool Class": "grasper",
                "Tool Name": "forceps",
                "BBox X": 2,
                "BBox Y": 2,
                "BBox Width": 4,
                "BBox Height": 4,
                "Mask Path": str(mask_path),
                "Mask Index": 0,
                "Visibility": "clear",
                "Reviewed": "TRUE",
                "Annotation Complete": "TRUE",
                "Class Valid": "TRUE",
                "Base Valid": "TRUE",
                "Mask Valid": "TRUE",
                "Visible": "TRUE",
                "Annotation Confidence": 1,
            }
        )
        decisions.append(
            json.dumps({"key": image_id, "decision": "accepted", "prompt": "forceps"})
        )
    _write_csv(
        image_manifest,
        [
            "Filepath",
            "Tool ID",
            "Sample ID",
            "Image ID",
            "Capture Session ID",
        ],
        image_rows,
    )
    _write_csv(object_manifest, object_fields, object_rows)
    review_log.write_text("\n".join(decisions) + "\n", encoding="utf-8")
    output = tmp_path / "arrow"

    dataset = build_dataset(
        SimpleNamespace(
            image_manifest=image_manifest,
            object_manifest=object_manifest,
            review_log=review_log,
            aliases_json=None,
            image_mask_dir=tmp_path,
            video_mask_dir=tmp_path,
            video_manifest=tmp_path / "missing-videos.csv",
            video_object_manifest=tmp_path / "missing-video-objects.csv",
            video_review_log=tmp_path / "missing-video-review.jsonl",
            output_path=output,
            train_size=1 / 3,
            validation_size=1 / 3,
            test_size=1 / 3,
            seed=1,
        )
    )

    rows = [row for split in dataset.values() for row in split]
    assert len(rows) == 3
    assert rows[0]["objects"][0]["tool_class"] == "grasper"
    assert rows[0]["objects"][0]["mask_index"] == 0
