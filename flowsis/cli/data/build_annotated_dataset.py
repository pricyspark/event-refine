from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from PIL import Image

from flowsis.data.annotations import (
    CAPTURE_SESSION_FIELD,
    IMAGE_ID_FIELD,
    canonical_tool_label,
    normalize_tool_name,
    parse_bool,
    read_csv,
    validate_visibility,
)
from flowsis.data.masks import load_mask_bundle


def _image_object_identity(
    image: dict[str, str],
    obj: dict[str, str],
    *,
    object_count: int,
    aliases: dict[str, str],
) -> tuple[str, str, str]:
    """Fill missing identity metadata for a single intended object."""
    if (
        object_count == 1
        and image.get("Tool Name", "").strip()
        and not obj.get("Tool ID", "").strip()
        and not obj.get("Tool Class", "").strip()
    ):
        return (
            normalize_tool_name(image["Tool Name"], aliases),
            image.get("Tool Class", ""),
            image.get("Tool ID", ""),
        )
    return (
        normalize_tool_name(obj["Tool Name"], aliases),
        obj.get("Tool Class", ""),
        obj.get("Tool ID", ""),
    )


def _has_ignore(path: Path) -> bool:
    if not path.is_file():
        return False
    ignore = load_mask_bundle(path).ignore
    return ignore is not None and bool(ignore.any())


def _reviewed_keys(path: Path) -> set[str]:
    decisions: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                decisions[str(record["key"])] = str(record["decision"])
    return {key for key, value in decisions.items() if value == "accepted"}


def _reviewed_video_frames(path: Path) -> set[tuple[str, int]]:
    frames = set()
    for key in _reviewed_keys(path):
        if ":" in key:
            video_id, frame_index = key.rsplit(":", 1)
            frames.add((video_id, int(frame_index)))
    return frames


def _group_assignments(
    rows: list[dict[str, Any]],
    *,
    validation_size: float,
    test_size: float,
    seed: int,
) -> list[str]:
    """Keep connected physical-tool/capture-session components in one split."""
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        values = [
            ("tool", str(value)) for value in row.get("tool_ids", []) if str(value)
        ]
        session = str(row.get("capture_session_id", ""))
        if session:
            values.append(("session", session))
        for value in values:
            if value in seen:
                union(index, seen[value])
            else:
                seen[value] = index
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    group_ids = list(groups)
    required_splits = 1 + int(validation_size > 0) + int(test_size > 0)
    if len(group_ids) < required_splits:
        raise ValueError(
            "Too few disconnected tool/session groups for leakage-safe train, "
            "validation, and test splits. Add independent capture sessions or set "
            "an unavailable split size to zero."
        )
    random.Random(seed).shuffle(group_ids)
    targets = {
        "validation": round(len(rows) * validation_size),
        "test": round(len(rows) * test_size),
    }
    assignments = ["train"] * len(rows)
    cursor = 0
    for split in ("validation", "test"):
        assigned = 0
        while cursor < len(group_ids) and assigned < targets[split]:
            indices = groups[group_ids[cursor]]
            cursor += 1
            for index in indices:
                assignments[index] = split
            assigned += len(indices)
    return assignments


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    images = read_csv(args.image_manifest)
    objects = read_csv(args.object_manifest)
    video_objects = (
        read_csv(args.video_object_manifest)
        if args.video_object_manifest.is_file()
        else []
    )
    reviewed = _reviewed_keys(args.review_log)
    if not reviewed:
        raise ValueError("Review log contains no accepted images.")
    reviewed_frames = _reviewed_video_frames(args.video_review_log)
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for obj in objects:
        validate_visibility(obj)
        by_image[obj[IMAGE_ID_FIELD]].append(obj)
    image_by_id = {image[IMAGE_ID_FIELD]: image for image in images}
    aliases_raw = json.loads(args.aliases_json.read_text()) if args.aliases_json else {}
    aliases = {str(key).casefold(): str(value) for key, value in aliases_raw.items()}
    class_names = set()
    for image_id in reviewed:
        image = image_by_id.get(image_id)
        annotated = by_image.get(image_id, [])
        if image is None:
            raise KeyError(f"Review log refers to unknown image {image_id!r}.")
        for obj in annotated:
            if parse_bool(obj.get("Class Valid")) and parse_bool(obj.get("Base Valid")):
                name, tool_class, _ = _image_object_identity(
                    image,
                    obj,
                    object_count=len(annotated),
                    aliases=aliases,
                )
                class_names.add(canonical_tool_label(name, tool_class, aliases))
    class_names.update(
        canonical_tool_label(
            obj.get("Tool Name", ""), obj.get("Tool Class", ""), aliases
        )
        for obj in video_objects
        if (obj.get("Video ID"), int(obj.get("Frame Index", -1))) in reviewed_frames
        and parse_bool(obj.get("Class Valid"))
        and parse_bool(obj.get("Base Valid"))
    )
    names = sorted(class_names)
    if not names:
        raise ValueError("No class-valid object annotations were found.")
    label_by_name = {name: index for index, name in enumerate(names)}
    rows: list[dict[str, Any]] = []
    for image in images:
        image_id = image.get(IMAGE_ID_FIELD, "")
        if image_id not in reviewed:
            continue
        image_path = (args.image_manifest.parent / image["Filepath"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing reviewed image {image_path}.")
        annotated = by_image.get(image_id, [])
        if any(
            not parse_bool(obj.get("Reviewed"))
            or not parse_bool(obj.get("Annotation Complete"))
            for obj in annotated
        ):
            raise ValueError(f"Accepted image {image_id} has incomplete object rows.")
        if any(not parse_bool(obj.get("Base Valid")) for obj in annotated):
            continue
        bundle_path = Path(
            annotated[0]["Mask Path"]
            if annotated
            else args.image_mask_dir / f"{image_id}.npz"
        ).resolve()
        if _has_ignore(bundle_path):
            continue
        with Image.open(image_path) as loaded:
            width, height = loaded.size
        row_objects = []
        for obj in annotated:
            if not parse_bool(obj.get("Mask Valid")):
                continue
            name, tool_class, tool_id = _image_object_identity(
                image,
                obj,
                object_count=len(annotated),
                aliases=aliases,
            )
            label = canonical_tool_label(name, tool_class, aliases)
            if label not in label_by_name:
                continue
            row_objects.append(
                {
                    "instance_id": obj.get("Instance ID", ""),
                    "area": float(obj["BBox Width"]) * float(obj["BBox Height"]),
                    "bbox": [
                        float(obj["BBox X"]),
                        float(obj["BBox Y"]),
                        float(obj["BBox Width"]),
                        float(obj["BBox Height"]),
                    ],
                    "category": label_by_name[label],
                    "tool_name": name,
                    "tool_class": tool_class,
                    "tool_id": tool_id,
                    "mask_path": str(Path(obj["Mask Path"]).resolve()),
                    "mask_index": int(obj["Mask Index"]),
                    "annotation_confidence": float(
                        obj.get("Annotation Confidence", 1.0)
                    ),
                }
            )
        tool_ids = sorted({obj["tool_id"] for obj in row_objects if obj["tool_id"]})
        rows.append(
            {
                "image_path": str(image_path),
                "video_path": "",
                "frame_index": -1,
                "image_id": image_id,
                "height": height,
                "width": width,
                "tool_ids": tool_ids,
                "sample_id": image.get("Sample ID", ""),
                "capture_session_id": image.get(CAPTURE_SESSION_FIELD, ""),
                "objects": row_objects,
            }
        )
    videos = (
        {row["Video ID"]: row for row in read_csv(args.video_manifest)}
        if args.video_manifest.is_file()
        else {}
    )
    video_objects_by_frame: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(
        list
    )
    for obj in video_objects:
        video_objects_by_frame[(obj["Video ID"], int(obj["Frame Index"]))].append(obj)
    for video_id, frame_index in sorted(reviewed_frames):
        video = videos.get(video_id)
        if video is None:
            raise KeyError(f"Reviewed base frame refers to unknown video {video_id!r}.")
        annotated = video_objects_by_frame.get((video_id, frame_index), [])
        if any(
            not parse_bool(obj.get("Reviewed"))
            or not parse_bool(obj.get("Annotation Complete"))
            for obj in annotated
        ):
            raise ValueError(
                f"Accepted video frame {video_id}:{frame_index} has incomplete objects."
            )
        if any(not parse_bool(obj.get("Base Valid")) for obj in annotated):
            continue
        bundle_path = Path(
            annotated[0]["Mask Path"]
            if annotated
            else args.video_mask_dir / video_id / f"{frame_index}.npz"
        ).resolve()
        if _has_ignore(bundle_path):
            continue
        video_path = (args.video_manifest.parent / video["Filepath"]).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing reviewed video {video_path}.")
        row_objects = []
        for obj in annotated:
            if not parse_bool(obj.get("Mask Valid")):
                continue
            name = normalize_tool_name(obj["Tool Name"], aliases)
            tool_class = obj.get("Tool Class", "")
            label = canonical_tool_label(name, tool_class, aliases)
            if label not in label_by_name:
                continue
            row_objects.append(
                {
                    "instance_id": obj.get("Track ID", ""),
                    "area": float(obj["BBox Width"]) * float(obj["BBox Height"]),
                    "bbox": [
                        float(obj["BBox X"]),
                        float(obj["BBox Y"]),
                        float(obj["BBox Width"]),
                        float(obj["BBox Height"]),
                    ],
                    "category": label_by_name[label],
                    "tool_name": name,
                    "tool_class": tool_class,
                    "tool_id": obj.get("Tool ID", ""),
                    "mask_path": str(Path(obj["Mask Path"]).resolve()),
                    "mask_index": int(obj["Mask Index"]),
                    "annotation_confidence": float(
                        obj.get("Annotation Confidence", 1.0)
                    ),
                }
            )
        tool_ids = sorted({obj["tool_id"] for obj in row_objects if obj["tool_id"]})
        rows.append(
            {
                "image_path": "",
                "video_path": str(video_path),
                "frame_index": frame_index,
                "image_id": f"{video_id}:{frame_index}",
                "height": int(video["Height"]),
                "width": int(video["Width"]),
                "tool_ids": tool_ids,
                "sample_id": "",
                "capture_session_id": video.get(CAPTURE_SESSION_FIELD, ""),
                "objects": row_objects,
            }
        )
    if not rows:
        raise ValueError("No reviewed, base-valid dataset rows were created.")
    return rows, names


def build_dataset(args: argparse.Namespace) -> DatasetDict:
    if abs(args.train_size + args.validation_size + args.test_size - 1.0) > 1e-6:
        raise ValueError("Split sizes must sum to one.")
    rows, names = build_rows(args)
    features = Features(
        {
            "image_path": Value("string"),
            "video_path": Value("string"),
            "frame_index": Value("int64"),
            "image_id": Value("string"),
            "height": Value("int64"),
            "width": Value("int64"),
            "tool_ids": Sequence(Value("string")),
            "sample_id": Value("string"),
            "capture_session_id": Value("string"),
            "objects": [
                {
                    "instance_id": Value("string"),
                    "area": Value("float32"),
                    "bbox": Sequence(Value("float32"), length=4),
                    "category": ClassLabel(names=names),
                    "tool_name": Value("string"),
                    "tool_class": Value("string"),
                    "tool_id": Value("string"),
                    "mask_path": Value("string"),
                    "mask_index": Value("int64"),
                    "annotation_confidence": Value("float32"),
                }
            ],
        }
    )
    assignments = _group_assignments(
        rows,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    if any(
        split not in assignments
        for split, size in (
            ("train", args.train_size),
            ("validation", args.validation_size),
            ("test", args.test_size),
        )
        if size > 0
    ):
        raise ValueError("Leakage-safe grouping produced an empty requested split.")
    dataset = Dataset.from_list(rows, features=features)
    split_sizes = {
        "train": args.train_size,
        "validation": args.validation_size,
        "test": args.test_size,
    }
    result = DatasetDict(
        {
            split: dataset.select(
                [
                    index
                    for index, assigned in enumerate(assignments)
                    if assigned == split
                ]
            )
            for split, size in split_sizes.items()
            if size > 0
        }
    )
    if args.output_path.exists():
        shutil.rmtree(args.output_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_to_disk(args.output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reviewed image segmentation Arrow data."
    )
    parser.add_argument(
        "--image-manifest", type=Path, default=Path("data/image_manifest.csv")
    )
    parser.add_argument(
        "--object-manifest",
        type=Path,
        default=Path("data/image_object_manifest.csv"),
    )
    parser.add_argument(
        "--review-log", type=Path, default=Path("data/image_review.jsonl")
    )
    parser.add_argument("--aliases-json", type=Path)
    parser.add_argument(
        "--image-mask-dir", type=Path, default=Path("data/masks/images")
    )
    parser.add_argument(
        "--video-mask-dir", type=Path, default=Path("data/masks/videos")
    )
    parser.add_argument(
        "--video-manifest", type=Path, default=Path("data/video_manifest.csv")
    )
    parser.add_argument(
        "--video-object-manifest",
        type=Path,
        default=Path("data/video_object_manifest.csv"),
    )
    parser.add_argument(
        "--video-review-log", type=Path, default=Path("data/video_review.jsonl")
    )
    parser.add_argument(
        "--output-path", type=Path, default=Path("data/segmentation-dataset")
    )
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()
