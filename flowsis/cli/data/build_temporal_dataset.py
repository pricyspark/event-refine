from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

from flowsis.data.annotations import (
    CAPTURE_SESSION_FIELD,
    canonical_tool_label,
    normalize_tool_name,
    parse_bool,
    read_csv,
    validate_visibility,
)


def _accepted_frames(path: Path) -> set[tuple[str, int]]:
    latest: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                latest[str(record["key"])] = str(record["decision"])
    accepted = set()
    for key, decision in latest.items():
        if decision != "accepted" or ":" not in key:
            continue
        video_id, frame_index = key.rsplit(":", 1)
        accepted.add((video_id, int(frame_index)))
    return accepted


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    videos = {row["Video ID"]: row for row in read_csv(args.video_manifest)}
    objects = read_csv(args.object_manifest) if args.object_manifest.is_file() else []
    accepted = _accepted_frames(args.review_log)
    if not accepted:
        raise ValueError("Video review log contains no accepted frames.")
    aliases_raw = json.loads(args.aliases_json.read_text()) if args.aliases_json else {}
    aliases = {str(key).casefold(): str(value) for key, value in aliases_raw.items()}
    names = sorted(
        {
            canonical_tool_label(
                obj.get("Tool Name", ""), obj.get("Tool Class", ""), aliases
            )
            for obj in objects
            if obj.get("Tool Name")
            and (obj["Video ID"], int(obj["Frame Index"])) in accepted
        }
    )
    if not names:
        raise ValueError("Video annotations contain no canonical tool names.")
    category = {name: index for index, name in enumerate(names)}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for obj in objects:
        validate_visibility(obj)
        key = (obj["Video ID"], int(obj["Frame Index"]))
        if key not in accepted:
            continue
        if not parse_bool(obj.get("Reviewed")) or not parse_bool(
            obj.get("Annotation Complete")
        ):
            raise ValueError(f"Accepted frame {key} contains an incomplete object row.")
        name = normalize_tool_name(obj.get("Tool Name", ""), aliases)
        label = canonical_tool_label(name, obj.get("Tool Class", ""), aliases)
        grouped[key].append(
            {
                "track_id": obj["Track ID"],
                "physical_tool_id": obj.get("Tool ID", ""),
                "tool_class": obj.get("Tool Class", ""),
                "tool_name": name,
                "category": category[label],
                "bbox": [
                    float(obj["BBox X"]),
                    float(obj["BBox Y"]),
                    float(obj["BBox Width"]),
                    float(obj["BBox Height"]),
                ],
                "mask_path": (
                    str(Path(obj["Mask Path"]).resolve())
                    if obj.get("Mask Path")
                    else ""
                ),
                "mask_index": int(obj.get("Mask Index", -1)),
                "visibility": obj["Visibility"],
                "mask_valid": parse_bool(obj.get("Mask Valid")),
                "class_valid": parse_bool(obj.get("Class Valid")),
                "base_valid": parse_bool(obj.get("Base Valid")),
                "visible": parse_bool(obj.get("Visible")),
                "annotation_confidence": float(obj.get("Annotation Confidence", 1.0)),
                "preferred_track": parse_bool(obj.get("Preferred Track")),
            }
        )

    rows = []
    for video_id, frame_index in sorted(accepted):
        if video_id not in videos:
            raise KeyError(f"Reviewed frame refers to unknown video {video_id!r}.")
        video = videos[video_id]
        path = (args.video_manifest.parent / video["Filepath"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing reviewed video: {path}")
        rows.append(
            {
                "video_id": video_id,
                "video_path": str(path),
                "frame_index": frame_index,
                "fps": float(video["FPS"]),
                "height": int(video["Height"]),
                "width": int(video["Width"]),
                "capture_session_id": video.get(CAPTURE_SESSION_FIELD, ""),
                "annotation_complete": True,
                "mask_bundle_path": str(
                    Path(
                        next(
                            (
                                obj["mask_path"]
                                for obj in grouped.get((video_id, frame_index), [])
                                if obj["mask_path"]
                            ),
                            args.mask_dir / video_id / f"{frame_index}.npz",
                        )
                    ).resolve()
                ),
                "objects": grouped.get((video_id, frame_index), []),
            }
        )
    return rows, names


def _split_videos(
    rows: list[dict[str, Any]], validation_size: float, test_size: float, seed: int
) -> dict[str, str]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    parent = {video_id: video_id for video_id in by_video}

    def find(video_id: str) -> str:
        while parent[video_id] != video_id:
            parent[video_id] = parent[parent[video_id]]
            video_id = parent[video_id]
        return video_id

    def union(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    seen: dict[tuple[str, str], str] = {}
    for video_id, video_rows in by_video.items():
        values = {
            ("session", str(video_rows[0]["capture_session_id"])),
            *{
                ("tool", str(obj["physical_tool_id"]))
                for row in video_rows
                for obj in row["objects"]
                if obj["physical_tool_id"]
            },
        }
        for value in values:
            if not value[1]:
                continue
            if value in seen:
                union(video_id, seen[value])
            else:
                seen[value] = video_id
    components: dict[str, list[str]] = defaultdict(list)
    for video_id in by_video:
        components[find(video_id)].append(video_id)
    component_ids = list(components)
    required_splits = 1 + int(validation_size > 0) + int(test_size > 0)
    if len(component_ids) < required_splits:
        raise ValueError(
            "Too few disconnected video/session/tool groups for leakage-safe "
            "splits. Add independent capture sessions or set an unavailable split "
            "size to zero."
        )
    random.Random(seed).shuffle(component_ids)
    frame_total = len(rows)
    targets = {
        "validation": round(frame_total * validation_size),
        "test": round(frame_total * test_size),
    }
    result = {video_id: "train" for video_id in by_video}
    cursor = 0
    for split in ("validation", "test"):
        count = 0
        while cursor < len(component_ids) and count < targets[split]:
            videos = components[component_ids[cursor]]
            cursor += 1
            for video_id in videos:
                result[video_id] = split
                count += len(by_video[video_id])
    return result


def build_dataset(args: argparse.Namespace) -> DatasetDict:
    if abs(args.train_size + args.validation_size + args.test_size - 1.0) > 1e-6:
        raise ValueError("Split sizes must sum to one.")
    rows, names = build_rows(args)
    features = Features(
        {
            "video_id": Value("string"),
            "video_path": Value("string"),
            "frame_index": Value("int64"),
            "fps": Value("float32"),
            "height": Value("int64"),
            "width": Value("int64"),
            "capture_session_id": Value("string"),
            "annotation_complete": Value("bool"),
            "mask_bundle_path": Value("string"),
            "objects": [
                {
                    "track_id": Value("string"),
                    "physical_tool_id": Value("string"),
                    "tool_class": Value("string"),
                    "tool_name": Value("string"),
                    "category": ClassLabel(names=names),
                    "bbox": Sequence(Value("float32"), length=4),
                    "mask_path": Value("string"),
                    "mask_index": Value("int64"),
                    "visibility": Value("string"),
                    "mask_valid": Value("bool"),
                    "class_valid": Value("bool"),
                    "base_valid": Value("bool"),
                    "visible": Value("bool"),
                    "annotation_confidence": Value("float32"),
                    "preferred_track": Value("bool"),
                }
            ],
        }
    )
    assignments = _split_videos(rows, args.validation_size, args.test_size, args.seed)
    if any(
        split not in assignments.values()
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
                    for index, row in enumerate(rows)
                    if assignments[row["video_id"]] == split
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
    parser = argparse.ArgumentParser(description="Build reviewed temporal Arrow data.")
    parser.add_argument(
        "--video-manifest", type=Path, default=Path("data/new/video_manifest.csv")
    )
    parser.add_argument(
        "--object-manifest",
        type=Path,
        default=Path("data/new/video_object_manifest.csv"),
    )
    parser.add_argument(
        "--review-log", type=Path, default=Path("data/new/video_review.jsonl")
    )
    parser.add_argument("--aliases-json", type=Path)
    parser.add_argument("--mask-dir", type=Path, default=Path("data/new/masks/videos"))
    parser.add_argument(
        "--output-path", type=Path, default=Path("data/new/temporal-dataset")
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
