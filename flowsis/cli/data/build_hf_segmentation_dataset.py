import csv
import shutil
import argparse
import random
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

from flowsis.data.masks import mask2xywh, load_mask
from flowsis.utils import load_classes

DEFAULT_MANIFEST_PATH = Path("data/manifests/frame_manifest.csv")
DEFAULT_MASKS_DIR = Path("data/masks")
DEFAULT_CLASSES_PATH = Path("data/manifests/classes.json")
DEFAULT_OUTPUT_PATH = Path("data/segmentation-dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Hugging Face segmentation dataset used by FlowSIS.",
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--masks-dir", type=Path, default=DEFAULT_MASKS_DIR)
    parser.add_argument("--classes-path", type=Path, default=DEFAULT_CLASSES_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument(
        "--include-missing-masks-as-empty",
        action="store_true",
        help=(
            "Include manifest frames without mask files as empty images. Only enable "
            "this when missing masks are authoritative negative annotations."
        ),
    )
    parser.add_argument(
        "--group-splits-by-video",
        action="store_true",
        help="Keep every source-video group in exactly one random split.",
    )
    return parser.parse_args()


def normalize_path(path: str | Path) -> str:
    return str(Path(path).expanduser())


def validate_split_sizes(
    train_size: float, validation_size: float, test_size: float
) -> None:
    total = train_size + validation_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "Split sizes must sum to 1.0, "
            f"but received train={train_size}, validation={validation_size}, test={test_size}."
        )


def source_grouped_splits(
    dataset: Dataset,
    *,
    validation_size: float,
    test_size: float,
    seed: int = 42,
) -> DatasetDict:
    row_sources = [
        tuple(sorted(set(int(value) for value in source_ids)))
        for source_ids in dataset["source_video_ids"]
    ]
    parent: dict[int, int] = {
        source_id: source_id for source_ids in row_sources for source_id in source_ids
    }

    def find(source_id: int) -> int:
        while parent[source_id] != source_id:
            parent[source_id] = parent[parent[source_id]]
            source_id = parent[source_id]
        return source_id

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for source_ids in row_sources:
        for source_id in source_ids[1:]:
            union(source_ids[0], source_id)

    grouped_indices: dict[int, list[int]] = {}
    empty_group = min(parent, default=0) - 1
    for index, source_ids in enumerate(row_sources):
        group = find(source_ids[0]) if source_ids else empty_group - index
        grouped_indices.setdefault(group, []).append(index)
    groups = list(grouped_indices)
    random.Random(seed).shuffle(groups)
    num_groups = len(groups)
    validation_groups = round(num_groups * validation_size)
    test_groups = round(num_groups * test_size)
    if validation_size > 0.0 and validation_groups == 0 and num_groups >= 3:
        validation_groups = 1
    if test_size > 0.0 and test_groups == 0 and num_groups >= 3:
        test_groups = 1
    while validation_groups + test_groups >= num_groups:
        if test_groups >= validation_groups and test_groups > 0:
            test_groups -= 1
        elif validation_groups > 0:
            validation_groups -= 1
    split_groups = {
        "validation": groups[:validation_groups],
        "test": groups[validation_groups : validation_groups + test_groups],
        "train": groups[validation_groups + test_groups :],
    }
    return DatasetDict(
        {
            name: dataset.select(
                [index for group in selected for index in grouped_indices[group]]
            )
            for name, selected in split_groups.items()
        }
    )


def build_dataset(
    output_path: Path,
    classes: tuple[
        dict[str, str],
        dict[str, int],
        dict[int, str],
    ],
    *,
    manifest_path: Path,
    masks_dir: Path,
    train_size: float,
    validation_size: float,
    test_size: float,
    include_missing_masks_as_empty: bool = False,
    group_splits_by_video: bool = False,
) -> None:
    validate_split_sizes(train_size, validation_size, test_size)

    rows_by_image: dict[str, dict[str, object]] = {}
    missing_mask_examples: list[dict[str, int]] = []
    vid2label, label2id, id2label = classes

    with manifest_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            video_id = row["video_id"]
            frame_idx = row["frame_idx"]
            image_id = row["id"]
            image_path = normalize_path(row["output_path"])

            mask_path = masks_dir / video_id / f"{frame_idx}.npz"
            if video_id in vid2label and not mask_path.exists():
                missing_mask_examples.append(
                    {
                        "video_id": int(video_id),
                        "frame_idx": int(frame_idx),
                        "image_id": int(image_id),
                    }
                )
            if (
                video_id not in vid2label or not mask_path.exists()
            ) and not include_missing_masks_as_empty:
                continue

            image_row = rows_by_image.setdefault(
                image_path,
                {
                    "image_path": image_path,
                    "image_id": int(image_id),
                    "height": int(row["height"]),
                    "width": int(row["width"]),
                    "modified": False,
                    "source_video_ids": [],
                    "objects": [],
                },
            )
            source_video_ids = image_row["source_video_ids"]
            assert isinstance(source_video_ids, list)
            source_video_ids.append(int(video_id))

            if video_id not in vid2label:
                continue

            if not mask_path.exists():
                continue

            label = vid2label[video_id]
            category_id = label2id[label]

            mask = load_mask(int(video_id), int(frame_idx), masks_dir)
            bbox = mask2xywh(mask)
            if bbox is None:
                continue

            objects = image_row["objects"]
            assert isinstance(objects, list)
            objects.append(
                {
                    "id": int(image_id),
                    "area": float(bbox[2] * bbox[3]),
                    "bbox": [float(value) for value in bbox],
                    "category": category_id,
                    "video_id": int(video_id),
                    "frame_idx": int(frame_idx),
                    "mask_path": normalize_path(mask_path),
                    "modified": False,
                }
            )

    rows = list(rows_by_image.values())

    if not rows:
        raise ValueError(
            "No segmentation dataset rows were created. Check the manifest, masks, "
            "and class metadata."
        )

    category_names = [id2label[idx] for idx in range(len(id2label))]
    features = Features(
        {
            "image_path": Value("string"),
            "image_id": Value("int64"),
            "height": Value("int64"),
            "width": Value("int64"),
            "modified": Value("bool"),
            "source_video_ids": [Value("int64")],
            "objects": [
                {
                    "id": Value("int64"),
                    "area": Value("float32"),
                    "bbox": Sequence(Value("float32"), length=4),
                    "category": ClassLabel(names=category_names),
                    "video_id": Value("int64"),
                    "frame_idx": Value("int64"),
                    "mask_path": Value("string"),
                    "modified": Value("bool"),
                }
            ],
        }
    )

    dataset = Dataset.from_list(rows, features=features)
    if group_splits_by_video:
        dataset_splits = source_grouped_splits(
            dataset,
            validation_size=validation_size,
            test_size=test_size,
        )
    else:
        held_out_size = validation_size + test_size
        split = dataset.train_test_split(test_size=held_out_size, seed=42)
        validation_ratio = validation_size / held_out_size if held_out_size > 0 else 0.5
        held_out = split["test"].train_test_split(
            test_size=1.0 - validation_ratio, seed=42
        )
        dataset_splits = DatasetDict(
            {
                "train": split["train"],
                "validation": held_out["train"],
                "test": held_out["test"],
            }
        )

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_splits.save_to_disk(output_path)

    if missing_mask_examples:
        print(
            "build_dataset_warning",
            {
                "missing_mask_examples": missing_mask_examples[:10],
                "num_missing_mask_examples": len(missing_mask_examples),
            },
        )
    print(
        "build_dataset",
        {
            "output_path": str(output_path),
            "num_rows": len(dataset),
            "num_train": len(dataset_splits["train"]),
            "num_validation": len(dataset_splits["validation"]),
            "num_test": len(dataset_splits["test"]),
            "notes": [
                "Rows are grouped by image path and may contain zero or more objects.",
                "Segmentation labels, boxes, and visible masks are object-level fields.",
                "Overlap compositing is applied only by the training dataset pipeline.",
                "Prompt banks are loaded once by flowsis-train-base, not stored per object.",
            ],
        },
    )


def main() -> None:
    args = parse_args()
    classes = load_classes(args.classes_path)
    build_dataset(
        args.output_path,
        classes,
        manifest_path=args.manifest_path,
        masks_dir=args.masks_dir,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        include_missing_masks_as_empty=args.include_missing_masks_as_empty,
        group_splits_by_video=args.group_splits_by_video,
    )


if __name__ == "__main__":
    main()
