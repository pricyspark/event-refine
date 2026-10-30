import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image
from typing import Any

from ..data.masks import load_mask_bundle


def load_object_image(example: dict[str, Any], **kwargs) -> dict:
    example.setdefault("modified", False)
    if "image" in example:
        return example

    if example["modified"]:
        raise ValueError("Cannot load an image for a previously modified sample.")

    if example.get("video_path") and int(example.get("frame_index", -1)) >= 0:
        import cv2

        capture = cv2.VideoCapture(str(example["video_path"]))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video {example['video_path']}.")
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(example["frame_index"]))
            ok, bgr = capture.read()
        finally:
            capture.release()
        if not ok:
            raise RuntimeError(
                f"Could not decode frame {example['frame_index']} from "
                f"{example['video_path']}."
            )
        example["image"] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    else:
        example["image"] = Image.open(example["image_path"])

    return example


def load_object_masks(example: dict[str, Any], **_: Any) -> dict[str, Any]:
    example.setdefault("modified", False)
    bundles = {}
    for obj in example["objects"]:
        obj.setdefault("modified", False)
        if "image_id" in example:
            obj.setdefault("source_image_id", example["image_id"])
        if "mask" in obj:
            continue

        if obj["modified"]:
            raise ValueError("Cannot load a mask for a previously modified object.")

        bundle_path = obj["mask_path"]
        if bundle_path not in bundles:
            bundles[bundle_path] = load_mask_bundle(bundle_path)
        bundle = bundles[bundle_path]
        mask_index = int(obj["mask_index"])
        if not 0 <= mask_index < len(bundle.masks):
            raise IndexError(
                f"mask_index {mask_index} is outside bundle {bundle_path} with "
                f"{len(bundle.masks)} masks."
            )
        obj["mask"] = bundle.masks[mask_index]

    return example


def load_local_dataset(path: str | Path) -> Dataset | DatasetDict:
    """Resolve versioned local asset paths in memory, leaving stored rows portable.

    Legacy datasets without the descriptor keep their existing path semantics.
    """
    root = Path(path).expanduser().resolve()
    loaded = load_from_disk(str(root))
    descriptor = root / "asset_paths.json"
    if not descriptor.is_file():
        return loaded
    contract = json.loads(descriptor.read_text())
    if contract != {"version": 1, "relative_to": "dataset_directory"}:
        raise ValueError(f"Unsupported asset path contract: {contract}")

    def resolve_paths(example: dict[str, Any]) -> dict[str, Any]:
        for field in ("image_path", "video_path"):
            if example.get(field):
                example[field] = str((root / example[field]).resolve())
        for obj in example.get("objects", []):
            if obj.get("mask_path"):
                obj["mask_path"] = str((root / obj["mask_path"]).resolve())
        return example

    return loaded.map(
        resolve_paths, keep_in_memory=True, desc="Resolving local asset paths"
    )
