from pathlib import Path
from PIL import Image
from typing import Any

from ..data.masks import load_mask
from ..data.object_records import get_object_records


def relocate_segmentation_paths(
    example: dict[str, Any],
    *,
    images_dir: str | Path = "data/frames",
    masks_dir: str | Path = "data/masks",
    **_: Any,
) -> dict[str, Any]:
    """Repair serialized paths after a segmentation dataset moves machines."""
    objects = get_object_records(example)
    source_video_ids = [int(value) for value in example.get("source_video_ids", [])]
    if objects:
        image_video_id = int(objects[0]["video_id"])
    elif source_video_ids:
        image_video_id = source_video_ids[0]
    else:
        image_video_id = None

    stored_image_path = Path(example["image_path"])
    if not stored_image_path.is_file():
        if image_video_id is None:
            raise FileNotFoundError(
                f"Stored image path does not exist: {stored_image_path}. The row "
                "has no object/source video metadata for relocation."
            )
        relocated_image = (
            Path(images_dir) / str(image_video_id) / stored_image_path.name
        )
        if not relocated_image.is_file():
            raise FileNotFoundError(
                f"Neither stored nor relocated image exists: {stored_image_path} / "
                f"{relocated_image}. Set --images-dir to the local frame root."
            )
        example["image_path"] = str(relocated_image)

    for obj in objects:
        stored_mask_path = Path(obj["mask_path"])
        if stored_mask_path.is_file():
            continue
        relocated_mask = (
            Path(masks_dir) / str(int(obj["video_id"])) / f"{int(obj['frame_idx'])}.npz"
        )
        if not relocated_mask.is_file():
            raise FileNotFoundError(
                f"Neither stored nor relocated mask exists: {stored_mask_path} / "
                f"{relocated_mask}. Set --masks-dir to the local mask root."
            )
        obj["mask_path"] = str(relocated_mask)
    example["objects"] = objects
    return example


def load_object_image(example: dict[str, Any], **kwargs) -> dict:
    if "image" in example:
        return example

    if example["modified"]:
        raise ValueError("Cannot load an image for a previously modified sample.")

    image_path = example["image_path"]
    example["image"] = Image.open(image_path)

    return example


def load_object_masks(example: dict[str, Any], **kwargs) -> dict:
    if "mask_dir" in kwargs:
        mask_dir = Path(kwargs["mask_dir"])
    else:
        mask_dir = None

    for obj in example["objects"]:
        if "mask" in obj:
            continue

        if obj["modified"]:
            raise ValueError("Cannot load a mask for a previously modified object.")

        video_id = obj["video_id"]
        frame_idx = obj["frame_idx"]

        obj["mask"] = load_mask(video_id, frame_idx, mask_dir)

    return example
