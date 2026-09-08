import hashlib
from typing import Any


def get_object_records(example: dict[str, Any]) -> list[dict[str, Any]]:
    objects = example.get("objects")
    if objects is None:
        return []
    if not isinstance(objects, list):
        raise TypeError(
            f"Expected example['objects'] to be a list, received {type(objects).__name__}."
        )
    return [dict(record) for record in objects]


def get_object_feature_schema(objects_feature: Any) -> Any:
    schema = getattr(objects_feature, "feature", objects_feature)
    if isinstance(schema, list):
        if len(schema) != 1:
            raise TypeError(
                f"Expected a single object feature schema, received {len(schema)} entries."
            )
        return schema[0]
    return schema


def numeric_image_id(value: object, fallback: int = 0) -> int:
    """Map stable string image IDs to the integer IDs expected by DETR losses."""
    if value is None or value == "":
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        digest = hashlib.sha256(str(value).encode()).digest()[:8]
        return int.from_bytes(digest, "big") & ((1 << 63) - 1)
