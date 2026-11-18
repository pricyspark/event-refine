"""Dynamic sampling weights for shared still-image and video-frame datasets."""

import math
from collections import Counter

from datasets import Dataset


def frame_sampling_weights(
    dataset: Dataset, *, video_fraction: float = 0.5
) -> list[float]:
    """Balance sources, then capture groups/videos, then frames within each group.

    With one source, give its groups equal weight regardless of video_fraction.
    Legacy still datasets retain their sample-group weighting.
    """
    if not math.isfinite(video_fraction) or not 0 <= video_fraction <= 1:
        raise ValueError("video_fraction must be finite and between zero and one.")
    count = len(dataset)
    if not count:
        raise ValueError("Cannot sample an empty base-training view.")
    columns = set(dataset.column_names)
    videos = dataset["video_id"] if "video_id" in columns else [""] * count
    samples = dataset["sample_id"] if "sample_id" in columns else [""] * count
    keys = [
        (
            ("video", str(video))
            if video
            else (("sample", str(sample)) if sample else ("image", str(index)))
        )
        for index, (video, sample) in enumerate(zip(videos, samples, strict=True))
    ]
    group_sizes = Counter(keys)
    group_counts = Counter(
        "video" if key[0] == "video" else "image" for key in group_sizes
    )
    mixed = len(group_counts) == 2
    if not mixed:
        return [1.0 / group_sizes[key] for key in keys]
    return [
        (video_fraction if key[0] == "video" else 1 - video_fraction)
        / group_counts["video" if key[0] == "video" else "image"]
        / group_sizes[key]
        for key in keys
    ]
