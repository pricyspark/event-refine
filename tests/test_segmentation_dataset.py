from datasets import Dataset

from flowsis.cli.data.build_hf_segmentation_dataset import source_grouped_splits


def test_source_grouped_splits_do_not_leak_videos() -> None:
    dataset = Dataset.from_dict(
        {
            "value": list(range(8)),
            "source_video_ids": [[0], [0, 1], [1], [2], [3], [4], [5], [6]],
        }
    )
    splits = source_grouped_splits(dataset, validation_size=0.2, test_size=0.2, seed=4)
    groups = {
        name: {
            video for source_ids in split["source_video_ids"] for video in source_ids
        }
        for name, split in splits.items()
    }
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    assert sum(len(split) for split in splits.values()) == len(dataset)
