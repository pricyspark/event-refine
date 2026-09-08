import numpy as np

from flowsis.data.augment.overlap import _verify_overlap_example


def test_overlap_keeps_disconnected_visible_fragments_and_records_diagnostics() -> None:
    full = np.zeros((1, 8, 8), dtype=np.bool_)
    full[0, 1:7, 1:7] = True
    visible = np.zeros_like(full)
    visible[0, 1:3, 1:3] = True
    visible[0, 5:7, 5:7] = True
    example = {"objects": [{"mask": full[0], "bbox": [1, 1, 6, 6]}]}

    assert _verify_overlap_example(
        example, min_visible_pixels=2, masks=full, visible_masks=visible
    )

    assert len(example["objects"]) == 1
    assert example["objects"][0]["visible_components"] == 2
    assert example["objects"][0]["visible_pixels"] == 8
    np.testing.assert_array_equal(example["objects"][0]["mask"], visible[0])
