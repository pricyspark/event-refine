from pathlib import Path

import pytest
import torch

from flowsis.data.prompts import LabelPrompts, load_prompt_bank


def test_load_embeddings_moves_cached_tensor_to_requested_device(
    tmp_path: Path,
) -> None:
    torch.save(torch.ones(2, 3), tmp_path / "object.pt")
    cache: dict[str, torch.Tensor] = {}

    cpu_embeddings = LabelPrompts.load_embeddings(
        "object",
        tmp_path,
        cache,
        device=torch.device("cpu"),
    )
    meta_embeddings = LabelPrompts.load_embeddings(
        "object",
        tmp_path,
        cache,
        device=torch.device("meta"),
    )

    assert cpu_embeddings.device.type == "cpu"
    assert meta_embeddings.device.type == "meta"
    assert cache["object"] is meta_embeddings


def test_prompt_bank_preserves_label_order_and_requires_rectangular_shape(
    tmp_path: Path,
) -> None:
    torch.save(torch.full((2, 768), 1.0), tmp_path / "first.pt")
    torch.save(torch.full((2, 768), 2.0), tmp_path / "second.pt")
    bank = load_prompt_bank(("second", "first"), tmp_path)
    assert bank.shape == (2, 2, 768)
    assert torch.equal(bank[0], torch.full((2, 768), 2.0))

    torch.save(torch.ones(1, 768), tmp_path / "second.pt")
    with pytest.raises(ValueError, match="same prompt count"):
        load_prompt_bank(("first", "second"), tmp_path)
