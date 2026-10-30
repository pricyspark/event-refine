from pathlib import Path

import pytest
import torch

from flowsis.data.flow_cache import FrozenFlowCache


class DifferenceFlow(torch.nn.Module):
    def __init__(self, scale: float = 1.0, offset: float = 0.0) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale), requires_grad=False)
        self.hparams = {"offset": offset}
        self.calls = 0
        self.batch_sizes: list[int] = []

    def forward(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.batch_sizes.append(len(current))
        return (
            (current[:, :2] - previous[:, :2]) * self.scale + self.hparams["offset"]
        )


def test_flow_cache_persists_deduplicates_and_batches_misses(tmp_path: Path) -> None:
    estimator = DifferenceFlow().eval()
    current = torch.rand(3, 3, 4, 5)
    previous = torch.rand_like(current)
    current = torch.cat((current, current[:1]))
    previous = torch.cat((previous, previous[:1]))
    cache = FrozenFlowCache(tmp_path, estimator, batch_size=2)

    actual = cache(current, previous)

    torch.testing.assert_close(actual, current[:, :2] - previous[:, :2])
    assert estimator.batch_sizes == [2, 1]
    assert len(list(tmp_path.rglob("*.pt"))) == 3
    assert not actual.requires_grad

    # Recreate both objects to exercise reuse across training invocations.
    restarted_estimator = DifferenceFlow().eval()
    restarted_cache = FrozenFlowCache(tmp_path, restarted_estimator)
    torch.testing.assert_close(restarted_cache(current, previous), actual)
    assert restarted_estimator.calls == 0


def test_flow_cache_distinguishes_orientation_pair_order_and_weights(
    tmp_path: Path,
) -> None:
    estimator = DifferenceFlow().eval()
    cache = FrozenFlowCache(tmp_path, estimator)
    current = torch.rand(1, 3, 4, 5)
    previous = torch.rand_like(current)
    original = cache(current, previous)

    torch.testing.assert_close(cache(previous, current), -original)
    torch.testing.assert_close(
        cache(current.flip(-1), previous.flip(-1)), original.flip(-1)
    )
    # A different previous image represents a different stride/pair.
    cache(current, previous + 0.1)
    assert estimator.calls == 4

    changed_estimator = DifferenceFlow(scale=2.0).eval()
    changed_cache = FrozenFlowCache(tmp_path, changed_estimator)
    assert changed_cache.directory != cache.directory
    torch.testing.assert_close(changed_cache(current, previous), original * 2)
    assert changed_estimator.calls == 1

    configured_estimator = DifferenceFlow(offset=0.5).eval()
    configured_cache = FrozenFlowCache(tmp_path, configured_estimator)
    assert configured_cache.directory != cache.directory
    torch.testing.assert_close(configured_cache(current, previous), original + 0.5)


def test_flow_cache_rejects_trainable_estimators_and_invalid_entries(
    tmp_path: Path,
) -> None:
    estimator = DifferenceFlow().eval()
    estimator.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        FrozenFlowCache(tmp_path, estimator)
    estimator.requires_grad_(False).train()
    with pytest.raises(ValueError, match="eval mode"):
        FrozenFlowCache(tmp_path, estimator)

    estimator.eval()
    cache = FrozenFlowCache(tmp_path, estimator)
    current = torch.rand(1, 3, 4, 5)
    previous = torch.rand_like(current)
    cache(current, previous)
    path = next(tmp_path.rglob("*.pt"))
    torch.save(torch.zeros(1), path)
    with pytest.raises(ValueError, match="Invalid cached flow"):
        cache(current, previous)


def test_flow_cache_separates_autocast_execution(tmp_path: Path) -> None:
    estimator = DifferenceFlow().eval()
    cache = FrozenFlowCache(tmp_path, estimator)
    current = torch.rand(1, 3, 4, 5)
    previous = torch.rand_like(current)
    expected = cache(current, previous)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(cache(current, previous), expected)
        cache(current, previous)
    cache(current, previous)

    assert estimator.calls == 2
    assert len(list(tmp_path.rglob("*.pt"))) == 2
