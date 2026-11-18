"""Metric logging regression checks, prepared with Codex assistance."""

import warnings

import pytest
import torch

from flowsis.cli.train.train_temporal import _mean, _metric_values


def test_metric_logging_detaches_history_without_changing_backprop():
    parameter = torch.tensor(3.0, requires_grad=True)
    loss = parameter.square()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        logged = _metric_values({"loss": loss, "other": loss.detach() / 3})
    assert logged == {"loss": 9.0, "other": 3.0}
    assert all(type(value) is float for value in logged.values())
    loss.backward()
    assert parameter.grad.item() == 6.0


def test_epoch_mean_uses_detached_scalar_history():
    assert _mean([]) == {}
    history = [
        _metric_values({"loss": torch.tensor(2.0, requires_grad=True)}),
        _metric_values({"loss": torch.tensor(6.0, requires_grad=True)}),
    ]
    assert _mean(history)["loss"] == pytest.approx(4.0)
