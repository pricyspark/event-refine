"""Lossless event storage, legacy loading, and installer flag coverage."""

import argparse
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from event_flow.cli import main, prepare_events
from event_flow.download import download_fred
from event_flow.events import load_event_map, save_event_map


@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (8, 8), (33, 65)])
@pytest.mark.parametrize("fill", ["empty", "full", "random"])
def test_compressed_event_map_preserves_every_bit(tmp_path, shape, fill):
    rng = np.random.default_rng(42)
    array = (
        rng.random(shape) < 0.2
        if fill == "random"
        else np.full(shape, fill == "full", dtype=bool)
    )
    original = array.copy()
    compressed = save_event_map(tmp_path / "events.npz", array)
    legacy = save_event_map(tmp_path / "events.npy", array)
    expected = torch.from_numpy(array.copy()).float()[None]
    torch.testing.assert_close(load_event_map(compressed), expected)
    torch.testing.assert_close(load_event_map(legacy), expected)
    np.testing.assert_array_equal(array, original)
    assert not list(tmp_path.glob(".*"))


@pytest.mark.parametrize(
    "bad", ["version", "shape", "payload", "dtype", "bitorder", "metadata"]
)
def test_invalid_packed_archive_is_rejected(tmp_path, bad):
    arrays = {
        "format_version": np.asarray(1),
        "shape": np.array([3, 5]),
        "bitorder": np.asarray("big"),
        "packed": np.array([0, 0], dtype=np.uint8),
    }
    if bad == "version":
        arrays["format_version"] = np.asarray(2)
    elif bad == "shape":
        arrays["shape"] = np.array([0, 5])
    elif bad == "payload":
        arrays["packed"] = np.array([0], dtype=np.uint8)
    elif bad == "dtype":
        arrays["packed"] = arrays["packed"].astype(np.int64)
    elif bad == "bitorder":
        arrays["bitorder"] = np.asarray("invalid")
    else:
        del arrays["shape"]
    path = tmp_path / "invalid.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        load_event_map(path)


def test_writer_rejects_non_boolean_maps(tmp_path):
    with pytest.raises(ValueError, match="boolean"):
        save_event_map(tmp_path / "events.npz", np.ones((3, 4), dtype=np.float32))
    assert not (tmp_path / "events.npz").exists()


@pytest.mark.parametrize("compress", [True, False])
def test_raw_event_preparation_obeys_storage_flag(tmp_path, compress):
    np.savez(
        tmp_path / "source.npz", x=np.array([2]), y=np.array([1]), t=np.array([0.5])
    )
    np.save(tmp_path / "times.npy", np.array([1.0]))
    prepare_events(
        argparse.Namespace(
            events=tmp_path / "source.npz",
            timestamps=tmp_path / "times.npy",
            window=1.0,
            height=3,
            width=4,
            output=tmp_path / "maps",
            compress_events=compress,
        )
    )
    path = tmp_path / "maps" / ("000000.npz" if compress else "000000.npy")
    actual = load_event_map(path)
    assert actual.sum() == 1 and actual[0, 1, 2] == 1


@pytest.mark.parametrize(
    "option,compress",
    [([], True), (["--compress-events"], True), (["--no-compress-events"], False)],
)
def test_download_cli_passes_compression_preference(monkeypatch, option, compress):
    captured = {}

    def fake_download(config_path, output, **kwargs):
        captured.update(kwargs)
        return {"status": "dry-run"}

    monkeypatch.setattr("event_flow.cli.download_fred", fake_download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "event-flow",
            "download-fred",
            "--config",
            "config.json",
            "--dry-run",
            *option,
        ],
    )
    main()
    assert captured["compress_events"] is compress


def test_packing_compresses_sparse_maps_without_loss(tmp_path):
    events = np.zeros((720, 1280), dtype=bool)
    events[100:120, 200:220] = True
    path = save_event_map(tmp_path / "events.npz", events)
    assert path.stat().st_size < events.nbytes // 8
    np.testing.assert_array_equal(load_event_map(path)[0].bool().numpy(), events)
