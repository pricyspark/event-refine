"""Offline selective download/import coverage, developed with Codex assistance."""

from copy import deepcopy
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest

from event_flow.data import ClipDataset, load_manifest
from event_flow.download import download_fred, file_sha256, load_config

REVISION = "980a8fa0331a8f03ffbcb30d4bf673ad439fc0fd"


def write_archive(path: Path, sequence_id: str, *, palette_error=False, unsafe=False):
    with ZipFile(path, "w") as archive:
        coordinates = []
        for index in range(6):
            rgb = io.BytesIO()
            Image.fromarray(np.full((32, 48, 3), 100 + index, dtype=np.uint8)).save(
                rgb, format="JPEG"
            )
            # Real FRED RGB names encode wall-clock HH_MM_SS.microseconds.
            archive.writestr(
                f"{sequence_id}/RGB/Video_{sequence_id}_16_03_03.{index * 33333:06d}.jpg",
                rgb.getvalue(),
            )
            event = np.empty((32, 48, 3), dtype=np.uint8)
            event[:] = [30, 37, 52]
            event[10, 12] = [64, 126, 200]
            event[11, 12] = [255, 255, 255]
            if palette_error:
                event[0, 0] = [1, 2, 3]
            encoded = io.BytesIO()
            Image.fromarray(event).save(encoded, format="PNG")
            stamp = (index + 1) * 33333
            archive.writestr(
                f"{sequence_id}/Event/Frames/Video_{sequence_id}_frame_{stamp}.png",
                encoded.getvalue(),
            )
            if index != 0:
                coordinates.append(
                    f"{stamp / 1e6:.6f}: -2, 4, 55, 25, 1, Betafpv air75"
                )
        archive.writestr(f"{sequence_id}/coordinates.txt", "\n".join(coordinates))
        archive.writestr(f"{sequence_id}/Event/events.hdf5", b"not extracted")
        if unsafe:
            archive.writestr("../outside.txt", b"no")


def make_config(tmp_path: Path, *, count=1, **archive_options):
    videos = []
    archives = {}
    for index in range(count):
        archive = tmp_path / f"{index}.zip"
        write_archive(archive, str(index), **archive_options)
        name = f"train/{index}.zip"
        archives[name] = archive
        videos.append(
            {
                "archive": name,
                "split": "train" if index == 0 else "validation",
                "sha256": file_sha256(archive),
                "size_bytes": archive.stat().st_size,
                "frame_start": 0,
                "frame_stop": None,
            }
        )
    config = {
        "format_version": 1,
        "repo_id": "GabrieleMagrini/FRED",
        "revision": REVISION,
        "videos": videos,
        "preprocess": {
            "event_background": [30, 37, 52],
            "event_palette": [[30, 37, 52], [64, 126, 200], [255, 255, 255]],
            "boxes": "extended",
            "clip_boxes": True,
            "class_mapping": "single-drone",
            "timestamp_source": "event_filename_us",
            "missing_annotations": "unlabeled",
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path, config, archives


def test_dry_run_has_no_downloads_or_output(tmp_path, monkeypatch):
    path, _, _ = make_config(tmp_path, count=2)
    monkeypatch.setattr(
        "event_flow.download.hf_hub_download",
        lambda **kwargs: pytest.fail("Dry run attempted network"),
    )
    output = tmp_path / "output"
    cache = tmp_path / "cache"
    plan = download_fred(path, output, cache_dir=cache, dry_run=True)
    assert plan["video_count"] == 2
    assert plan["revision"] == REVISION
    assert not output.exists() and not cache.exists()


def test_selective_download_creates_loader_ready_clips_and_provenance(
    tmp_path, monkeypatch
):
    path, config, archives = make_config(tmp_path, count=2)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(archives[kwargs["filename"]])

    monkeypatch.setattr("event_flow.download.hf_hub_download", fake_download)
    output = tmp_path / "output"
    result = download_fred(path, output, cache_dir=tmp_path / "cache")
    assert result["status"] == "imported"
    assert [call["filename"] for call in calls] == ["train/0.zip", "train/1.zip"]
    assert all(
        call["revision"] == REVISION and call["repo_type"] == "dataset"
        for call in calls
    )
    classes, sequences = load_manifest(output / "manifest.json")
    assert classes == ("drone",)
    assert [sequence.split for sequence in sequences] == ["train", "validation"]
    assert not sequences[0].frames[0].annotated
    assert sequences[0].frames[1].boxes == ((0.0, 4.0, 48.0, 25.0),)
    assert sequences[0].frames[1].timestamp == pytest.approx(0.066666)
    clips = ClipDataset(output / "manifest.json", split="train", clip_length=4)
    assert len(clips) == 2
    assert clips[0][0].events.sum() == 2
    assert not list(output.rglob("*.hdf5"))
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["config"] == config
    assert provenance["videos"][0]["clipped_boxes"] == 5
    assert provenance["manifest_sha256"] == file_sha256(output / "manifest.json")
    assert json.loads((output / "manifest.json").read_text())["sequences"][0]["frames"][
        1
    ]["track_ids"] == [1]
    # Repeating the same configuration validates the existing import without
    # downloading again or publishing partially overwritten outputs.
    repeated = download_fred(path, output, cache_dir=tmp_path / "cache")
    assert repeated["status"] == "already-imported-and-validated"
    assert len(calls) == 2


def test_frame_subset_keeps_original_timestamps(tmp_path, monkeypatch):
    path, config, archives = make_config(tmp_path)
    config["videos"][0].update(frame_start=2, frame_stop=6)
    path.write_text(json.dumps(config))
    monkeypatch.setattr(
        "event_flow.download.hf_hub_download",
        lambda **kwargs: str(archives[kwargs["filename"]]),
    )
    output = tmp_path / "output"
    download_fred(path, output, cache_dir=tmp_path / "cache")
    _, sequences = load_manifest(output / "manifest.json")
    assert len(sequences[0].frames) == 4
    assert sequences[0].frames[0].timestamp == pytest.approx(0.099999)
    assert len(ClipDataset(output / "manifest.json", split="train")) == 1


@pytest.mark.parametrize("failure", ["checksum", "palette", "unsafe", "range"])
def test_failed_import_never_publishes_partial_output(tmp_path, monkeypatch, failure):
    path, config, archives = make_config(
        tmp_path, palette_error=failure == "palette", unsafe=failure == "unsafe"
    )
    if failure == "checksum":
        config["videos"][0]["sha256"] = "0" * 64
    if failure == "range":
        config["videos"][0]["frame_stop"] = 100
    path.write_text(json.dumps(config))
    monkeypatch.setattr(
        "event_flow.download.hf_hub_download",
        lambda **kwargs: str(archives[kwargs["filename"]]),
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        download_fred(path, output, cache_dir=tmp_path / "cache")
    assert not output.exists()
    assert not list(tmp_path.glob(".output-*"))
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.parametrize("change", ["revision", "duplicate", "test_leak", "background"])
def test_config_rejects_unreproducible_or_leaking_selections(tmp_path, change):
    path, config, _ = make_config(tmp_path)
    if change == "revision":
        config["revision"] = "main"
    elif change == "duplicate":
        config["videos"].append(deepcopy(config["videos"][0]))
    elif change == "test_leak":
        config["videos"][0]["archive"] = "test/21.zip"
    else:
        config["preprocess"]["event_background"] = [999, 0, 0]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)


def test_committed_presets_have_requested_counts_and_consistent_holdout():
    root = Path(__file__).resolve().parents[1] / "configs/data"
    for count in (1, 2, 5):
        name = f"fred-{count}-video" + ("s" if count != 1 else "") + ".json"
        config = load_config(root / name)
        assert len(config.videos) == count
        assert config.revision == REVISION
        assert [video.archive for video in config.videos] == [
            f"train/{index}.zip" for index in range(count)
        ]
        if count > 1:
            assert config.videos[1].split == "validation"


@pytest.mark.parametrize("source_count", [5, 6])
def test_fractional_selection_uses_first_half_and_original_timestamps(
    tmp_path, monkeypatch, source_count
):
    from zipfile import ZipFile

    path, config, archives = make_config(tmp_path)
    if source_count == 5:
        # Remove the final pair/annotation to exercise floor rounding for an
        # odd number of frames, retaining the real source naming convention.
        original = archives["train/0.zip"]
        replacement = tmp_path / "short.zip"
        with ZipFile(original) as source, ZipFile(replacement, "w") as destination:
            for name in source.namelist():
                if name.endswith("16_03_03.166665.jpg") or name.endswith(
                    "_frame_199998.png"
                ):
                    continue
                content = source.read(name)
                if name.endswith("coordinates.txt"):
                    content = b"\n".join(
                        line
                        for line in content.splitlines()
                        if not line.startswith(b"0.199998:")
                    )
                destination.writestr(name, content)
        archives["train/0.zip"] = replacement
        config["videos"][0].update(
            sha256=file_sha256(replacement), size_bytes=replacement.stat().st_size
        )
    config["videos"][0]["frame_fraction"] = 0.5
    path.write_text(json.dumps(config))
    monkeypatch.setattr(
        "event_flow.download.hf_hub_download",
        lambda **kwargs: str(archives[kwargs["filename"]]),
    )
    output = tmp_path / "output"
    result = download_fred(path, output, cache_dir=tmp_path / "cache")
    _, sequences = load_manifest(output / "manifest.json")
    assert len(sequences[0].frames) == source_count // 2
    assert sequences[0].frames[0].timestamp == pytest.approx(0.033333)
    assert sequences[0].frames[-1].timestamp == pytest.approx(
        (source_count // 2) * 0.033333
    )
    assert result["videos"][0]["frame_stop"] == source_count // 2
    assert result["videos"][0]["frame_fraction"] == 0.5


@pytest.mark.parametrize("fraction", [0, -0.5, 1.1, float("nan"), True, "half"])
def test_invalid_fraction_is_rejected(tmp_path, fraction):
    path, config, _ = make_config(tmp_path)
    config["videos"][0]["frame_fraction"] = fraction
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="frame_fraction"):
        load_config(path)


@pytest.mark.parametrize("start,stop", [(1, None), (0, 3)])
def test_fraction_cannot_be_combined_with_explicit_bounds(tmp_path, start, stop):
    path, config, _ = make_config(tmp_path)
    config["videos"][0].update(frame_fraction=0.5, frame_start=start, frame_stop=stop)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Fractional selection"):
        load_config(path)


def test_ten_video_preset_selects_half_of_each_training_archive():
    path = Path(__file__).resolve().parents[1] / "configs/data/fred-10-videos-half.json"
    config = load_config(path)
    assert len(config.videos) == 10
    assert all(video.frame_fraction == 0.5 for video in config.videos)
    assert [video.archive for video in config.videos] == [
        f"train/{index}.zip" for index in (0, 1, 2, 3, 4, 5, 6, 7, 10, 11)
    ]
    assert [
        video.archive for video in config.videos if video.split == "validation"
    ] == ["train/1.zip"]


@pytest.mark.parametrize("compress", [True, False])
def test_download_storage_flag_and_existing_import_compatibility(
    tmp_path, monkeypatch, compress
):
    path, _, archives = make_config(tmp_path)
    monkeypatch.setattr(
        "event_flow.download.hf_hub_download",
        lambda **kwargs: str(archives[kwargs["filename"]]),
    )
    output = tmp_path / "output"
    result = download_fred(
        path, output, cache_dir=tmp_path / "cache", compress_events=compress
    )
    _, sequences = load_manifest(output / "manifest.json")
    expected = ".npz" if compress else ".npy"
    assert all(frame.events.suffix == expected for frame in sequences[0].frames)
    assert result["event_storage"] == ("packed-npz-v1" if compress else "npy")
    assert len(ClipDataset(output / "manifest.json", split="train")[0]) == 4
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["event_storage"] == result["event_storage"]
    assert (
        download_fred(
            path, output, cache_dir=tmp_path / "cache", compress_events=compress
        )["status"]
        == "already-imported-and-validated"
    )
    with pytest.raises(ValueError, match="different event storage"):
        download_fred(
            path, output, cache_dir=tmp_path / "cache", compress_events=not compress
        )
