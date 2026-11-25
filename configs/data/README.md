# Reproducible FRED subsets

Prepared with Codex assistance. Run commands from the Event-Flow repository root.

| Config | Downloaded source videos | Training | Validation | Download size |
|---|---|---|---|---|
| `fred-1-video.json` | `train/0.zip` | 0 | none | 450.8 MB |
| `fred-2-videos.json` | `train/{0,1}.zip` | 0 | 1 | 903.6 MB |
| `fred-5-videos.json` | `train/{0,1,2,3,4}.zip` | 0, 2, 3, 4 | 1 | 2.42 GB |
| `fred-10-videos-half.json` | `train/{0,1,2,3,4,5,6,7,10,11}.zip` | all except 1 | 1 | 4.85 GB |

Sizes are decimal compressed download bytes, excluding extraction and local cache
storage. All presets pin the same full Hugging Face commit and archive SHA256
checksums. The official test split stays untouched. Sequence 1 remains validation
in every multi-video preset. These are small development subsets, not a full
benchmark protocol.

```bash
# Inspect exact files/bytes without network access or downloads.
event-flow download-fred --config configs/data/fred-1-video.json --dry-run

# Download, verify, and import one selected preset.
event-flow download-fred --config configs/data/fred-1-video.json
event-flow download-fred --config configs/data/fred-2-videos.json
event-flow download-fred --config configs/data/fred-5-videos.json
event-flow download-fred --config configs/data/fred-10-videos-half.json
```

Run only the preset you need. The shared `data/fred-cache` reuses matching ZIPs
across presets. Outputs default to `data/<config filename without .json>/`;
`--output` and `--cache-dir` override these paths. These output/cache paths resolve
relative to your current working directory. Configuration file selection is
explicit, so a different working directory can use absolute config/output paths.

The command writes `manifest.json` for the existing loaders, `provenance.json`,
a copy of the config, original RGB JPEGs, and bit-packed compressed event NPZ maps. It downloads
whole selected ZIPs but does not extract their raw HDF5 streams. Existing imports
with matching configs and event storage are validated and reused; a different config cannot overwrite
an existing output. Failed imports are staged temporarily and leave no complete
output. Cached downloads remain available for a retry.

Event storage is configurable independently of which frames are selected:

```bash
# Default: pack eight boolean pixels into each byte and compress to NPZ.
event-flow download-fred --config configs/data/fred-5-videos.json --compress-events

# Opt out: save one-byte-per-pixel boolean NPY files.
event-flow download-fred --config configs/data/fred-5-videos.json --no-compress-events
```

The loader detects `.npz` and `.npy` automatically. The flag and storage format
are recorded in provenance, and `--dry-run` prints the requested format. Both
formats decode to identical occupancy values. Changing storage for an existing
output requires a different `--output` directory; the original data is preserved
and the shared archive cache avoids repeat downloads. `prepare-events` accepts
the same flags.

```bash
event-flow train \
  --manifest data/fred-2-videos/manifest.json \
  --validation-split validation --architecture dfine \
  --output outputs/fred-two-video
```

Each video config supports `frame_start` (inclusive) and `frame_stop` (exclusive;
`null` means the end). The 1/2/5-video presets import full videos.
The 10-video preset sets `frame_fraction: 0.5`, importing the first
`floor(total_frames * 0.5)` frames of each video, including the validation video.
Fractional selection requires `frame_start: 0` and `frame_stop: null`; it cannot
be combined with explicit bounds. Fractions must be finite and in `(0,1]`.
The exact resolved frame bounds and fraction are recorded in provenance.
IDs 8 and 9 are omitted because they belong to the official test split.
The full 4.85 GB of selected archives must still download; existing cached
archives 0–4 are reused, leaving about 2.43 GB of additional downloads.

For faster local tests, copy a
config and select a contiguous interval containing annotations. Bounds are
positions in the complete timestamp-sorted frame list; annotation matching uses
original event filename timestamps, never renumbered subset positions. A shorter
interval reduces extraction/preprocessing, not the archive download size.

Preprocessing is explicit and recorded: extended pixel-xyxy boxes are clipped
to the image plane, degenerate clipped boxes are dropped and counted, drone model
names map to class 0 (`drone`), track IDs/source labels are retained as metadata,
and event timestamp microseconds become seconds. Palette mismatches and mismatched
image dimensions cause an error. Missing annotation timestamps default to
`unlabeled`, not negative examples. Only change `missing_annotations` to `empty`
if the annotation source is confirmed exhaustive.

The three rendered event colors are explicitly specified. Every non-background
pixel becomes occupancy 1; raw polarity, counts, and intra-window timestamps are
not retained. The accumulation window/causality of the publisher's rendering
cannot be verified by this importer. Use raw events for experiments requiring an
explicit event window.

Source: [FRED](https://huggingface.co/datasets/GabrieleMagrini/FRED), revision
`980a8fa0331a8f03ffbcb30d4bf673ad439fc0fd`. Please cite the dataset's authors as
specified in its dataset card. Configs and importer belong in Git; downloaded
archives, generated manifests, images, event maps, and caches belong under ignored
`data/`.
