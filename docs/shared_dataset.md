# Shared still-image and saved-video-frame datasets

Build the dataset in `flowsis-data`, then use the same export for both stages:

```shell
cd ../flowsis-data
python build_dataset.py --output-path exports/combined-dataset
cd ../flowsis
flowsis-train-base --dataset-path ../flowsis-data/exports/combined-dataset \
  --text-embeddings-dir path/to/embeddings --video-fraction 0.5
flowsis-train-temporal --dataset-path ../flowsis-data/exports/combined-dataset \
  --base-model outputs/base/best
```

The combined builder is the supported path for the new `video_frame_manifest.csv`
format. The older `flowsis-build-temporal-dataset` command still serves legacy
`video_object_manifest.csv` annotations; it does not translate the new format.

Each export references the original JPG/PNG and NPZ assets. Keep their relative
layout when moving data. `load_local_dataset(path)` resolves asset paths in memory,
including `mask_bundle_path`. Raw `datasets.load_from_disk` does not resolve them.

Base/detector command-line workflows select `base_eligible` rows automatically.
Python callers can use `base_dataset_view(split)` or `temporal_dataset_view(split)`
from `flowsis.data.loaders`; these select rows without changing the shared class
vocabulary or split assignment. The temporal loader chooses saved PNGs when present,
retains timestamps, and keeps track identities and supervision flags through
occlusion. Legacy MP4-based temporal rows remain supported.

Base training uses dynamic sampling with replacement. `--video-fraction` (default
0.5) controls the expected source mixture when both pools exist. Stills are balanced
by capture sample, videos by video ID, and frames are uniform within each group.
With only one pool, it receives all draws regardless of this option. Ungrouped
stills count as individual groups. `--samples-per-epoch` defaults to the eligible
row count. Sampling is separate from annotation-confidence loss weighting.

Temporal training samples dynamic clip lengths and strides, balancing eligible
videos. Every intervening frame must exist, and all frames must belong to the same
interval, even for strides greater than one. A clip needs a visible, mask-valid
observation. Intervals too short for a geometry are excluded from that geometry.
If no interval supports any configured geometry, loader construction fails clearly.

Validation uses deterministic evenly spread starts and one video per batch. It
reports overall and per-video losses. Ensure the smaller of `--validation-batches`
and `--batches-per-epoch` covers all eligible validation videos.

Splits come from the export and are shared across model stages. Never pretrain the
base model or replay on frames held out for temporal evaluation. The default builder
groups physical tools; session and whole-video grouping are also available. Its
optional section splits are for within-recording development and retain explicit
interval IDs and configurable gaps. Each video currently contributes only its first
contiguous accepted run. Short runs may be training-only; consult `build_report.json`
for intervals, fallback reasons, and snippet counts. Keep the export fixed during an
experiment, since accepting more frames and rebuilding may change split assignments.

For mask-head fine-tuning, the same export can also be passed to
`--still-dataset-path` for base-valid replay from its training split. Class names
and their order must match the base checkpoint exactly; the combined export keeps
one vocabulary for both views. A rebuilt vocabulary requires matching text embeddings
and a compatible base checkpoint.
