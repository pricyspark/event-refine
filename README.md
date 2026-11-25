# Event-Flow

Event-Flow is a causal RGB/event object detector with recurrent refinement of
**bounding boxes and class scores**, initially targeting FRED (the Florence
RGB-Event Drone Dataset). It starts from FlowSIS's D-FINE/RT-DETRv2 adapters but
uses a detection-only training and inference path: no segmentation masks,
SigLIP2 prompts, or optical-flow model are required.

Implemented with Codex assistance. Detection accuracy on FRED has not yet been
measured; the implementation has synthetic integration and gradient tests.

## Model

1. Build aligned `[R,G,B,E]` inputs, where `E` is binary event occupancy.
2. Run D-FINE or RT-DETRv2 on every frame. Initialize its fourth input-channel
   weights to zero and preserve pretrained RGB weights.
3. Pool spatial event context for each detector query.
4. Use content/spatial attention to associate current queries with previous
   recurrent memory, then update a GRU state. Query indices are not track IDs.
5. Predict residual corrections to normalized boxes and class logits. Both
   correction heads start at zero, preserving the detector predictions initially.

Training backpropagates through contiguous clips into the event stem, detector,
and temporal head. Each clip starts with empty state. Streaming inference carries
state forward and resets at sequence boundaries, timestamp gaps, or image-size
changes. This implementation does not estimate optical flow, emit persistent
track IDs, or produce predictions between RGB frames.

## Setup

```bash
mamba env create -f environment.yml
conda activate event-flow
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
event-flow --help
```

All default dependencies, including build tools, are installed by mamba. Pip
only installs this local project; `--no-deps --no-build-isolation` prevents it from
resolving another dependency set or downloading a separate build environment.
Conda can be used instead of mamba with the same environment file.

If an earlier environment creation stopped during the old PTLFlow/OpenCV pip
step, keep the installed environment and update it from this corrected file:

```bash
mamba env update -n event-flow -f environment.yml
conda activate event-flow
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
event-flow --help
```

PTLFlow and pip OpenCV are excluded from the default environment because the
Event-Flow detector does not use optical flow. They are available through
`python -m pip install -e '.[legacy-flow]'` only for the inherited FlowSIS
optical-flow commands; that optional installation has its own dependency
resolution and is not needed for dataset import, training, or evaluation.
`pip check` verifies declared package dependencies, not GPU/driver availability.

Alternatively use the existing FlowSIS environment, with a separate shell for
this project. Editable-installing both projects in one environment can make their
inherited `flowsis` Python package and `flowsis-*` commands conflict. The new API
is `event_flow`; the inherited adapters remain under `flowsis`.

Transformers `>=5.13,<5.14` is required. Native ResNet and HGNetV2 backbones are
supported. Image size must be a positive multiple of 32 (at least 64 for training
with one image at a time). The initial CLI processes one clip at a time in FP32.

## Prepare FRED

The reproducible downloader provides presets for **1, 2, or 5 full videos**, plus **10 half videos**, each with
a pinned Hugging Face revision, archive sizes/checksums, preprocessing, and
sequence-level split assignments:

```bash
# List selected archives and bytes without downloading.
event-flow download-fred --config configs/data/fred-1-video.json --dry-run

# Choose one preset to download, verify, and import.
event-flow download-fred --config configs/data/fred-1-video.json
# Alternatives: fred-2-videos.json, fred-5-videos.json, or fred-10-videos-half.json
# All configs are under configs/data/.
```

See [subset configurations and training commands](configs/data/README.md).
Generated data lives under ignored `data/<config-name>/`; the shared download
cache is `data/fred-cache`. Each import writes a loader-ready `manifest.json`,
bit-packed compressed event maps, and provenance with source checksums, config, and environment
versions. The multi-video presets reserve sequence 1 for validation. Selected
ZIPs are downloaded whole; raw HDF5 streams are not extracted. Boundary-crossing
boxes are clipped. Missing annotation timestamps remain unsupervised by default.

Event maps use lossless bit packing plus compressed NPZ by default. Both the
downloader and `prepare-events` accept `--compress-events` (default) and
`--no-compress-events` (legacy NPY). Training detects the file format without a
flag. Existing imports are never silently converted or overwritten; use
`--no-compress-events` to reuse an old NPY import, or a new `--output` directory
to reimport from the shared ZIP cache with compression.

A measurement over all 16,942 maps in the installed five-video subset reduced
logical event-file size from 15.62 GB to 46.3 MB (99.7% smaller). RGB plus events
would occupy about 2.13 GB instead of 17.70 GB, excluding archive cache and
filesystem block overhead. See [storage measurement](docs/event_storage_benchmark.json).

For manually downloaded/extracted data, the original importer is also available:

Download and extract sequences from the
[official FRED repository/dataset](https://github.com/miccunifi/FRED). The importer
expects the published layout:

```text
FRED/
  train/<sequence>/RGB/*.jpg
  train/<sequence>/Event/Frames/*.png
  train/<sequence>/coordinates.txt
  test/<sequence>/...
```

The importer follows the official loader's natural frame ordering and annotation
timestamp cadence. It uses `coordinates.txt` in the extended/padded coordinate
system. Verify that RGB, events, and boxes are registered to that same image
plane; the loader rejects different image dimensions and out-of-bounds boxes.

**Choose the event visualization's background explicitly.** For example, only if
the event PNGs encode every active pixel against a black RGB background:

```bash
event-flow import-fred \
  --root /path/to/FRED \
  --event-background 0 0 0 \
  --output data/fred.json
```

Do not use that value for gray-background visualizations. Rendered images recover
occupancy only when every event is represented by a non-background pixel. If the
renderer cancels opposite polarities, applies a colormap, or uses temporal decay,
prepare occupancy directly from raw events instead. This importer does not parse
FRED's raw HDF5 schema or establish its rendering window/causality.

Missing annotation timestamps are marked unsupervised and excluded from training
clips and metrics; they are never assumed to be empty background. Explicit empty
annotations are supported in a custom manifest. No frames are removed before
matching timestamp indices. Timestamp mismatches fail with an error rather than
silently shifting labels. Override `--frame-period` when needed; the published
loader's convention, `0.033333`, is the default.

For raw events, export registered `x`, `y`, and `t` arrays to an NPZ file and
prediction timestamps to a one-dimensional NPY file. Both timestamps must use
the same units. For example, with timestamps in seconds:

```bash
event-flow prepare-events \
  --events data/events.npz --timestamps data/frame_times.npy \
  --window 0.033333 --height 480 --width 640 \
  --output data/event-occupancy
```

This produces one bit-packed compressed NPZ map per prediction timestamp using `(t-window, t]`;
future events are excluded, repeat events collapse to one, and polarity is
intentionally discarded. Point manifest frames to these maps with
`"event_encoding": "binary"`. The loader detects NPY/NPZ automatically. See [the manifest specification](docs/event_flow.md).

## Train

```bash
event-flow train \
  --manifest data/fred.json \
  --architecture dfine \
  --clip-length 4 --batch-size 2 --epochs 10 \
  --output outputs/event-flow
```

Use `--detector-model /path/to/detector` for a pretrained local detector, or
`--architecture rtdetrv2` for RT-DETRv2. The event stem is trainable from the start.
`--learning-rate` controls the detector and `--temporal-learning-rate` controls
the refinement branch. Training uses matched focal, L1, and GIoU losses on the
refined predictions plus a configurable auxiliary loss on the base predictions.
It does not run the inherited segmentation objectives or native D-FINE auxiliary
layer/distribution losses.

To train for another 10 epochs from an existing run, load its complete checkpoint:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 event-flow train \
  --manifest data/fred-5-videos/manifest.json \
  --validation-split validation \
  --init-from outputs/fred-5-videos-batch3-10epochs/last \
  --batch-size 3 --clip-length 4 --image-size 640 \
  --epochs 10 --log-every 25 \
  --output outputs/fred-5-videos-batch3-next10epochs
```

`--init-from` loads both the detector and temporal head. It starts a fresh AdamW
optimizer with the supplied learning rates (defaults: detector `1e-5`, temporal
`1e-4`); existing checkpoints do not contain optimizer state. Epochs and step
counters start again at 1 and 0 for this new run, and `--epochs 10` means ten
additional epochs. `best/` tracks the best validation loss within the new run.
Keep the previous run for comparisons. Classes and image size must match the
checkpoint; architecture and temporal hidden dimension come from the checkpoint.
Use a new output directory and do not combine `--init-from` with `--detector-model`.

`--batch-size` counts independent clips (default: 1) and also applies to validation.
With batch size 2 and clip length 4, each forward processes two frames at the
same temporal position, retaining separate recurrent memory for each clip.
The loss is averaged over clips and frames before one backward/optimizer update.
Memory resets between batches; final partial batches are retained and epoch loss
is weighted by their actual clip count. Different image sizes across clips are
resized and padded before batching. Larger batches increase VRAM use and change
batch normalization statistics and the number of updates per epoch.
`--max-steps` counts optimizer updates, so two steps at batch size 2 process up to
four clips. Profiler warmup/active steps also count optimizer updates.

For a smoke run, add `--max-steps 2 --max-validation-clips 2 --image-size 64 --device cpu`. This checks
plumbing, not model quality. Checkpoints are written to `outputs/event-flow/last`.

Training prints startup/device and dataset counts, then progress after the first
batch, every 10 batches, and each phase's final batch. Use `--log-every 1` to log each
batch. Records include completed/total clips, mean loss, elapsed time, estimated
remaining phase time, clips per second, and cumulative loading/compute times.
Progress is also appended to `progress.jsonl` in the output directory. Checkpoint
start/completion is printed; `training.jsonl` remains an epoch-level summary.
Timing includes preprocessing and loss computation in the compute field and is
not a GPU-kernel profiler. Changes to logging only affect newly started runs.
Checkpoints are still saved at epoch boundaries; optimizer-state resume is not
implemented, so interrupting an epoch loses work since the last checkpoint.

To capture a short CPU/CUDA trace after warm-up:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 event-flow train \
  --manifest data/fred-5-videos/manifest.json \
  --architecture dfine --device cuda \
  --max-steps 20 --log-every 5 \
  --profile --profile-warmup 10 --profile-steps 5 \
  --output outputs/profile-threads-2
```

Run this with no competing GPU training process. The first 10 steps are excluded
from the trace; the following 5 are recorded. The capture is exported before
training finishes. No validation split is supplied for this diagnostic.
`--profile` alone does not limit training; use `--max-steps` as shown. The first
epoch must contain enough training steps for the complete capture window.

Open `outputs/profile-threads-2/profile/trace.json` in
[Perfetto](https://ui.perfetto.dev/) using **Open trace file**. Inspect the CPU
thread, CUDA streams, and named `data_load`, `preprocess_and_transfer`, `forward`,
`matching_and_loss`, `backward`, `gradient_clip`, and `optimizer_step` regions.
`operators.txt` lists the most expensive CPU/device operators; `metadata.json`
records the device, thread counts, capture window, and CUDA event count. CUDA
tracing requires working driver/CUPTI profiling support; the exporter reports
when it captured no device events. Shape, stack, and memory recording are disabled
to keep trace overhead down. Profiled timings are diagnostic and should not be
compared directly with unprofiled throughput benchmarks.

FRED's official train/test layout has no assumed validation set. For model
selection, assign entire training sequences to a `validation` split in the
manifest, then pass `--validation-split validation`. The best validation-loss
checkpoint is saved to `best`. Keep the test split for final evaluation.

## Predict and evaluate

```bash
event-flow predict \
  --manifest data/fred.json --split test \
  --model outputs/event-flow/last --output outputs/predictions

event-flow evaluate \
  --manifest data/fred.json --split test \
  --model outputs/event-flow/last --output outputs/evaluation
```

Both commands write `predictions.jsonl` with original-coordinate pixel `xyxy`
boxes, scores, labels, timestamps, and sequence IDs. Evaluation also writes
`metrics.json` with class-aware AP50 and AP50:95, using 101-point interpolation.
This is a non-crowd box evaluator, not the full COCO evaluation protocol.
Evaluation defaults to a low score threshold of `0.001` and at most 100 detections
per frame; prediction defaults to `0.25`. Both settings are configurable.

Each checkpoint contains the complete four-channel detector and temporal head.
There is no dependency on a segmentation/base checkpoint, prompt bank, or optical
flow checkpoint. Runtime recurrent state is not saved.

## Python API

```python
from event_flow.checkpoint import load_checkpoint

model, classes = load_checkpoint("outputs/event-flow/last", device="cuda")
# rgb: [B,3,H,W] float in [0,1]; events: aligned [B,1,H,W] with values 0/1.
# Use torch.inference_mode() when streaming without gradients.
pixels, pixel_mask = model.prepare(rgb, events)
output = model(pixels, pixel_mask=pixel_mask, state=None)
# Carry output.state.detach() into the next frame; reset on a new sequence/gap.
```

## Throughput experiments

Use the isolated benchmark runner to compare independent clip batching, CPU
threads, repeated GPU input checks, and optional `torch.compile` without changing
normal training. For example:

```bash
python scripts/benchmark_throughput.py \
  --manifest data/fred-5-videos/manifest.json \
  --suite batching --batch-sizes 1 2 --threads 2 \
  --warmup 10 --steps 50 --repeats 3 \
  --output outputs/bench-batching
```

See [commands, experiment definitions, and result interpretation](docs/throughput_benchmarks.md).
The runner isolates cases in separate processes, records OOM/errors, excludes
warm-up from timing, and writes per-case results plus CSV summaries. Compilation
is an explicit experimental suite; it is not enabled by normal training.

## Development and provenance

```bash
python -m pytest tests/test_event_flow.py -q
```

Git history and the initial working snapshot are preserved in
[the source snapshot record](docs/source_snapshot.json). The original FlowSIS
README is archived at [docs/flowsis-readme.md](docs/flowsis-readme.md); its mask
training, prompt, and deployment commands describe the inherited project.
The `upstream` remote points to FlowSIS. Nothing is published automatically.

Please cite FRED when using its data; see the
[official citation](https://github.com/miccunifi/FRED#citation).
