# FlowSIS

FlowSIS is an end-to-end, set-based instance-segmentation model built on D-FINE
or RT-DETRv2. It retains every detector query, creates one shared stride-4 pixel
feature map, conditions queries with a frozen SigLIP2 prompt bank, and predicts
one independent mask per query. Accepted masks are also exposed as a max-logit
binary foreground union.

## Setup

```bash
conda env create -f environment.yml
conda activate flowsis
pip install -e .
```

Transformers is pinned to `>=5.13,<5.14` because the differentiable stride-4
backbone tap is validated against the 5.13 detector layouts.

## Train

<!-- Training/checkpoint overview clarified with Codex assistance. -->
Train in this order: **detector → base instance segmentation → temporal
refinement**. The base contains the detector plus the set-mask head and prompt
bank. The temporal pipeline contains that complete base, a pretrained optical-flow
model, and a recurrent refinement branch. A trained base checkpoint is loaded
directly with `flowsis-train-temporal --base-model PATH_TO_BASE/best`; the base
and flow model are frozen during initial temporal training.

For the current combined coarse dataset, use the
[step-by-step commands](docs/combined_coarse_commands.md),
[overnight workflow](docs/overnight_training.md), and
[short temporal smoke script](experiments/smoke_temporal.sh).

Build the reviewed Arrow dataset, generate and review its three-prompt SigLIP2
suite, and then train or select a detector checkpoint. Every detector class
receives the same number of `[P,768]` prompt vectors.

```bash
flowsis-build-annotated-dataset
flowsis-author-text-prompts
# Review/edit data/text_prompts.json before embedding it.
flowsis-embed-text-prompts
flowsis-train-detector --detector dfine

flowsis-train-base \
  --detector-model outputs/detectors/dfine/final \
  --text-embeddings-dir data/text-embeddings \
  --dataset-path data/segmentation-dataset \
  --train-stages head:5,neck:3,all:2 \
  --output-dir outputs/base
```

Manual prompt authoring is the dependency-free default. An optional API-backed
draft generator is available with `pip install -e '.[prompt-generation]'` and
`flowsis-generate-text-prompts`; generated descriptions always require review.

The builder stores absolute external image, video, and mask paths in Arrow.
Rebuild the lightweight Arrow dataset after moving its source annotations or
media to another machine.

For a quick end-to-end plumbing test, bound both training and validation work:

```bash
flowsis-train-base \
  --detector-model outputs/good/dfine-m-single \
  --text-embeddings-dir data/text-embeddings \
  --dataset-path data/segmentation-dataset \
  --output-dir outputs/base-smoke \
  --train-stages head:1 \
  --max-train-images 20 \
  --max-validation-images 10 \
  --max-steps 20 \
  --batch-size 1 \
  --num-workers 0 \
  --no-use-overlap-augment
```

The detector runs in FP32 by default even when mask-head AMP is enabled. Keep
`--detector-amp` disabled for D-FINE; use `--no-amp` as a full-precision
diagnostic if a GPU still produces non-finite values.

To require CUDA rather than silently accepting the automatic CPU fallback, add
`--device cuda`. Training logs a `runtime` event at startup containing the
selected device, GPU name, PyTorch/CUDA versions, and AMP state.

Useful baselines and retrained ablations include:

```bash
# Frozen detector baseline
flowsis-train-base --train-stages head:10 ...

# Immediate end-to-end training
flowsis-train-base --train-stages all:10 ...

# Architectural controls (retrain each for causal comparisons)
flowsis-train-base --text-conditioning learned ...
flowsis-train-base --text-conditioning none ...
flowsis-train-base --no-box-prior ...
flowsis-train-base --pixel-detail stride8 ...
```

For the staged checkpoint, unfreezing, learning-rate, throughput, and seed
comparison protocol, see [experiments/BASE_TRAINING.md](experiments/BASE_TRAINING.md).
The accompanying `experiments/run_base_sweep.sh` writes a reproducible manifest
and stability/throughput metrics for every run. Its default primary detector is
the dataset-tuned D-FINE Large checkpoint.

Checkpoints are self-contained model directories. Hugging Face detector files
and a versioned `base.pt` live together; `base.pt` contains the set-mask weights,
prompt bank, ordered labels, thresholds, and architecture metadata. External
prompt files are only needed when starting a new training run.

## Evaluate and deploy

```bash
flowsis-evaluate-base \
  --model-path outputs/base/best \
  --dataset-path data/segmentation-dataset \
  --split test

# Same-checkpoint reliance tests; retraining is still required for causal claims
flowsis-ablate-base \
  --model-path outputs/base/best \
  --dataset-path data/segmentation-dataset \
  --split test

flowsis-live-base \
  --model-path outputs/base/best \
  --video-source data/videos/20260901_191034.mp4

flowsis-profile-base \
  --model-path outputs/base/best \
  --video-source data/videos/20260901_191034.mp4
```

Evaluation reports class-aware matched and oracle-query mask quality, union
metrics, boundary F1, box/mask AP50, precision/recall, empty-frame false
positives, and per-class metrics. Validation uses deterministic geometry;
overlap compositing remains training-only.

The dataset builder keeps connected physical-tool and capture-session groups
in one split, preventing rotation, setup, and background leakage. Only reviewed,
exhaustively annotated images contribute background supervision.

The temporal refiner is a separate, identity-initialized residual branch. The
base detector and set-mask head still run completely on every frame; the
refiner recurrently combines each current instance observation with flow-warped
previous refined logits, hidden state, and shared stride-4 features. Runtime
track selection remains a separate 30-frame policy and does not backpropagate
into segmentation.

The saved artifacts remain separate: a base checkpoint directory contains the
detector files and `base.pt`, while `temporal-best.pt` contains only the temporal
branch. Temporal inference therefore needs the matching base directory,
`temporal-best.pt`, and the same pretrained flow model/checkpoint used in
training. If temporal training uses `--finetune-mask-head`, pair its temporal
checkpoint with the accompanying `base-best` directory. Dataset class names and
their order must match the base checkpoint exactly.

For data prepared in the neighboring `flowsis-data` repository, use its combined
export for both base and temporal stages. See
[shared dataset loading](docs/shared_dataset.md) for commands, sampling, and splits.

The reviewed image/video annotation and training workflow is documented in
[docs/annotation_and_temporal_training.md](docs/annotation_and_temporal_training.md).

See [docs/end_to_end_architecture.md](docs/end_to_end_architecture.md) for tensor
shapes, loss flow, checkpoint semantics, and the enhancement roadmap.

## Detector interface

```python
from flowsis.pretrained import load_detector

detector = load_detector()  # RT-DETRv2 default
detector = load_detector(architecture="dfine")
detector = load_detector("outputs/detectors/dfine/final")
```

PTLFlow `0.4.2` requires Python `<3.14`, so the environment uses Python 3.13.


### Temporary collected-data dataset

Build the current reviewed still-image dataset in the adjacent `flowsis-data`
repository with `python build_dataset.py` (requires `datasets`, NumPy, and Pillow).
The builder and its documentation live there; the resulting dataset is at
`../flowsis-data/data/segmentation-dataset`. Pass that path to the training CLI's
`--dataset-path` option. Image and mask paths are stored relative to the dataset directory. Move the
source repository as a unit to preserve the relative layout. Model training,
evaluation, and visualization resolve these paths when opening the dataset.
For Python callers, use `flowsis.data.loaders.load_local_dataset(path)` rather
than opening these exports directly with `datasets.load_from_disk`. Each image
retains all of its instances. `area` is foreground mask pixel count, including
after augmentation. Explicitly reviewed empty images are supported by the data
builder when an empty mask bundle verifies that no positive masks were saved.

Collected single-tool views already cover 360 degrees. Training and augmentation
previews now default to extra rotation from 0 to 45 degrees.
`--rotation-min-degrees` and `--rotation-max-degrees` override that range (use
`--rotation-max-degrees 360` for the original full-angle augmentation). Synthetic
overlap remains enabled, with zero to one overlay and `--overlap-p 0.25`: a 25%
chance of one additional image before visibility/severity filtering. An overlay
image can itself contain multiple tools. Use `--no-use-overlap-augment` to disable
it, or `--overlap-min-overlays 1 --overlap-max-overlays 1` for one overlay on every
sample. Validation and test images are not augmented this way.

The collected-data export uses image-level `tool_ids` and per-object `tool_id`;
there is no singular image-level physical tool field. Object `instance_id`
preserves the original annotation identifier (scoped to its source image). The
mask loader adds runtime `source_image_id` so synthetic composites retain that
provenance, and initializes image/object `modified` flags at runtime. These flags
are not stored in new exports. Base-model training balances capture setups by
Sample ID, assigning each sample equal total weight across its images. Images
without a Sample ID are weighted individually. Detector-only training continues
to use its existing image-shuffling policy.
