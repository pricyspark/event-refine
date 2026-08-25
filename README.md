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

First train or select a detector checkpoint and generate pooled SigLIP2 prompt
embeddings. Every detector class needs the same number of `[P,768]` prompt
vectors.

```bash
flowsis-train-detector --detector dfine

flowsis-train-base \
  --detector-model outputs/detectors/dfine/final \
  --text-embeddings-dir data/manifests/text-embeddings \
  --dataset-path data/segmentation-dataset \
  --images-dir data/frames \
  --masks-dir data/masks \
  --train-stages head:5,neck:3,all:2 \
  --output-dir outputs/base
```

Datasets can be moved between machines: if a serialized image or mask path no
longer exists, the loader relocates it below `--images-dir` or `--masks-dir`
using the stored video/frame metadata. Rebuilding is therefore optional when
the annotations and local assets are otherwise unchanged.

For a quick end-to-end plumbing test, bound both training and validation work:

```bash
flowsis-train-base \
  --detector-model outputs/good/dfine-m-single \
  --text-embeddings-dir data/manifests/text-embeddings \
  --dataset-path data/segmentation-dataset \
  --images-dir data/frames \
  --masks-dir data/masks \
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

Checkpoints are self-contained model directories. Hugging Face detector files
and a versioned `base.pt` live together; `base.pt` contains the set-mask weights,
prompt bank, ordered labels, thresholds, and architecture metadata. External
prompt files are only needed when starting a new training run. Legacy `head.pt`
checkpoints intentionally fail with an incompatibility error.

## Evaluate and deploy

```bash
flowsis-evaluate-base \
  --model-path outputs/base/best \
  --dataset-path data/segmentation-dataset \
  --images-dir data/frames \
  --masks-dir data/masks \
  --split test

# Same-checkpoint reliance tests; retraining is still required for causal claims
flowsis-ablate-base \
  --model-path outputs/base/best \
  --dataset-path data/segmentation-dataset \
  --images-dir data/frames \
  --masks-dir data/masks \
  --split test

flowsis-live-base \
  --model-path outputs/base/best \
  --video-source data/raw/1.mp4

flowsis-profile-base \
  --model-path outputs/base/best \
  --video-source data/raw/1.mp4
```

Evaluation reports class-aware matched and oracle-query mask quality, union
metrics, boundary F1, box/mask AP50, precision/recall, empty-frame false
positives, and per-class metrics. Validation uses deterministic geometry;
overlap compositing remains training-only.

The segmentation dataset builder groups repeated annotations by image. For new
data, use `--group-splits-by-video` (or supply official pre-made splits) to avoid
source leakage. `--include-missing-masks-as-empty` is intentionally opt-in and
must only be used when missing masks are authoritative negatives.

The generic temporal and optical-flow modules remain available, but the old
selection-coupled recurrent pipeline has been removed. Per-instance temporal
association and recurrent state are deferred until the set-based baseline is
established.

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
