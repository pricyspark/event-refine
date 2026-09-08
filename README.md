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

Build the reviewed Arrow dataset, generate and review its three-prompt SigLIP2
suite, and then train or select a detector checkpoint. Every detector class
receives the same number of `[P,768]` prompt vectors.

```bash
flowsis-build-annotated-dataset
flowsis-author-text-prompts
# Review/edit data/new/text_prompts.json before embedding it.
flowsis-embed-text-prompts
flowsis-train-detector --detector dfine

flowsis-train-base \
  --detector-model outputs/detectors/dfine/final \
  --text-embeddings-dir data/new/text-embeddings \
  --dataset-path data/new/segmentation-dataset \
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
  --text-embeddings-dir data/new/text-embeddings \
  --dataset-path data/new/segmentation-dataset \
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
  --dataset-path data/new/segmentation-dataset \
  --split test

# Same-checkpoint reliance tests; retraining is still required for causal claims
flowsis-ablate-base \
  --model-path outputs/base/best \
  --dataset-path data/new/segmentation-dataset \
  --split test

flowsis-live-base \
  --model-path outputs/base/best \
  --video-source data/new/videos/20260901_191034.mp4

flowsis-profile-base \
  --model-path outputs/base/best \
  --video-source data/new/videos/20260901_191034.mp4
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
