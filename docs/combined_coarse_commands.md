# Commands for the regenerated coarse dataset

Prepared with Codex assistance on 2026-11-13. No training code changes are
required for these runs. Run the blocks sequentially in one Bash shell. The
inspected export has 11 classes and 2,235/370/363 train/validation/test rows.
Use a new `FLOWSIS_RUN` for a new experiment series; do not reuse old outputs.
Commands below have not been run for GPU training.

## Environment and prompts

```bash
conda activate flowsis
cd /home/pricyspark/flowsis
export FLOWSIS_DATASET=../flowsis-data/exports/combined-sections-coarse
export FLOWSIS_RUN=coarse-sections-v1
export FLOWSIS_DETECTOR_RUN="outputs/detectors/$FLOWSIS_RUN"
export FLOWSIS_BASE_RUNS="outputs/base/$FLOWSIS_RUN"
export FLOWSIS_TEMPORAL_RUNS="outputs/temporal/$FLOWSIS_RUN"
export FLOWSIS_PROMPTS=configs/prompt_suites/combined-coarse-v1.json
export FLOWSIS_EMBEDDINGS=data/combined-coarse/text-embeddings
export PYTHONUNBUFFERED=1
set -o pipefail

python -m pytest -q tests/test_dataset_mixture.py tests/test_temporal_dataset.py \
  tests/test_base_training.py tests/test_temporal.py

flowsis-author-text-prompts --dataset-path "$FLOWSIS_DATASET" \
  --output-path "$FLOWSIS_PROMPTS"
# Finish/review descriptions for all 11 classes before continuing.
flowsis-embed-text-prompts --dataset-path "$FLOWSIS_DATASET" \
  --prompt-suite "$FLOWSIS_PROMPTS" --output-dir "$FLOWSIS_EMBEDDINGS" \
  --model google/siglip2-base-patch16-224 --device cuda
```

Do not reuse the old nine-class embeddings. If console commands are unavailable,
install this checkout in the activated environment with `python -m pip install -e .`.

## Detector smoke and baseline

```bash
flowsis-train-detector --detector dfine \
  --model ustc-community/dfine-medium-obj365 --dataset-path "$FLOWSIS_DATASET" \
  --output-dir "${FLOWSIS_DETECTOR_RUN}-smoke" \
  --epochs 1 --max-steps 20 --batch-size 2 --num-workers 0 \
  --image-size 640 --lr 1e-4 --backbone-lr 1e-5 --max-grad-norm 1 \
  --rotation-min-degrees -22.5 --rotation-max-degrees 22.5 \
  --use-overlap-augment --overlap-min-overlays 0 --overlap-max-overlays 1 --overlap-p 0.25 \
  --sanity-decode --device cuda --seed 42

flowsis-train-detector --detector dfine \
  --model ustc-community/dfine-medium-obj365 --dataset-path "$FLOWSIS_DATASET" \
  --output-dir "$FLOWSIS_DETECTOR_RUN" \
  --epochs 20 --batch-size 8 --num-workers 4 --image-size 640 \
  --lr 1e-4 --backbone-lr 1e-5 --max-grad-norm 1 --device cuda --seed 42 \
  --rotation-min-degrees -22.5 --rotation-max-degrees 22.5 \
  --use-overlap-augment --overlap-min-overlays 0 --overlap-max-overlays 1 --overlap-p 0.25
```

D-FINE stays FP32; these commands deliberately omit detector AMP. Start from
generic pretrained weights because older FlowSIS weights may have seen the new
holdout tools. The smaller backbone LR is a proposed starting recipe, not a
previously established optimum.

Select the lowest-validation-loss saved detector as an initial candidate:

```bash
export FLOWSIS_DETECTOR=$(python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ['FLOWSIS_DETECTOR_RUN'])
events = [json.loads(line) for line in (root / 'training_log.jsonl').read_text().splitlines()]
best = min((e for e in events if e['event'] == 'validation_epoch'), key=lambda e: e['loss'])
checkpoint = root / f"checkpoint-{best['global_step']:06d}"
assert checkpoint.is_dir(), checkpoint
print(checkpoint)
PY
)

for iou in 0.5 0.75; do
  flowsis-evaluate-detector --detector dfine --model "$FLOWSIS_DETECTOR" \
    --dataset-path "$FLOWSIS_DATASET" --split validation --iou-threshold "$iou" \
    --output-dir "$FLOWSIS_DETECTOR_RUN/eval-iou${iou}" --device cuda
done
```

Optionally evaluate other saved checkpoints before fixing `FLOWSIS_DETECTOR`.
Loss selection is a starting rule; it need not maximize AP. Keep the same chosen
detector for all base comparisons.

## Base smoke, memorization and mixture comparisons

This shell helper fixes the common settings. Extra flags override the defaults
when needed. It refuses to reuse an output directory.

```bash
base_run() {
  local name=$1
  shift
  local output="$FLOWSIS_BASE_RUNS/$name"
  if [[ -e "$output" ]]; then
    echo "Output already exists: $output" >&2
    return 1
  fi
  flowsis-train-base --detector dfine --detector-model "$FLOWSIS_DETECTOR" \
    --dataset-path "$FLOWSIS_DATASET" --text-embeddings-dir "$FLOWSIS_EMBEDDINGS" \
    --output-dir "$output" --device cuda --image-size 640 \
    --batch-size 2 --num-workers 4 --train-stages head:10 \
    --samples-per-epoch 2235 --video-fraction 0.5 \
    --head-lr 1e-4 --scheduler-type constant_with_warmup --warmup-ratio 0.05 \
    --gradient-clip 1 --text-conditioning frozen --no-detector-amp --amp \
    --rotation-min-degrees -22.5 --rotation-max-degrees 22.5 \
    --use-overlap-augment --overlap-min-overlays 0 --overlap-max-overlays 1 --overlap-p 0.25 \
    --seed 42 "$@"
}

base_run smoke --train-stages head:1 --max-steps 20 \
  --max-validation-images 16 --batch-size 1 --num-workers 0 \
  --no-use-overlap-augment --no-amp

base_run memorize --max-train-images 8 --validation-split train \
  --max-validation-images 8 --samples-per-epoch 8 \
  --train-stages head:50 --max-steps 200 --num-workers 0 --no-amp \
  --no-use-overlap-augment --no-use-rotation-augment \
  --no-use-roi-square-augment --no-use-photometric-augment

base_run mix50 --video-fraction 0.5
base_run stills --video-fraction 0
base_run videos --video-fraction 1
```

Stop after smoke if losses are nonfinite, optimizer steps keep skipping, or
checkpoints fail to save. On memorization, require a clear loss decrease and
improvement in mask overlap on those same eight rows. The first eight rows are
a loader/optimization diagnostic, not a representative class benchmark. The
frozen detector may still classify them incorrectly. Do not use the memorization
checkpoint to initialize the full runs.

The mixture runs each draw 2,235 anchor images per epoch and start from the same
detector, seed and fresh mask head. This holds optimizer updates fixed. Source
balancing is not class balancing. Overlay donors are uniform draws from all
training rows, independent of `--video-fraction`; `stills` and `videos` therefore
describe anchor sources rather than strictly source-exclusive training. For a
strict source-only comparison, also disable synthetic overlap or restrict the
donor dataset in code. More head epochs cannot fix a frozen detector's
classification errors.

```bash
for name in mix50 stills videos; do
  flowsis-evaluate-base --model-path "$FLOWSIS_BASE_RUNS/$name/best" \
    --dataset-path "$FLOWSIS_DATASET" --split validation \
    --output-path "$FLOWSIS_BASE_RUNS/$name/validation.json" \
    --pr-output-dir "$FLOWSIS_BASE_RUNS/$name/validation-pr" \
    --device cuda --no-detector-amp
done
```

Evaluate source-specific quality without changing training code by creating
validation-only views. The loader resolves original media paths before saving;
these temporary views do not copy or alter the source media.

```bash
export FLOWSIS_VIEWS="/tmp/$FLOWSIS_RUN-validation-views"
python - <<'PY'
import os
from pathlib import Path
from datasets import DatasetDict
from flowsis.data.loaders import load_local_dataset

split = load_local_dataset(os.environ['FLOWSIS_DATASET'])['validation']
root = Path(os.environ['FLOWSIS_VIEWS'])
for name, video in [('stills', False), ('videos', True)]:
    rows = split.select([i for i, row in enumerate(split) if bool(row['video_id']) == video])
    DatasetDict(validation=rows).save_to_disk(str(root / name))
PY

for name in mix50 stills videos; do
  for source in stills videos; do
    flowsis-evaluate-base --model-path "$FLOWSIS_BASE_RUNS/$name/best" \
      --dataset-path "$FLOWSIS_VIEWS/$source" --split validation \
      --output-path "$FLOWSIS_BASE_RUNS/$name/validation-$source.json" \
      --device cuda --no-detector-amp
  done
done
```

Compare matched Dice, mask AP50, oracle Dice, box AP50 and per-class support.
Validation has no needleholder, pliers, rongeur or tube examples. Do not treat
their absence as measured class failure. Use `best`, which is selected by
matched Dice subject to detector guardrails, rather than automatically using
`final`.

Only after reviewing these runs, try gentle unfreezing:

```bash
base_run gentle --train-stages head:10,neck:3,all:2 \
  --detector-lr 1e-6 --backbone-lr 1e-7
```

It has a longer training budget than `mix50`; use a fresh `head:15` control if
you need to isolate the benefit of unfreezing from extra updates:

```bash
base_run head15 --train-stages head:15
```

## Temporal smoke and controlled comparisons

After reviewing base validation, set the chosen base. `mix50` below is a default
candidate, not a declared winner. Keep this base fixed across temporal runs.

```bash
export FLOWSIS_BASE="$FLOWSIS_BASE_RUNS/mix50/best"

temporal_run() {
  local name=$1
  shift
  local output="$FLOWSIS_TEMPORAL_RUNS/$name"
  if [[ -e "$output" ]]; then
    echo "Output already exists: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  flowsis-train-temporal --dataset-path "$FLOWSIS_DATASET" \
    --base-model "$FLOWSIS_BASE" --output-dir "$output" \
    --epochs 10 --batches-per-epoch 100 --validation-batches 40 \
    --batch-size 1 --num-workers 4 --image-size 640 \
    --min-clip-length 4 --max-clip-length 8 \
    --strides 1 2 3 --stride-weights 0.7 0.2 0.1 --flow-checkpoint things \
    --learning-rate 1e-4 --gradient-clip 1 --device cuda --seed 42 "$@" \
    2>&1 | tee "$output/console.log"
}

temporal_run smoke --epochs 1 --batches-per-epoch 4 --validation-batches 4 \
  --min-clip-length 4 --max-clip-length 4 --strides 1 --stride-weights 1 \
  --num-workers 0

temporal_run mixed-strides
temporal_run stride1 --strides 1 --stride-weights 1
```

The mixed-stride and stride-one runs have the same update budget, clip-length
range and validation settings. They start fresh from the same base; the base
and flow estimator stay frozen. Short intervals can make eligible video
coverage differ by training geometry. Check all four validation videos appear
in `validation_per_video`.

Once frozen temporal refinement improves sequential held-out quality, test
mask-head adaptation with replay in a separate run:

```bash
temporal_run adapt-head --finetune-mask-head --base-head-lr-scale 0.1 \
  --still-dataset-path "$FLOWSIS_DATASET"
```

Replay from this combined export includes base-valid training images and video
frames. For the adapted run, pair `temporal-best.pt` with `base-best`; frozen
runs pair `temporal-best.pt` with the original `FLOWSIS_BASE`. The temporal CLI
does not provide resume or a full run manifest. Preserve commands/export reports
and console logs. These commands start without temporal AMP.

## Real multi-tool examples and augmentation policy

Audit of this regenerated export on 2026-11-13:

| Split | Real multi-tool images | Capture Sample IDs |
| --- | ---: | --- |
| Train | 20 | 209 (7), 211 (4), 215 (8), 216 (1) |
| Validation | 0 | None |
| Test | 6 | 210 (6) |

All have two annotated objects and are still images. Multi-tool does not by
itself establish physical overlap or occlusion. The five capture groups, rather
than 26 correlated views, are the useful units for split planning. Before tuning
an overlap policy, assign at least one real multi-tool capture group to validation
while preserving another in test. Keep groups intact, verify training class
coverage, and retrain detector/base models using the revised split. An existing
model trained on a newly moved validation group cannot provide clean validation.

The command sheet and sweep script now explicitly set 0–45 degree rotations and
synthetic overlap with zero or one donor image and `p=0.25`. Extra CLI arguments
can override these settings. Memorization disables these augmentations and base
smoke disables overlap. Validation/test use deterministic center-square crops,
without random rotation or synthetic overlap. Temporal training uses a center
crop/resize and random horizontal flip, with no synthetic overlap or rotation;
replay disables augmentation too.

For zero-to-one donors, `p=0.25` means a 25% candidate paste probability.
The donor is another training image, potentially containing multiple tools, so
one donor does not necessarily mean one added object. Masks are pasted at their
existing coordinates; there is no independent random donor rotation or
translation. The complete composite is rotated afterward. Visible masks, boxes
and areas are recomputed; very small visible objects are dropped, and most
severe candidates are rejected. A paste need not actually intersect a base tool.
Therefore candidate frequency is not the measured frequency of accepted physical
overlap. Donors stay in the current training split, preventing validation/test
donor leakage.

Real multi-tool examples currently have **no special priority**. In the base
sampler, equal still-group weights give the four multi-tool groups 4/73 of the
still pool. At a video fraction of 0.5, that is about 2.74% of anchors, or 61
expected draws per 2,235-draw epoch, before augmentation and cropping. Synthetic
pasting can also alter these real examples. The detector instead shuffles rows
uniformly, giving 20 real multi-tool anchors per full epoch; it does not use the
base sampler's source/group balancing. Synthetic objects retain their annotation
confidence, and neither trainer currently applies a real-versus-synthetic loss
scale. Base mask losses average objects within images, so two objects do not
automatically double an image's weight.

To enforce greater priority for real interactions, a subsequent code change
should preserve real multi-tool anchors from extra synthetic pastes and add a
modest real-group sampling boost (for example a 3× starting ablation), normalized
within the still pool so `--video-fraction` remains meaningful. A sampling boost
should be optional, default to one, and be compared on held-out real groups.
Object count is only a proxy for real overlap; use an explicit reviewed overlap
flag when available. If adding loss scaling instead, apply an image-level scale:
multiplying every object's annotation confidence in an image by the same factor
cancels in the current normalized mask loss. These prioritization changes have
**not** been implemented in the current code or commands.

After fixing validation support, a simple augmentation-rate comparison is:

```bash
base_run overlap-off --no-use-overlap-augment
base_run overlap25 --overlap-p 0.25
base_run overlap50 --overlap-p 0.5
```

Keep the detector, seed, anchor mixture and update budget fixed. Report real
multi-tool and single-tool validation separately, together with overall quality;
synthetic-only validation cannot establish performance on real overlaps.

## Code needed for evaluation

Training requires no changes to the current checkout, which already contains
the shared-dataset loader and dynamic sampling changes. Keep those changes for
these experiments; an older committed revision may lack them.

The existing temporal objective uses ground-truth query/track matching. Its loss
is a useful training diagnostic but does not measure runtime selection/association.
Before claiming temporal improvement, add a sequential quality evaluator that
compares base and temporal output on identical held-out frames, resets state at
interval boundaries, and reports per-video overlap, missed observations, target
selection and flow-aligned flicker. Report oracle-matched refinement separately
from actual runtime selection. The existing temporal profiler measures speed,
not segmentation quality. No command in this document substitutes for that
missing evaluation.

Keep test untouched during recipe selection. Run finalists with seeds 42, 43
and 44 in new directories. Empty-frame rejection and severe-occlusion recovery
remain unmeasured because this export lacks supporting examples.
