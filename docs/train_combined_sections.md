# Training experiments for the combined export

Prepared with Codex assistance on 2026-11-13. Commands run from the FlowSIS
repository in the `flowsis` environment. These are proposed experiments, not
measured training results. Use a fresh output directory for every run.

## Dataset findings

The inspected export is `../flowsis-data/exports/combined-sections`:

| Split | Stills | Video frames | Labels represented | Videos |
| --- | ---: | ---: | ---: | ---: |
| Train | 560 | 1,674 | 25 | 9 |
| Validation | 119 | 251 | 13 | 4 |
| Test | 118 | 246 | 14 | 4 |

The 25 labels combine tool name and class. Historical checkpoints and
`data/tool-class/text-embeddings` use nine coarse labels and cannot initialize
the base model directly on this export. Base construction requires exact
agreement between dataset, detector and prompt-bank class order. Temporal
training also checks the base checkpoint's class order.

Train contains 26 multi-tool images; validation/test contain none. There are no
empty frames. Train visibility is 2,258 clear objects and two partial objects;
all validation/test objects are clear. `adsonneedle forceps` and
`unnamed needleholder` have only one training object each. Inspect these names
and their source annotations before deciding whether they are intended classes.
Changing splits cannot create independent examples of a singleton class.

The four validation videos cover bandage scissors (24 objects),
microvascularcoupler forceps (30), mprsurgical mallet (47), and crile forceps
(150). Video-pooled metrics will be dominated by crile forceps. Report equal
per-video averages alongside pooled results.

Section splitting places portions of the same four videos in all three splits.
An eight-frame gap prevents immediate adjacency at the corresponding split
transitions, but does not establish independence of tools, backgrounds or
recordings. Treat this export as a within-recording development benchmark.
Keep its split fixed across detector, base, temporal training and replay.

The report also excludes 6,531 video annotations outside each video's first
contiguous accepted run. Inspect those exclusions before investing in annotation:
they may include useful data, but adding disjoint runs requires preserving
interval boundaries and reviewing the builder's selection policy.

## Choose the benchmark before training

Use coarse labels for the first experiments, preserving fine names as metadata.
A read-only source-manifest check found that fine-label tool grouping assigns
all nine videos to training and leaves test empty. It also includes stills
without Sample IDs omitted by section mode, expanding fine labels from 25 to 28.
Simply changing the fine-label export to whole-tool splitting is insufficient.

Keep two separate coarse exports. Start development with the same section
parameters as the current export, so changes in label granularity are easy to
understand. The stronger whole-tool benchmark has limited class coverage:

| Source-manifest dry split, coarse/tool/seed 42 | Rows | Videos | Represented labels |
| --- | ---: | ---: | --- |
| Train | 2,316 | 6 | All 11 |
| Validation | 482 | 1 | forceps, scissors |
| Test | 503 | 2 | osteotome, scissors |

This is an independent-tool check for the represented labels, not an eleven-class
benchmark. Coarse session splitting at seed 42 also has limited support:
validation represents forceps/pliers/scissors and test only forceps. Choosing
session grouping does not solve class scarcity.

Whole-tool splitting groups connected tool/sample/video identities. Session
splitting answers a different question about capture conditions; verify session
IDs before using it. Sample splitting keeps whole recordings together but can
share physical tools across splits.

```bash
conda activate flowsis
python ../flowsis-data/build_dataset.py \
  --root ../flowsis-data/data \
  --output-path ../flowsis-data/exports/combined-sections-coarse \
  --include-videos --label-fields tool_class \
  --video-sections 3 --split-gap 8 --min-interval-frames 8 \
  --short-video-policy train \
  --validation-size 0.15 --test-size 0.15 --seed 42

python ../flowsis-data/build_dataset.py \
  --root ../flowsis-data/data \
  --output-path ../flowsis-data/exports/combined-tools-coarse \
  --include-videos --video-sections 0 --split-by tool \
  --label-fields tool_class \
  --validation-size 0.15 --test-size 0.15 --seed 42
```

Inspect `build_report.json` before training: require usable temporal intervals in
both holdouts, train coverage of every label, and sufficient independent groups
for the claimed class comparisons. Requested fractions do not override grouping
or training class coverage. If fine labels have only one physical tool, a
tool-held-out closed-set evaluation for those labels is impossible with current
data. Collect another physical instance or limit the claim to represented labels.
Do not select a seed by model accuracy. Choose a split using coverage/group
constraints and freeze it before model comparisons.

The coarse setting reduces label fragmentation, but the new data includes `tube` and
`needleholder`, so do not assume compatibility with the old nine-class bank.
Verify the resulting vocabulary; train a matching detector and prompt bank.

## 1. Establish a matching detector and prompt bank

Use the coarse section export for the first development experiment. Fine tool
names remain in object metadata; coarse training does not require losing them.
For the independent-tool benchmark, repeat the complete detector/base/temporal
workflow on `combined-tools-coarse` with fresh output directories and a fresh
generic detector initializer. A section-trained checkpoint has already seen
tools held out by the whole-tool split and cannot serve as its clean initializer.
Review each export's build report before training.

```bash
FLOWSIS_DATASET=../flowsis-data/exports/combined-sections-coarse
FLOWSIS_PROMPTS=configs/prompt_suites/combined-coarse-v1.json
FLOWSIS_EMBEDDINGS=data/combined-coarse/text-embeddings
FLOWSIS_BASE=outputs/base/combined-sections-coarse-head10-s42

flowsis-author-text-prompts \
  --dataset-path "$FLOWSIS_DATASET" --output-path "$FLOWSIS_PROMPTS"
# Complete/review all visual descriptions before embedding.
flowsis-embed-text-prompts \
  --dataset-path "$FLOWSIS_DATASET" --prompt-suite "$FLOWSIS_PROMPTS" \
  --output-dir "$FLOWSIS_EMBEDDINGS" \
  --model google/siglip2-base-patch16-224 --device cuda

flowsis-train-detector --detector dfine \
  --model ustc-community/dfine-medium-obj365 \
  --dataset-path "$FLOWSIS_DATASET" \
  --output-dir outputs/detectors/combined-sections-coarse20-s42 \
  --epochs 20 --batch-size 8 --image-size 640 --num-workers 4 \
  --lr 1e-4 --backbone-lr 1e-5 --max-grad-norm 1 --device cuda --seed 42
```

This detector recipe is a starting experiment: the smaller backbone LR differs
from historical training. D-FINE stays FP32 by omitting detector `--amp`.
For a detector plumbing smoke run, use another output directory and add
`--epochs 1 --max-steps 20 --num-workers 0 --sanity-decode` (replace duplicate
flags). For a memorization diagnostic, disable augmentation and use
`--overfit-single-batch --epochs 100`; this performs one update per epoch.
Evaluate that diagnostic on its training batch, not the holdout.

The detector CLI resets incompatible classification tensors when constructing a
model for a new vocabulary. Use the generic pretrained initializer above for the
new benchmark: an old FlowSIS checkpoint may already have trained on physical
tools now assigned to validation/test. Only test an old detector as a transfer
initializer after verifying its historical training data excludes the new
holdout groups. Its old class accuracy does not carry over to a changed
vocabulary. Do not use `--resume-from` to change datasets or vocabularies.

Review saved detector checkpoints on validation with AP50/AP75 and per-class
precision/recall; validation loss alone need not choose the best detector.
For each candidate, set its actual path and run:

```bash
FLOWSIS_DETECTOR=outputs/detectors/combined-sections-coarse20-s42/final
flowsis-evaluate-detector --detector dfine --model "$FLOWSIS_DETECTOR" \
  --dataset-path "$FLOWSIS_DATASET" --split validation \
  --output-dir outputs/evaluation/combined-detector-final-ap50 \
  --iou-threshold 0.5 --device cuda
```

Repeat with `--iou-threshold 0.75` and a distinct evaluation output directory.
Then set `FLOWSIS_DETECTOR` to the chosen checkpoint. Report class support and
macro averages over represented classes; absent-class zeros cannot establish
performance on those classes.

## 2. Smoke-test and train the base

Run the complete loader with bounded optimization; capping to the first few
rows could silently test only still images.

```bash
flowsis-train-base --detector dfine --detector-model "$FLOWSIS_DETECTOR" \
  --dataset-path "$FLOWSIS_DATASET" \
  --text-embeddings-dir "$FLOWSIS_EMBEDDINGS" \
  --output-dir "${FLOWSIS_BASE}-smoke" --device cuda \
  --train-stages head:1 --max-steps 20 --max-validation-images 16 \
  --video-fraction 0.5 --batch-size 1 --num-workers 0 \
  --no-use-overlap-augment --no-detector-amp --no-amp

flowsis-train-base --detector dfine --detector-model "$FLOWSIS_DETECTOR" \
  --dataset-path "$FLOWSIS_DATASET" \
  --text-embeddings-dir "$FLOWSIS_EMBEDDINGS" \
  --output-dir "$FLOWSIS_BASE" --device cuda \
  --train-stages head:10 --image-size 640 --batch-size 2 --num-workers 4 \
  --video-fraction 0.5 \
  --head-lr 1e-4 --scheduler-type constant_with_warmup --warmup-ratio 0.05 \
  --gradient-clip 1 --text-conditioning frozen --no-detector-amp --amp --seed 42

flowsis-evaluate-base --model-path "$FLOWSIS_BASE/best" \
  --dataset-path "$FLOWSIS_DATASET" --split validation \
  --output-path "$FLOWSIS_BASE/validation.json" \
  --pr-output-dir "$FLOWSIS_BASE/validation-pr" --device cuda --no-detector-amp
```

Between smoke and full training, run a small memorization diagnostic in another
directory: take eight training rows with `--max-train-images 8`, use
`--validation-split train --max-validation-images 8`,
`--samples-per-epoch 8 --train-stages head:50 --max-steps 200`, and disable overlap,
rotation, ROI and photometric augmentation with their `--no-use-...-augment`
flags. Keep the detector frozen. Mask loss should fall and oracle mask quality
should improve markedly on these same rows. Failure suggests a loader, alignment,
optimization or capacity problem worth resolving before full training. Correct
classification is still limited by the frozen detector. This is a training-set
diagnostic and cannot estimate generalization.

The default epoch draw count is the eligible training-row count. Run `--video-fraction 0`
and `1` with distinct output directories as source-mixture ablations after the
0.5 baseline. Hold the draw count, optimizer updates and other settings fixed.
Report still/video validation separately; the current base evaluator combines
them, so source-specific evaluation requires selecting views or extending it.
Avoid assuming that 0.5 balances classes: it balances sources, then samples or
videos within each source. Rare classes can still receive few draws.

Select using matched Dice and mask AP50 together with per-class results, box AP
and detector guardrails. Oracle Dice diagnoses mask shape independently of
correct top-class/confidence acceptance. Historical frozen runs reached about
95% oracle Dice but only 69% matched Dice. More mask epochs alone did not resolve
classification/query-selection limitations.

Only after the frozen baseline, compare a fresh run with
`--train-stages head:10,neck:3,all:2 --detector-lr 1e-6 --backbone-lr 1e-7`.
The original 1e-5/1e-6 unfreezing failed guardrails at every unfrozen epoch;
gentler unfreezing gave a small, volatile peak improvement. Keep guardrails and
compare `best`, not automatically `final`. These old results are methodological
evidence, not numerical targets for the new taxonomy and split.

## 3. Train temporal refinement with the selected base frozen

```bash
mkdir -p outputs/temporal/combined-sections-coarse-smoke
set -o pipefail
flowsis-train-temporal --dataset-path "$FLOWSIS_DATASET" \
  --base-model "$FLOWSIS_BASE/best" \
  --output-dir outputs/temporal/combined-sections-coarse-smoke \
  --epochs 1 --batches-per-epoch 4 --validation-batches 4 \
  --min-clip-length 4 --max-clip-length 4 --strides 1 --stride-weights 1 \
  --batch-size 1 --num-workers 0 --image-size 640 --device cuda --flow-checkpoint things \
  2>&1 | tee outputs/temporal/combined-sections-coarse-smoke/console.log

mkdir -p outputs/temporal/combined-sections-coarse-frozen-s42
flowsis-train-temporal --dataset-path "$FLOWSIS_DATASET" \
  --base-model "$FLOWSIS_BASE/best" \
  --output-dir outputs/temporal/combined-sections-coarse-frozen-s42 \
  --epochs 10 --batches-per-epoch 100 --validation-batches 40 \
  --min-clip-length 4 --max-clip-length 8 \
  --strides 1 2 3 --stride-weights 0.7 0.2 0.1 \
  --batch-size 1 --num-workers 4 --image-size 640 \
  --learning-rate 1e-4 --gradient-clip 1 --device cuda --seed 42 --flow-checkpoint things \
  2>&1 | tee outputs/temporal/combined-sections-coarse-frozen-s42/console.log
```

Start without temporal AMP to establish numerical behavior. Eight frames and
all detector queries can use substantial memory even with a frozen base. Reduce
clip length if necessary, recording it as a changed experiment. The suggested
100 training batches are a bounded pilot, not a dataset-defined epoch or an
established optimal budget. Extend only after inspecting learning curves.

Validation uses max-length clips at stride one and deterministic starts. Forty
batches cover the original four eligible validation videos; check
`validation_per_video` after a rebuild. Longer strides and lengths have fewer
eligible videos, so training coverage can depend on clip geometry. The one-frame
video cannot contribute a temporal clip. The current CLI has no flow-cache
flags, resumable optimizer state or base-style run manifest; retain the command,
source revision, export report and console log. Best is `temporal-best.pt` by
validation loss; `temporal.pt` is the latest epoch.

## 4. Experiments that answer whether temporal training helps

Compare on exactly the same held-out frames and preprocessing:

| Experiment | Question | Required evidence |
| --- | --- | --- |
| Base alone vs frozen-base temporal | Does recurrence improve masks? | Paired Dice/IoU, boundary F1, missed frames and per-video deltas |
| Stride 1 vs mixed 1/2/3 training | Does motion sampling help? | Fixed stride-1 validation plus separate stride-2/3 evaluation |
| Short clips vs full held-out intervals | Does recurrent state drift? | Reset at each interval boundary; errors over elapsed time |
| Frozen base vs adapted mask head with replay | Does adaptation help without losing still quality? | Temporal deltas plus the same still validation suite |

The temporal training objective uses ground-truth query matching/track identity.
Its validation loss does **not** measure runtime detection, selection or track
association. This repository has a temporal FPS profiler, but no corresponding
end-to-end temporal quality evaluation CLI. A sequential quality evaluator is
needed before declaring temporal improvement; report oracle-matched refinement
and actual runtime selection separately. Runtime currently selects a target, so
define that target explicitly for multi-tool frames rather than comparing it to
all-instance base metrics. Measure FPS separately from quality.

For flicker, compare flow-aligned predictions in annotated visible regions
alongside accuracy; constant or empty masks can look artificially stable.
Aggregate by recording and report variation across recordings. Overlapping clips
are correlated and must not be treated as independent experimental replicates.

If the frozen temporal branch improves held-out quality, repeat from the same
base with `--finetune-mask-head --base-head-lr-scale 0.1
--still-dataset-path "$FLOWSIS_DATASET"` and a fresh output directory. Replay uses
base-valid training rows from this path, including video rows when present.
Pair `temporal-best.pt` with its saved `base-best` for that experiment.

Repeat finalists with seeds 42, 43 and 44 after choosing the recipe on validation.
Use test only after model/threshold selection. Add independently reviewed empty,
multi-tool, partial/severe/occluded and reappearance sequences before claiming
false-positive control or occlusion recovery; splitting current clear footage
cannot supply those cases.

## Checks performed in this review

- `python -m pytest -q tests/test_dataset_mixture.py
  tests/test_temporal_dataset.py tests/test_base_training.py tests/test_temporal.py`:
  **26 passed**.
- Checked existence of all referenced image and mask paths: 4,468 in train,
  740 in validation and 728 in test (unique paths counted within each split).
- No identical image paths, still Sample IDs or sequence IDs across splits;
  the same four recording IDs occur in all splits as expected for section mode.
- Decoded one eight-frame clip per eligible video per split at 64 pixels:
  eight training videos and four each in validation/test. Representative
  full-size still/video images and mask dimensions also agreed in each split.
- No GPU training or inference evaluation was run. CUDA was unavailable in this
  session. File existence and sampled decoding do not verify every mask pixel or
  all image decodes.
