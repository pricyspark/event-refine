> Historical notes for the original coarse-v2 runner. Commands reflect its previous CLI; use [the current workflow](overnight_training.md) for new runs.

# Overnight detector-to-base-to-temporal run

Prepared with Codex assistance on 2026-11-13. This workflow preserves the current
detector result and its dataset split. It does not restart detector training.

The completed `coarse-v2` run is reviewed in
`outputs/overnight/coarse-v2/REVIEW.md`. Its detector/base results are usable,
but temporal training had an input scaling/aliasing error. The code now converts
unit RGB clips to a separate raw uint8 detector batch and preserves flow frames.
Use `bash experiments/retrain_temporal_coarse_v2.sh` to rerun only the temporal
stages against the saved base; the original temporal checkpoint needs replacement.
The earlier finite-backprop smoke tests missed this error. New tests check pixel
parity with runtime preprocessing and input immutability.

## Progress inspected

The coarse-sections detector completed all 20 epochs:

| Epoch, one-based | Training loss | Validation loss |
| --- | ---: | ---: |
| 1 | 40.3962 | 5.0791 |
| 5 | 13.2724 | 1.3063 |
| 10 | 11.1707 | 0.9293 |
| 11 | 10.9182 | 0.8783 |
| 12 | — | 0.8599 |
| 13 | — | 0.8633 |
| 14 | 10.7399 | 0.8336 |
| 15 | 10.1282 | 0.7965 |
| 16 | 10.0141 | 0.8014 |
| 17 | 9.9751 | 0.7820 |
| 18 | 10.1008 | 0.7507 |
| 19 | 9.9721 | 0.7735 |
| 20 | 9.9924 | 0.7467 |

Validation loss is down 85.3% from epoch 1, and about 6.3% from epoch 15.
Classification-related validation VFL fell 1.7718 → 0.1793; main box loss fell
0.03971 → 0.02336 and GIoU 0.06921 → 0.04237. Epoch 20 is the lowest-loss
checkpoint; epoch 18 is only about 0.53% worse. Late training loss is near a
plateau, without a sustained rising-validation-loss trend. No saved AP evaluation
was present at inspection, so these are optimization findings, not measured AP gains.
Train and validation loss totals include different terms/augmentation and should
not be compared as directly equivalent quantities.

The final checkpoint completion marker is present. SHA-256 hashes confirm
`checkpoint-005600/model.safetensors` and `final/model.safetensors` are identical.
The runner therefore compares epoch 20 (`checkpoint-005600`) against the
next-lowest-loss epoch 18 (`checkpoint-005040`), rather than evaluating epoch 20
twice. It automatically skips waiting because detector training is complete.
The base/temporal training budgets remain unchanged; detector loss alone does
not justify changing their learning rates or schedules.

## Quick video-loading smoke now

In the activated environment, this uses CPU and does not start another GPU job:

```bash
conda activate flowsis
cd /home/pricyspark/flowsis
python -m experiments.run_overnight --data-smoke-only --smoke-image-size 128
```

It reads actual saved frames/masks from train and validation, testing lengths
4 and 8 at stride 1 and length 8 at stride 3. The loader respects interval
boundaries. It is a decoding/supervision check at reduced resolution; the
overnight workflow separately performs complete training smokes at 640 pixels.

The review passed this loader check at 64 pixels, a cached pretrained RAFT-Small
CPU forward at 128 pixels, and a temporal branch forward/backward. Subsequently,
the user completed four- and eight-frame CUDA training smokes at 640 pixels on
an RTX 5070 (`temporal-smoke-DOPPzQ` and `temporal-smoke-mNuetk`). Both saved
finite weights, updated the initially zero residual head, and validated all
four videos. This confirms those smoke configurations fit and backpropagate;
it does not establish segmentation quality or full-run performance.

The smoke output's `timm.models.layers` deprecation is a dependency import notice.
PTLFlow also imports other model families that emit `xFormers not available` and
`alt_cuda_corr is not compiled` messages. The selected RAFT-Small defaults to
standard correlation (`alternate_corr=False`), so these messages do not indicate
that its required computation failed. No dependency installation is needed for
the validated configuration.

The trainer's warning about converting a tensor with `requires_grad=True` to a
scalar was fixed: metrics are detached and moved to CPU scalars before storing
epoch history. This preserves the actual optimization loss/backpropagation and
avoids retaining its graph in logging history. Third-party notices remain visible.

The eight-update base bootstrap has very poor mask Dice and is for wiring tests.
Its final logged learning rate of zero reflects the default linear scheduler
reaching the eight-step budget; it does not mean all updates used zero LR.
Full base training must establish useful masks before temporal quality is judged.

## Queue the overnight run

Preview the commands without waiting for the detector or starting training:

```bash
python -m experiments.run_overnight --dry-run
```

Then launch once, using a fresh output directory:

```bash
mkdir -p outputs/overnight
nohup python -u -m experiments.run_overnight \
  --detector-run outputs/detectors/coarse-sections-v1 \
  --dataset-path ../flowsis-data/exports/combined-sections-coarse \
  --embeddings-dir data/combined-coarse/text-embeddings \
  --output-dir outputs/overnight/coarse-v1 \
  > outputs/overnight/coarse-v1-launch.log 2>&1 < /dev/null &
echo $!
```

Watch orchestration and the current stage's individual log:

```bash
tail -f outputs/overnight/coarse-v1-launch.log
# Once the corresponding stage starts:
tail -f outputs/overnight/coarse-v1/temporal-main.log
```

For this completed detector, execution starts directly with AP evaluation. If
used with an unfinished detector, the queue waits up to four hours for ALL requested detector validation epochs
and the final checkpoint's completion marker. It never kills/resumes the
detector or starts another GPU phase while waiting. A stalled or failed detector
causes a timeout rather than training from incomplete weights. Reused/restarted
log sequences are rejected. The output directory must be new.

The required sequence is:

1. Evaluate epoch 20 and epoch 18 on validation at AP50/AP75. Choose the highest
   macro AP50 over represented validation classes. Generally, the runner compares
   the lowest-loss checkpoint with final; when these represent the same epoch,
   it substitutes the next-lowest-loss checkpoint. This does not exhaustively
   search every epoch.
2. Run an eight-update base smoke, then temporal smokes with four- and
   eight-frame clips. Smoke checkpoints are not used to initialize full training.
3. Train the base mask head for 10 epochs with the detector frozen, 50/50
   still/video anchors, synthetic overlap probability 0.25, and rotation 0–45°.
   D-FINE stays FP32 and the mask head uses AMP. Save guardrail-qualified `best`.
4. Train temporal refinement for 8 epochs × 75 batches, batch size 1,
   lengths 4–8 and stride weights 0.7/0.2/0.1 for strides 1/2/3. Base and flow
   stay frozen; temporal AMP is disabled initially. Validation has up to 40
   batches per epoch, covering all four eligible validation recordings.
5. Evaluate base versus temporal on paired held-out frames with 16-frame reset
   windows and whole exported intervals (up to 256 frames). Evaluate the base
   on separate still/video validation views too.
6. If elapsed time is below eight hours, run an optional equal-budget base
   control with synthetic overlap disabled. Its detector initialization still
   comes from overlap-augmented detector training. This control is for learning
   about base augmentation, not isolating all effects of synthetic data.

The temporal checkpoint is paired with the main base. An optional control
winning a different base metric does not change that trained pair. The report
lists the pairing explicitly. Required work runs to completion; eight hours is
only the cutoff for STARTING the optional control, not a hard runtime limit.
This is a proposed training budget, not a guarantee of morning completion or
improvement. To omit the optional control, add `--skip-control`. To shorten the
required pilot, override e.g. `--base-epochs 6 --temporal-epochs 5
--temporal-batches 50` in a fresh directory.

Crucially, every temporal command specifies `--flow-checkpoint things`. The
installed PTLFlow does not load pretrained weights when that argument is None.
The RAFT-Small things checkpoint is already in the local Torch cache; a local
path can also be supplied via `--flow-checkpoint`.

## Morning artifacts

Start with `outputs/overnight/coarse-v1/REPORT.md` and `summary.json`. The report
is updated between phases, including on failure, so it identifies the last
completed work and retained models. Commands, source hashes, embedding hashes,
dataset build report and individual logs are retained.

The primary artifacts are:

- `base/overlap25/best/`: the selected base checkpoint.
- `temporal/main/temporal-best.pt`: temporal checkpoint paired with that base.
- `temporal/eval-clips16/` and `temporal/eval-intervals/`: paired evaluation
  `frames.csv`, `summary.json` and `frame_position.png`.
- `base/*/validation*.json`: pooled, still-only and video-only base quality.
- `detector/*/metrics.json`: detector AP, class support and PR curves.

Each temporal evaluation reports Dice/IoU, paired improvement, improved/worsened
fractions and descriptive delta standard deviation, by frame position and by
video. Equal-video mean delta is included so the longest recording does not
automatically dominate the conclusion. The longer-position curve can have fewer
supporting videos; inspect its counts and coverage.

Two evaluation modes answer different questions:

- `oracle_matched`: GT matches queries/tracks, isolating mask refinement. Frame
  0 is seeded from base logits, as in training; no refinement occurs at that
  position. Positive later-frame paired deltas demonstrate refinement under
  these matching assumptions.
- `runtime_single_target`: stateless highest-confidence base prediction versus
  the actual temporal tracker/selector on the same frames. Only frames with one
  visible mask-valid target are scored. Missing or wrong-class selections score
  zero. The tracker requires confirmation, so frame 0 may have no temporal
  selection even when the base is correct. This warm-up is deliberately included.

Temporal improvement is not guaranteed to grow monotonically with frame number.
Compare each temporal prediction to the base on the SAME frame, rather than
interpreting raw frame-0 versus frame-7 accuracy as a causal temporal effect.
Frames within a recording are correlated; these descriptive statistics do not
claim independent-frame significance. State resets at every exported interval
boundary, preventing held-out gaps from being silently bridged.

The test split is not evaluated or used for model selection. This remains a
within-recording benchmark with no real multi-tool validation examples or empty
frames, and very limited visibility/occlusion diversity. The current real-first
sampling/augmentation proposal has not been implemented; this run establishes a
reproducible baseline without changing the active detector's training policy.
