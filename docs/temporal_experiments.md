# Overnight temporal verification suite

Prepared with Codex assistance. Reuses `outputs/overnight/dev-v4` and its frozen
base/refiner pair. It evaluates **validation only**; no source checkpoint is
modified and no detector or base retraining is performed.

## Launch

```bash
conda activate flowsis
cd /home/pricyspark/flowsis
python -m experiments.run_temporal_experiments --train-candidates --dry-run

mkdir -p outputs/temporal
nohup python -u -m experiments.run_temporal_experiments \
  --run-dir outputs/overnight/dev-v4 \
  --output-dir outputs/temporal/dev-v4-experiments \
  --train-candidates \
  > outputs/temporal/dev-v4-experiments-launch.log 2>&1 < /dev/null &
```

Omit `--train-candidates` for checkpoint-only experiments. Keep the computer awake.
There is no fixed runtime guarantee: jobs run serially, with repeated checkpoint
loading and full evaluation rather than competing GPU processes.

To continue after interruption, repeat the same command with `--resume`. Completed
jobs are skipped after checking required outputs; incomplete jobs restart from
scratch. An interrupted *training job* does not continue from its last epoch.
Changes to configuration, input model hashes, dataset fingerprint, validation asset
size/mtime or relevant source hashes require a new output directory. Keep evaluated
source assets, code and input models unchanged during the suite; final checks reject
changes during execution. If one job fails, independent jobs
continue; final exit status is nonzero and the report lists failures. Add
`--fail-fast` to stop immediately.

## Experiments

Default checkpoint-only plan: 39 stages. With `--train-candidates`: 53 stages, including two fresh refiner training jobs. The original detector, bases and refiner remain untouched.

- Three policies × identity/learned mask refinement × lengths 4, 8, 16, 32, 64,
  full = 36 runtime jobs.
- Two full-interval oracle jobs: identity versus learned mask correction with
  ground-truth association.
- One stateless evaluation of the stronger saved standalone base, using the same
  top-score single-target metric as runtime, rather than comparing unrelated
  whole-dataset Dice averages.

Policies:

1. `current`: existing confirmation, selection and initial refinement.
2. `fallback`: when the selector has no confirmed choice, return the strongest
   accepted current base mask while tracking continues internally. This does not
   promote tentative tracks, change selector history or use ground truth.
3. `history`: fallback plus bypass refinement for a track without real prior
   history. Seed its state with base logits and zero hidden state, then refine on
   subsequent observations.

These are opt-in switches; normal deployment and existing evaluator defaults keep
their previous behavior. Identity disables propagation/residual mask correction;
other model parameters and tracker/selection remain. It is not an ablation of all
learned components.

`--all-frames` removes the old six-chunks-per-video sampling cap and retains
single-frame tail chunks. Every runtime condition therefore scores the same
eligible frames, with state reset only at snippet, gap and interval boundaries.
`full` is passed as `--clip-length 0 --all-frames`, representing each complete
contiguous exported interval. This does not join disjoint splits or gaps.

With `--train-candidates`, there are 53 stages total. Two new refiners start from
fresh initialization on the same frozen base, seed and update budget: maximum
training length 8 versus 16, each default 16 epochs × 75 batches. Both select
checkpoints using the same 8-frame validation geometry; longer training does not
silently remove short validation videos. Each gets six full-interval policy ×
identity/learned evaluations. The saved eight-epoch model is also retained as a
reference, but differs in training budget. No synthetic association-error training
is introduced yet; use diagnostics to design that later experiment.

## Morning review

- `REPORT.md`: results by policy, length and refiner; strongest standalone base;
  stage failures and interpretation limits.
- `comparisons.csv`: frame-/video-weighted deltas, post-start deltas, low-Dice
  frames/runs, and time to a correct prediction (Dice >=0.5). Delay is measured
  from the first scored target observation in each snippet/target; unsuccessful
  snippets are counted separately rather than hidden from the median.
- `JOB/frames.csv`, `summary.json`, `frame_position.png`: all scored frames,
  position counts and per-recording outcomes.
- `JOB/runtime_diagnostics.csv`: every input frame, selected query/class/track,
  confidence, fallback use, image path and tracker state. Join on video ID,
  clip_start and frame_index when diagnosing the previously severe mallet interval.
- `experiment_config.json`, `commands.jsonl`, `state.json`, individual stage logs:
  provenance, commands and resume status.

Compare current versus fallback to isolate confirmation cost. Compare fallback
versus history to isolate new-track initialization. Compare identity versus learned
within the same policy/length to isolate mask correction. Compare lengths on the
identical scored frames. The reporter rejects repeated/mismatched frame cohorts
and large changes in paired stateless baseline scores.

Optional training uses 16 epochs with 75 updates per epoch by default for both
max-8 and max-16 recipes, with the same 8-frame validation geometry. This matches
optimizer updates, not frame count or compute: longer training clips expose more
frames and can require more GPU memory. A positive result would justify a further
frame-budget-controlled comparison.

The current validation data has no empty scenes. Empty-scene counts are reported
when annotations support them; zero coverage is **not evidence of low false-positive
rates**. Runtime Dice remains a single-visible-target evaluation; it does not yet
measure complete multi-object tracking or unobserved physical-tool generalization.
Longer frame positions have fewer snippets, so inspect counts. Multiple seeds and
an untouched test evaluation should follow any promising validation result.
Stage times include decoding and duplicate baseline inference; they are not a
standalone deployment FPS benchmark.

## Quick smoke or a smaller sweep

```bash
python -m experiments.run_temporal_experiments \
  --output-dir outputs/temporal/dev-v4-policy-smoke \
  --lengths 4 full --max-videos 1 --max-frames-per-video 3 --skip-oracle
```

This is a small runtime smoke, not the full benchmark; do not combine subset limits
with `--train-candidates`. Choose `--device cpu` if CUDA is unavailable. Override
`--training-epochs`, `--training-batches`, `--num-workers`, `--seed` or `--lengths`
when needed. The dataset defaults to the completed run's provenance; use
`--dataset-path` only for an intentional alternate compatible export.


## Implementation verification

Targeted regression tests cover runtime fallback without premature confirmation,
valid-history initialization/reset, empty detections, exhaustive coverage including
singleton tails, fixed validation geometry for longer training, comparable frame
cohorts, startup/failure statistics, failure reporting and resume behavior. A
real-checkpoint CPU smoke completed all seven policies/control cases on a three-frame
subset in outputs/temporal/dev-v4-experiment-smoke-v2. Its scores verify plumbing,
not model effectiveness. The full GPU matrix has not been launched during setup.
