# General overnight baseline recipe

Prepared with Codex assistance. Run from the FlowSIS repository with the training
environment active. The runner consumes a **finished export**; it does not rebuild
or change the dataset. Old coarse-v2 notes are in
[the historical run notes](overnight_coarse_v2_history.md).

## Check, then launch

```bash
conda activate flowsis
cd /home/pricyspark/flowsis
python -m experiments.run_overnight \
  --dataset-path ../flowsis-data/exports/combined-tool-sample-coarse-v3 \
  --require-temporal --dry-run
```

The regenerated v3 export now has 11 consistent canonical classes, including
`needle_holder`, and passes the overnight dry-run preflight. Future exports with
conflicting separator/case variants are rejected; the runner does not silently
merge classes or rewrite saved data.

After that preflight passes:

```bash
mkdir -p outputs/overnight
nohup python -u -m experiments.run_overnight \
  --dataset-path ../flowsis-data/exports/combined-tool-sample-coarse-v3 \
  --output-dir outputs/overnight/baseline-v3 \
  --require-temporal \
  > outputs/overnight/baseline-v3-launch.log 2>&1 < /dev/null &
```

Use a fresh output directory for each recipe. Omit `--output-dir` to generate a
timestamped name. This orchestrator does not resume interrupted runs automatically;
completed training checkpoints remain available for individual trainer resume or
explicit detector reuse. Keep the computer awake; `nohup` survives terminal closure,
not system sleep. Initial model/flow downloads require network access unless cached.

## What runs

1. Audit taxonomy, local assets, base training/validation coverage, split sharing,
   reserved Tool IDs, and supervised clip geometries. Save the export fingerprint,
   coverage warnings, source hashes and copied split manifests.
2. Prepare compatible SigLIP2 embeddings. Without supplied prompts, embed canonical
   labels and pipeline templates only; no LLM/API call or fabricated visual
   descriptions. `--prompt-suite reviewed.json` enables all three reviewed prompt
   types. `--embeddings-dir path` instead validates an existing bank. Choose one.
3. Train a fresh detector: default D-FINE, 20 epochs, batch 2, backbone LR 1e-5,
   other LR 1e-4, adaptive warm-up, gradient clipping, FP32. Compare the best-loss
   checkpoint with a distinct final/runner-up checkpoint using represented-class
   macro AP50 on validation. Also save AP75 and PR curves.
4. Run an eight-update base smoke and short temporal backprop smokes before the main
   base run. Temporal clip length adapts to 8, 4, or 2 according to actual supervision
   in both train and validation; whole intervals remain respected.
5. Train the base mask head for 10 epochs with the detector frozen, synthetic
   overlap p=0.25, rotation -22.5° to +22.5°, and AMP. Select its guardrail-qualified
   best matched-Dice checkpoint and evaluate it on validation.
6. Verify actual saved-base mask-logit parity between runtime and temporal input
   paths, with unchanged normalized clip inputs; fail before temporal training if
   the semantic preprocessing check fails. Train a fresh corrected temporal refiner for 8 epochs × 75 batches by default,
   batch 1, LR 1e-4, FP32. Increase validation/training batches if necessary to cover
   eligible validation videos. Validate checkpoint preprocessing marker and finite
   weights. Main temporal training keeps its base fixed.
7. Evaluate paired base/temporal results with oracle matching and actual runtime
   tracking: short chunks, chunks up to 256 frames, and an identity-mask-refinement
   control. The latter zeros propagation/residual strengths while retaining the
   same tracker, trained visibility branch and selector; it does not isolate every
   learned temporal component. CSV/JSON/plots retain frame positions, counts,
   cold-start costs and per-video outcomes. At most six disjoint chunks per video
   are sampled; this is not guaranteed exhaustive full-recording evaluation.
8. Evaluate nonempty still/video validation views. If the optional-stage start
   budget allows, train a matched no-synthetic-overlap base control and compare it
   on the same validation views. Record the stronger standalone base independently
   of the base paired with the temporal refiner.
9. Write a report, model manifest/checksums, prompt hashes and learning curves.
   Failures stop dependent stages and retain a partial report; optional control
   failure preserves the main models. No training quality or completion-time
   guarantee is implied by successful execution.

A still-only export produces detector/base models and explicitly skips temporal.
Use `--require-temporal` when a temporal model is required, so missing clips fail
before training. `--skip-temporal` explicitly disables it.

## Outputs to inspect

- `REPORT.md`: selected paths, measured validation results, warnings and suggested
  next experiments, including weak classes and oracle/matched gaps.
- `MODEL_MANIFEST.json`: complete inference-file paths/hashes, temporal/base pairing,
  and discovered flow-weight paths. Preserve entire base folders, not base.pt alone.
- `dataset_audit.json`, `dataset_build_report.json`,
  `dataset_split_assignments.json` when available, `provenance.json`,
  `PROMPT_EMBEDDINGS.json`, `commands.jsonl`, and one log per stage.
- `learning_curves.png`, evaluator PR curves, base per-class/source metrics, and
  `temporal/eval-*/frames.csv`, `summary.json`, and plots.

The default recipe uses validation only. Add `--evaluate-test` for final evaluation
of available `test`, `test_known`, and `test_unseen` cohorts **after recipes are
fixed**. These results never select checkpoints. Unseen-tool scene metrics still
include known companions unless target-filtered separately. Existing evaluators'
IoU thresholds are fixed AP50/AP75, not COCO AP50:95. Temporal runtime metrics
currently score frames with one visible mask-valid target, not full multi-object
tracking quality; interpret multi-tool coverage accordingly.

## Useful overrides

```bash
python -m experiments.run_overnight --help
```

- `--detector-epochs`, `--base-epochs`, `--temporal-epochs`, `--temporal-batches`:
  training budgets; `--seed` for repeatability.
- `--detector-batch-size`, `--batch-size`, `--image-size`, `--num-workers`:
  hardware settings. Temporal batch size remains 1. There is no automatic OOM retry.
- `--detector rtdetrv2` and optional `--detector-model`: alternative initialization.
  Fresh pretrained initialization is the default, so prior project weights are not
  inadvertently reused in an unseen-tool experiment.
- `--detector-run path`: explicitly reuse/wait for an existing compatible full-epoch
  detector job; `--wait-hours` bounds waiting. Check its training provenance.
- `--skip-control`: omit the extra base comparison. `--optional-after-hours 8` is a
  deadline for **starting** that optional stage, not a total runtime cap.
- `--data-smoke-only --smoke-image-size 128`: audit and decode train/validation clips
  without training or downloading models. Still requires `--dataset-path`.

For follow-up work, prioritize data coverage warnings, low-Dice classes, real
occlusion/empty scenes, identity-versus-learned temporal deltas and another seed
before expanding learning-rate or architecture searches.
