# Base-model training experiments

Use the detector already fine-tuned on this dataset as the primary starting
checkpoint. Its classifier is aligned with the segmentation dataset's ten
classes, and its queries already localize this domain. Do not pass the default
Hub checkpoint directly to `train_base.py`: the official D-FINE Object365 or
RT-DETRv2 COCO classifier does not have the same class count and label order as
the prompt bank.

The fair official-checkpoint control is an official model with a newly
initialized, dataset-sized detector classifier. It answers whether prior
detector fine-tuning helps; it does not pretend that incompatible official
class logits are usable as-is. Create the control without detector updates:

```bash
flowsis-train-detector \
  --detector dfine \
  --model ustc-community/dfine-large-obj365 \
  --dataset-path data/segmentation-dataset \
  --max-steps 0 \
  --epochs 1 \
  --output-dir outputs/controls/dfine-large-official-adapted
```

Then point `DFINE_OFFICIAL_CONTROL_MODEL` at its `final` directory when running
the initialization suite. Use the analogous `RTDETRV2_OFFICIAL_CONTROL_MODEL`
variable for an RT-DETRv2 control made with `--detector rtdetrv2` and its
official model ID.

## Protocol

Run the suites in order rather than taking the Cartesian product:

1. `initialization`: compare each fine-tuned detector to its same-architecture
   official-adapted control. D-FINE versus RT-DETRv2 is a useful deployment
   comparison, but not an isolated checkpoint comparison because capacity and
   compute differ.
2. `unfreezing`: compare a frozen detector, immediate full tuning, and staged
   unfreezing. Keep the initialization winner.
3. `scheduler`: compare constant-after-warmup against linear and cosine decay.
   The constant schedule is the default for these sweeps because a global
   linear schedule otherwise decays detector and backbone learning rates while
   those parameter groups are still frozen.
4. `learning-rate`: test conservative, baseline, and aggressive discriminative
   learning rates with the winning unfreezing schedule.
5. `throughput`: find the largest stable batch size and worker count using
   short `all`-stage runs. An out-of-memory result is a rejected configuration,
   not a reason to change model quality settings.
6. `precision`: compare FP32 with mask-head AMP. Set `DETECTOR_AMP_SAFE=1` only
   for RT-DETRv2 (or another backend whose full-AMP stability you have already
   established); do not enable it for D-FINE.
7. `seeds`: rerun the selected recipe with three seeds before claiming a win.

Examples:

```bash
experiments/run_base_sweep.sh initialization

# Run only D-FINE L when RT-DETRv2 has already been measured.
experiments/run_base_sweep.sh initialization-dfine

PRIMARY_MODEL=outputs/good/dfine-l-single \
  experiments/run_base_sweep.sh unfreezing

# Avoid repeating staged training when initialization-dfine was run first.
PRIMARY_MODEL=outputs/good/dfine-l-single SKIP_STAGED=1 \
  experiments/run_base_sweep.sh unfreezing

BATCH_SIZES="1 2 4 8" NUM_WORKERS="4 8 12" \
  experiments/run_base_sweep.sh throughput
```

Arguments after the suite are forwarded to every run. This is useful for a
cheap screening pass:

```bash
experiments/run_base_sweep.sh learning-rate \
  --max-train-images 500 \
  --max-validation-images 200
```

For D-FINE, keep `--no-detector-amp`; its box-distribution decoding can overflow
in FP16. Test detector AMP only as a separate RT-DETRv2 throughput/stability
ablation. On this repository's recorded TITAN RTX inference profiles, AMP was
slower than FP32, so it should be measured rather than assumed faster.

## Selection criteria

Each output directory contains `run_config.json` and `training_log.jsonl`.
Reject runs with non-finite loss, skipped optimizer steps that persist after the
first epoch, repeated gradient-norm spikes, or detector guardrail failures.
Among stable runs, select by validation matched Dice subject to box AP50 and
empty-frame false-positive guardrails. Use `train_images_per_second` and
`peak_memory_mb` only to break quality-equivalent ties or choose the operational
batch/worker configuration.

Summarize and rank a suite with:

```bash
python experiments/summarize_base_sweep.py outputs/experiments/base
```

Do not select on the test split. Evaluate the one final recipe on test only
after the three-seed validation comparison.
