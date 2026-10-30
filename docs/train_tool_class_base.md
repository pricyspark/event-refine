# Train the nine-class FlowSIS base model

Prepared with Codex assistance for the current sample-split dataset and the
epoch-18 D-FINE Medium checkpoint. Commands run from `/home/pricyspark/flowsis`.
The prompt CLI was updated to respect exported category labels, including
coarse labels, rather than require tool-name-plus-class labels. Its 11 tests
passed. Dataset and detector class order were checked and match.

This workflow trains instance segmentation, including the set-mask head and
staged detector fine-tuning. Temporal refinement is a separate later workflow.
No embedding generation, smoke training, or full training has been run by Codex.

## 1. Activate the existing environment and set paths

```bash
conda activate flowsis
cd /home/pricyspark/flowsis

FLOWSIS_DATASET=../flowsis-data/data/segmentation-dataset
FLOWSIS_DETECTOR=outputs/detectors/dfine-tool-class-20epochs/checkpoint-001170
FLOWSIS_PROMPTS=data/tool-class/text_prompts.json
FLOWSIS_EMBEDDINGS=data/tool-class/text-embeddings
FLOWSIS_BASE=outputs/base/dfine-tool-class-epoch18-v1
```

Keep these variables in the same shell for subsequent commands. The existing
environment used for detector training should already have the dependencies.
If this checkout is not installed, run `python -m pip install -e .` in it.
No API key or external prompt-generation service is required.

Keep the same dataset export used for detector evaluation. The nine ordered
labels are bonepunch, forceps, gouge, mallet, osteotome, pliers, retractor,
rongeur, scissors. Tool names remain metadata and are not additional classes.

## 2. Complete and review the text prompts

A scaffold was created at `data/tool-class/text_prompts.json`. It contains three
entries per class: exact class label, `This is a photo of <label>.`, and an
empty visual description. Fill the third entry for each class. To author them
interactively and save progress:

```bash
flowsis-author-text-prompts \
  --dataset-path "$FLOWSIS_DATASET" \
  --output-path "$FLOWSIS_PROMPTS"
```

Example structure (one entry, not the entire suite):

```json
{
  "scissors": [
    "scissors",
    "This is a photo of scissors.",
    "scissors with two opposing blades joined at a pivot and handles ending in finger rings"
  ]
}
```

Use concise, stable visual geometry appropriate to the represented instruments:
shaft, jaws, tips, blades, teeth, handles, curvature, and openings. Check the
training images and known instrument metadata when writing descriptions.
Keep descriptions broad enough to cover all members of a coarse class;
forceps should not be described as universally having ring handles, for example.
Avoid colors, backgrounds, anatomical context, use cases, and arbitrary IDs.
Begin each description with its exact label (required by the interactive CLI).
Preserve the first two prompts, include all nine labels, and remove no entries.
Descriptions should be short; the text encoder has a 64-token limit.

The fixed three-prompt suite is the initial configuration. Do not reuse the
older differently named embeddings in `data/manifests/text-embeddings`.

## 3. Generate and verify the embedding bank

```bash
flowsis-embed-text-prompts \
  --dataset-path "$FLOWSIS_DATASET" \
  --prompt-suite "$FLOWSIS_PROMPTS" \
  --output-dir "$FLOWSIS_EMBEDDINGS" \
  --model google/siglip2-base-patch16-224 \
  --device cuda
```

The first invocation may download the SigLIP2 tokenizer and text weights.
`--device cpu` also works for this one-time text embedding step if needed.
Expect nine `.pt` files, each `[3, 768]`; their ordered bank is `[9, 3, 768]`.
The CLI checks prompt-suite labels and lengths. Regenerate embeddings whenever
prompt text changes. The text encoder is frozen and does not need to run for
every training image.

```bash
python - <<'PY'
import json
from pathlib import Path
from flowsis.cli.data.embed_text_prompts import taxonomy_from_dataset
from flowsis.data.prompts import load_prompt_bank

labels = list(taxonomy_from_dataset(Path('../flowsis-data/data/segmentation-dataset')))
config = json.loads(Path(
    'outputs/detectors/dfine-tool-class-20epochs/checkpoint-001170/config.json'
).read_text())
detector_labels = [config['id2label'][str(i)] for i in range(len(config['id2label']))]
assert labels == detector_labels, (labels, detector_labels)
bank = load_prompt_bank(labels, 'data/tool-class/text-embeddings')
assert tuple(bank.shape) == (9, 3, 768), bank.shape
print(labels)
print('prompt bank:', tuple(bank.shape))
PY
```

## 4. Run a bounded smoke test

```bash
flowsis-train-base \
  --detector dfine \
  --detector-model "$FLOWSIS_DETECTOR" \
  --dataset-path "$FLOWSIS_DATASET" \
  --text-embeddings-dir "$FLOWSIS_EMBEDDINGS" \
  --output-dir "${FLOWSIS_BASE}-smoke" \
  --device cuda \
  --train-stages head:1 \
  --max-train-images 20 \
  --max-validation-images 10 \
  --max-steps 20 \
  --batch-size 1 \
  --num-workers 0 \
  --no-use-overlap-augment \
  --no-detector-amp \
  --no-amp
```

This checks decoding images/masks, prompt loading, detector feature taps,
mask-head optimization, validation, and saving. It is not a quality estimate
or a continuation checkpoint for the full run. Inspect the `runtime` and
`epoch` events; require finite losses and no persistent optimizer skips.

## 5. Start the full staged baseline

```bash
flowsis-train-base \
  --detector dfine \
  --detector-model "$FLOWSIS_DETECTOR" \
  --dataset-path "$FLOWSIS_DATASET" \
  --text-embeddings-dir "$FLOWSIS_EMBEDDINGS" \
  --output-dir "$FLOWSIS_BASE" \
  --device cuda \
  --image-size 640 \
  --batch-size 2 \
  --num-workers 4 \
  --train-stages head:5,neck:3,all:2 \
  --head-lr 1e-4 \
  --detector-lr 1e-5 \
  --backbone-lr 1e-6 \
  --scheduler-type constant_with_warmup \
  --warmup-ratio 0.05 \
  --gradient-clip 1.0 \
  --text-conditioning frozen \
  --no-detector-amp \
  --amp \
  --seed 42
```

This is a ten-epoch initial segmentation baseline, not a claim that ten epochs
is optimal. The new mask head needs its own convergence assessment; the
detector's prior 20 epochs do not count as mask training. The stages are:

- `head:5`: learn the mask head with the detector frozen.
- `neck:3`: train the mask head and detector parameters outside the backbone.
- `all:2`: fine-tune the complete model, with the smallest LR on the backbone.

`constant_with_warmup` avoids decaying detector LRs while the detector is still
frozen. These settings follow the repository's staged baseline and experiment
protocol; they are not yet optimized for this subset. Keep default mask losses,
stride-4 detail, box prior, and augmentation initially. If memory is exhausted,
reduce batch size to one. Mask-head AMP is enabled, but D-FINE remains FP32;
use `--no-amp` to diagnose non-finite values. Keep `--no-detector-amp` for D-FINE.

Use a fresh output directory for each invocation. Do not use `--resume-from`
to initialize from the detector checkpoint: that flag is for resuming base
training state, while `--detector-model` initializes this new model.

## 6. Inspect and evaluate the segmentation model

`run_config.json` and `training_log.jsonl` capture the run. Look for increasing
`validation_matched_dice` and mask AP, stable box AP, finite losses and gradients,
and `passes_detector_guardrails: true`. `best` is saved by validation matched
Dice among epochs satisfying the detector guardrails; `final` is simply the
last epoch. If `best` is absent, inspect guardrail failures rather than assume
the final model is acceptable. Empty-frame quality cannot be established from
this dataset, which has no background-only images. Sparse-class and real
multi-tool evaluation limitations also remain.

```bash
flowsis-evaluate-base \
  --model-path "$FLOWSIS_BASE/best" \
  --dataset-path "$FLOWSIS_DATASET" \
  --split validation \
  --output-path "$FLOWSIS_BASE/validation.json" \
  --pr-output-dir "$FLOWSIS_BASE/validation-pr" \
  --device cuda \
  --no-detector-amp
```

Use this validation run to assess mask quality, detector preservation, and
confidence thresholds. Extend the schedule or compare unfreezing policies
based on these curves. Evaluate test only after model and threshold selection.
Saved base checkpoints include the detector, mask head, labels, and prompt bank.
