#!/usr/bin/env bash
# Temporal training smoke test, prepared with Codex assistance.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

dataset=${DATASET_PATH:-../flowsis-data/exports/combined-sections-coarse}
detector=${DETECTOR_MODEL:-outputs/detectors/coarse-sections-v1/checkpoint-005600}
embeddings=${TEXT_EMBEDDINGS_DIR:-data/combined-coarse/text-embeddings}
clip_length=${CLIP_LENGTH:-4}
python_bin=${PYTHON_BIN:-python}
export PYTHONUNBUFFERED=1

mkdir -p outputs
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  output=$OUTPUT_DIR
  mkdir -p "$(dirname "$output")"
  mkdir "$output" # Refuse to overwrite an existing smoke run.
else
  output=$(mktemp -d outputs/temporal-smoke-XXXXXX)
fi
echo "Smoke artifacts: $output"

base=${BASE_MODEL:-}
if [[ -z "$base" ]]; then
  "$python_bin" -m flowsis.cli.train.train_base \
    --detector dfine --detector-model "$detector" \
    --dataset-path "$dataset" --text-embeddings-dir "$embeddings" \
    --output-dir "$output/base" --device cuda --image-size 640 \
    --train-stages head:1 --max-steps 8 --max-validation-images 8 \
    --batch-size 1 --num-workers 0 --video-fraction 0.5 \
    --rotation-min-degrees -22.5 --rotation-max-degrees 22.5 \
    --no-use-overlap-augment --no-detector-amp --no-amp \
    --seed 42 2>&1 | tee "$output/base.log"
  base="$output/base/best"
fi

"$python_bin" -m flowsis.cli.train.train_temporal \
  --dataset-path "$dataset" --base-model "$base" \
  --output-dir "$output/temporal" --device cuda --image-size 640 \
  --flow-model raft_small --flow-checkpoint things \
  --epochs 1 --batches-per-epoch 4 --validation-batches 4 \
  --batch-size 1 --num-workers 0 \
  --min-clip-length "$clip_length" --max-clip-length "$clip_length" \
  --strides 1 --stride-weights 1 \
  --learning-rate 1e-4 --gradient-clip 1 --seed 42 \
  2>&1 | tee "$output/temporal.log"

"$python_bin" - "$output/temporal/temporal-best.pt" <<'PY'
import math
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
assert checkpoint.get("training_preprocessing") == "unit_rgb_to_uint8_detector_v1", "Trainer lacks the temporal preprocessing fix"
weights = checkpoint["state_dict"]
assert all(torch.isfinite(value).all() for value in weights.values()), "Nonfinite saved weights"
# This layer starts exactly at zero; nonzero values confirm a gradient update.
assert torch.count_nonzero(weights["residual_head.weight"]) > 0, "Residual head did not update"
assert math.isfinite(checkpoint["validation"]["loss"]), "Nonfinite validation loss"
videos = checkpoint["validation_per_video"]
assert len(videos) == 4, f"Expected all four validation videos, got {len(videos)}"
print("PASS: finite checkpoint, residual-head update, and all four validation videos evaluated.")
print("Validation:", checkpoint["validation"])
PY

echo "Smoke passed. Base used: $base"
echo "Logs and checkpoints: $output"
echo "These checkpoints test wiring and optimization, not segmentation quality."
