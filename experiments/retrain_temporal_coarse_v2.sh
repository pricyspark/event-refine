#!/usr/bin/env bash
# Recovery from the coarse-v2 preprocessing bug, prepared with Codex assistance.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1

output=${OUTPUT_DIR:-outputs/temporal/coarse-v2-fixed-input-v1}
dataset=../flowsis-data/exports/combined-sections-coarse
base=outputs/overnight/coarse-v2/base/overlap25/best
mkdir -p "$(dirname "$output")"
mkdir "$output"

# Reuse the trained base and require plausible mask quality before a full run.
BASE_MODEL="$base" CLIP_LENGTH=8 OUTPUT_DIR="$output/smoke" \
  bash experiments/smoke_temporal.sh
python - "$output/smoke/temporal/temporal-best.pt" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
loss = checkpoint["validation"]["dice_loss"]
assert loss < 0.5, f"Trained-base smoke has unexpectedly poor soft Dice loss: {loss}"
print("Trained-base preprocessing sanity check passed; validation Dice loss:", loss)
PY

python -m flowsis.cli.train.train_temporal \
  --dataset-path "$dataset" --base-model "$base" --output-dir "$output/main" \
  --device cuda --image-size 640 --flow-model raft_small --flow-checkpoint things \
  --epochs 8 --batches-per-epoch 75 --validation-batches 40 \
  --batch-size 1 --num-workers 4 --min-clip-length 4 --max-clip-length 8 \
  --strides 1 2 3 --stride-weights 0.7 0.2 0.1 \
  --learning-rate 1e-4 --gradient-clip 1 --seed 42 \
  2>&1 | tee "$output/training.log"

for geometry in clips16 intervals; do
  length=16
  if [[ "$geometry" == intervals ]]; then length=256; fi
  python -m flowsis.cli.eval.evaluate_temporal \
    --dataset-path "$dataset" --base-model "$base" \
    --temporal-model "$output/main/temporal-best.pt" \
    --device cuda --flow-model raft_small --flow-checkpoint things \
    --clip-length "$length" --clips-per-video 6 --mode both \
    --output-dir "$output/eval-$geometry" \
    2>&1 | tee "$output/eval-$geometry.log"
done
echo "Temporal retraining and paired evaluation complete: $output"
