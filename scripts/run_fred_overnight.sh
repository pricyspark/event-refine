#!/usr/bin/env bash
# Run after importing configs/data/fred-10-videos-full.json.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

manifest=data/fred-10-videos-full/manifest.json
checkpoint=outputs/fred-5-videos-batch3-next10epochs/best
run_dir=${1:-outputs/fred-10-full-overnight}

if [[ ! -f "$manifest" || ! -f "$checkpoint/event_flow.pt" ]]; then
    echo "Missing dataset or starting checkpoint. Import the full-video preset first." >&2
    exit 1
fi
if [[ -e "$run_dir" ]]; then
    echo "Output already exists: $run_dir. Supply a fresh output directory as argument 1." >&2
    exit 1
fi

# Check the active environment before creating outputs or running evaluations.
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in this Python environment"'
mkdir -p "$run_dir"
exec > >(tee "$run_dir/console.log") 2>&1

python -m event_flow evaluate \
    --manifest "$manifest" --split validation \
    --model "$checkpoint" --output "$run_dir/evaluation-start"

python -m event_flow train \
    --manifest "$manifest" --validation-split validation \
    --init-from "$checkpoint" --device cuda \
    --image-size 640 --clip-length 4 --batch-size 3 \
    --learning-rate 1e-5 --temporal-learning-rate 1e-4 \
    --epochs 2 --log-every 100 \
    --output "$run_dir/training"

for candidate in best last; do
    python -m event_flow evaluate \
        --manifest "$manifest" --split validation \
        --model "$run_dir/training/$candidate" \
        --output "$run_dir/evaluation-$candidate"
done

python - "$run_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
print("\nValidation comparison (full sequence 1):")
for candidate in ("start", "best", "last"):
    metrics = json.loads((root / f"evaluation-{candidate}/metrics.json").read_text())
    print(f"{candidate:>5}: AP50={metrics['AP50']:.4f} AP50:95={metrics['AP50_95']:.4f}")

records = [
    json.loads(line)
    for line in (root / "training/progress.jsonl").read_text().splitlines()
]
for epoch in (1, 2):
    ends = [
        record
        for record in records
        if record.get("event") == "progress"
        and record.get("epoch") == epoch
        and record.get("completed_clips") == record.get("total_clips")
    ]
    seconds = sum(record["elapsed_seconds"] for record in ends)
    print(f"Epoch {epoch}: training + validation = {seconds / 3600:.2f} hours (excludes checkpoint writes)")
PY
