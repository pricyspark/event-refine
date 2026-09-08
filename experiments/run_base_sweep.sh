#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   experiments/run_base_sweep.sh throughput
#   PRIMARY_MODEL=outputs/good/rtdetrv2-r50-single \
#     experiments/run_base_sweep.sh unfreezing --detector rtdetrv2
#
# Every argument after the suite is forwarded to train_base.py. Environment
# variables below can override repository-local defaults.

suite=${1:-}
if [[ -z "${suite}" ]]; then
  echo "usage: $0 SUITE [train_base args...]" >&2
  echo "suites: initialization, initialization-dfine, initialization-rtdetrv2," >&2
  echo "        unfreezing, scheduler, learning-rate, throughput, precision, seeds" >&2
  exit 2
fi
shift

text_embeddings_dir=${TEXT_EMBEDDINGS_DIR:-data/new/text-embeddings}
dataset_path=${DATASET_PATH:-data/new/segmentation-dataset}
output_root=${OUTPUT_ROOT:-outputs/experiments/base}
primary_model=${PRIMARY_MODEL:-outputs/good/dfine-l-single}
dfine_model=${DFINE_MODEL:-outputs/good/dfine-l-single}
rtdetrv2_model=${RTDETRV2_MODEL:-outputs/good/rtdetrv2-r50-single}
primary_name=${PRIMARY_NAME:-$(basename "${primary_model}")}
dfine_name=${DFINE_NAME:-$(basename "${dfine_model}")}
rtdetrv2_name=${RTDETRV2_NAME:-$(basename "${rtdetrv2_model}")}
dfine_official_control=${DFINE_OFFICIAL_CONTROL_MODEL:-}
rtdetrv2_official_control=${RTDETRV2_OFFICIAL_CONTROL_MODEL:-}

common=(
  --text-embeddings-dir "${text_embeddings_dir}"
  --dataset-path "${dataset_path}"
  --device cuda
  --image-size 640
  --warmup-ratio 0.05
  --scheduler-type constant_with_warmup
  --amp
  --no-detector-amp
)
extra=("$@")

run_experiment() {
  local name=$1
  local model=$2
  shift 2
  echo "==> ${name}"
  python -m flowsis.cli.train.train_base \
    "${common[@]}" \
    --detector-model "${model}" \
    --output-dir "${output_root}/${name}" \
    "$@" \
    "${extra[@]}"
}

run_dfine_initialization() {
  run_experiment "init-${dfine_name}-finetuned" "${dfine_model}" \
    --train-stages head:5,neck:3,all:2
  if [[ -n "${dfine_official_control}" ]]; then
    run_experiment "init-${dfine_name}-official-adapted" \
      "${dfine_official_control}" \
      --train-stages head:5,neck:3,all:2
  else
    echo "Skipping D-FINE official control; set DFINE_OFFICIAL_CONTROL_MODEL." >&2
  fi
}

run_rtdetrv2_initialization() {
  run_experiment "init-${rtdetrv2_name}-finetuned" "${rtdetrv2_model}" \
    --train-stages head:5,neck:3,all:2
  if [[ -n "${rtdetrv2_official_control}" ]]; then
    run_experiment "init-${rtdetrv2_name}-official-adapted" \
      "${rtdetrv2_official_control}" \
      --train-stages head:5,neck:3,all:2
  else
    echo "Skipping RT-DETRv2 official control;" \
      "set RTDETRV2_OFFICIAL_CONTROL_MODEL." >&2
  fi
}

case "${suite}" in
  initialization)
    # Architecture results are reported separately; they are not a controlled
    # checkpoint-initialization comparison because the model capacities differ.
    run_dfine_initialization
    run_rtdetrv2_initialization
    ;;
  initialization-dfine)
    run_dfine_initialization
    ;;
  initialization-rtdetrv2)
    run_rtdetrv2_initialization
    ;;
  unfreezing)
    run_experiment "${primary_name}-unfreeze-head-only" "${primary_model}" \
      --train-stages head:10
    run_experiment "${primary_name}-unfreeze-immediate-all" "${primary_model}" \
      --train-stages all:10
    if [[ ${SKIP_STAGED:-0} != 1 ]]; then
      run_experiment "${primary_name}-unfreeze-staged" "${primary_model}" \
        --train-stages head:5,neck:3,all:2
    else
      echo "Skipping staged unfreezing; reusing the staged initialization run." >&2
    fi
    ;;
  scheduler)
    run_experiment "${primary_name}-scheduler-constant" "${primary_model}" \
      --train-stages head:5,neck:3,all:2 \
      --scheduler-type constant_with_warmup
    run_experiment "${primary_name}-scheduler-linear" "${primary_model}" \
      --train-stages head:5,neck:3,all:2 \
      --scheduler-type linear
    run_experiment "${primary_name}-scheduler-cosine" "${primary_model}" \
      --train-stages head:5,neck:3,all:2 \
      --scheduler-type cosine
    ;;
  learning-rate)
    run_experiment "${primary_name}-lr-conservative" "${primary_model}" \
      --train-stages head:5,neck:3,all:2 \
      --head-lr 5e-5 --detector-lr 5e-6 --backbone-lr 5e-7
    run_experiment "${primary_name}-lr-baseline" "${primary_model}" \
      --train-stages head:5,neck:3,all:2 \
      --head-lr 1e-4 --detector-lr 1e-5 --backbone-lr 1e-6
    run_experiment "${primary_name}-lr-aggressive" "${primary_model}" \
      --train-stages head:5,neck:3,all:2 \
      --head-lr 2e-4 --detector-lr 2e-5 --backbone-lr 2e-6
    ;;
  throughput)
    # These are short all-parameter training runs. Compare epoch
    # train_images_per_second and peak_memory_mb, not wall time including eval.
    for batch_size in ${BATCH_SIZES:-1 2 4}; do
      for num_workers in ${NUM_WORKERS:-4 8}; do
        run_experiment "${primary_name}-throughput-b${batch_size}-w${num_workers}" \
          "${primary_model}" \
          --train-stages all:1 \
          --batch-size "${batch_size}" \
          --num-workers "${num_workers}" \
          --max-steps 100 \
          --max-validation-images 50 \
          --save-every-epochs 100
      done
    done
    ;;
  precision)
    run_experiment "${primary_name}-precision-fp32" "${primary_model}" \
      --train-stages all:1 \
      --max-steps 100 \
      --max-validation-images 50 \
      --no-amp \
      --no-detector-amp
    run_experiment "${primary_name}-precision-head-amp" "${primary_model}" \
      --train-stages all:1 \
      --max-steps 100 \
      --max-validation-images 50 \
      --amp \
      --no-detector-amp
    if [[ ${DETECTOR_AMP_SAFE:-0} == 1 ]]; then
      run_experiment "${primary_name}-precision-full-amp" "${primary_model}" \
        --train-stages all:1 \
        --max-steps 100 \
        --max-validation-images 50 \
        --amp \
        --detector-amp
    else
      echo "Skipping detector AMP; set DETECTOR_AMP_SAFE=1 only for a validated backend." >&2
    fi
    ;;
  seeds)
    # Run this after selecting the checkpoint, unfreezing schedule, and LR.
    for seed in 42 123 777; do
      run_experiment "${primary_name}-seed-${seed}" "${primary_model}" \
        --train-stages head:5,neck:3,all:2 \
        --seed "${seed}"
    done
    ;;
  *)
    echo "unknown suite: ${suite}" >&2
    exit 2
    ;;
esac
