#!/usr/bin/env bash

: <<'USAGE'
Queue few-shot baseline experiments on CinC2016HeartSound.

Defaults:
  DATA_ROOT=./data
  SHOTS=10,20,30
  RUNS=5
  EPOCHS=60
  GPU_IDS=${GPU:-0}
  EXPS="onefitsall patchtst tslib_autoformer tslib_crossformer tslib_dlinear tslib_fedformer tslib_informer tslib_timesnet cosco_resnet resnet tapnet"

Example:
  GPU_IDS=0,1 DATA_ROOT=./data \
  bash scripts/experiments/ucr_batch/run_cinc2016heart_baselines_fewshot_queue.sh \
    --cleanup_checkpoints

Extra arguments are forwarded to the underlying training scripts.
USAGE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATA_ROOT="${DATA_ROOT:-./data}"
SHOTS="${SHOTS:-10,20,30}"
RUNS="${RUNS:-5}"
EPOCHS="${EPOCHS:-60}"
GPU_IDS="${GPU_IDS:-${GPU:-0}}"
EXPS="${EXPS:-onefitsall patchtst tslib_autoformer tslib_crossformer tslib_dlinear tslib_fedformer tslib_informer tslib_timesnet cosco_resnet resnet tapnet}"
LAUNCHER_NAME="${LAUNCHER_NAME:-cinc2016heart_baselines_fewshot}"
if [[ -z "${JOB_NAME_TEMPLATE:-}" ]]; then
  JOB_NAME_TEMPLATE="{experiment}_cinc2016heart_fewshot"
fi

QUEUE_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  QUEUE_ARGS+=(--dry-run)
fi

uv run python -u "$SCRIPT_DIR/run_ucr_experiment_queue.py" \
  --experiments "$EXPS" \
  --protocol fewshot \
  --job-name-template "$JOB_NAME_TEMPLATE" \
  --data-path "$DATA_ROOT" \
  --gpu-ids "$GPU_IDS" \
  --dataset-families cinc2016heart \
  --launcher-name "$LAUNCHER_NAME" \
  "${QUEUE_ARGS[@]}" \
  -- \
  --shots "$SHOTS" \
  --num_runs "$RUNS" \
  --epochs "$EPOCHS" \
  "$@"
