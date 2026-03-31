#!/usr/bin/env bash
# bash scripts/ablations/run_onefitsall_ucr_all.sh \
#   --protocol fewshot \
#   --shots 1,2,5,10 \
#   --num_runs 5 \
#   --epochs 60 \
#   --lr 1e-4 \
#   --gpu 0 \
#   --job-name fewshot
#   --cleanup_checkpoints


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python -u "$SCRIPT_DIR/../experiments/ucr_batch/run_ucr_batch.py" \
  --experiment onefitsall \
  "$@"
