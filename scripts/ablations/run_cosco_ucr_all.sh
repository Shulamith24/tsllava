#!/usr/bin/env bash
# bash scripts/ablations/run_cosco_ucr_all.sh \
#   --job-name fewshot \
#   --shots 1,2,5,10 \
#   --num_runs 3 \
#   --epochs 100 \
#   --device cuda \
#   --cleanup_checkpoints

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python -u "$SCRIPT_DIR/../experiments/ucr_batch/run_ucr_batch.py" \
  --experiment cosco_resnet \
  --protocol fewshot \
  "$@"
