#!/usr/bin/env bash

"""
uv run scripts/ablations/run_resnet_ucr_all.sh \
  --job-name fewshot_resnet \
  --shots 1,2,5,10 \
  --num_runs 5 \
  --epochs 60 \
  --lr 1e-4 \
  --device cuda \
  --cleanup_checkpoints
"""

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python -u "$SCRIPT_DIR/../experiments/ucr_batch/run_ucr_batch.py" \
  --experiment resnet \
  --protocol fewshot \
  "$@"
