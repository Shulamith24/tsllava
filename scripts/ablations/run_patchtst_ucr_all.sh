#!/usr/bin/env bash
"""
uv run scripts/ablations/run_patchtst_ucr_all.sh \
  --protocol fewshot \
  --job-name fewshot_patchtst \
  --shots 1,2,5,10 \
  --num_runs 5 \
  --epochs 60 \
  --device cuda \
  --cleanup_checkpoints
"""



set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python -u "$SCRIPT_DIR/../experiments/ucr_batch/run_ucr_batch.py" \
  --experiment patchtst \
  "$@"
