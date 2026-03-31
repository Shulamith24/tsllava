#!/usr/bin/env bash
"""uv run scripts/ablations/run_1nn_dtw_ucr_all.sh \
  --job-name fewshot_1nn_dtw \
  --shots 1,2,5,10 \
  --num_runs 5"""

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python -u "$SCRIPT_DIR/../experiments/ucr_batch/run_ucr_batch.py" \
  --experiment 1nn_dtw \
  --protocol fewshot \
  "$@"
