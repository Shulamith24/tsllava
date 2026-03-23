#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python -u "$SCRIPT_DIR/../experiments/ucr_batch/run_ucr_batch.py" \
  --experiment onefitsall \
  "$@"
