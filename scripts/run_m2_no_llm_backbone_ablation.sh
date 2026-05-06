#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  python -m scripts.experiments.ucr_batch.run_m2_no_llm_backbone_ablation "$@"
else
  uv run python -m scripts.experiments.ucr_batch.run_m2_no_llm_backbone_ablation "$@"
fi
