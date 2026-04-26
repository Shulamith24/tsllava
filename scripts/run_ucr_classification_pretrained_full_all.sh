#!/usr/bin/env bash
#TODO: padding模式未测试
#tail -f results/ucr_batches/m2_pretrained/full/first/logs/ACSF1.log

: <<'USAGE'
example usage:
  uv run  scripts/run_ucr_classification_pretrained_fewshot_all.sh \
  --job-name fewshot_first   --data-path ./data \
  --local_checkpoint ./results/curriculum_pretrain_stage0_tsqa_m4/Llama_3_2_1B/newts_dual_branch_curriculum_v3/stage1_transfer_checkpoint.pt \
  --epochs 60   --batch_size 4   --eval_batch_size 4 \
  --gradient_accumulation_steps 8   --device cuda  --gradient_checkpointing --cleanup_checkpoints
USAGE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python -u "$SCRIPT_DIR/experiments/ucr_batch/run_ucr_batch.py" \
  --experiment m2_pretrained \
  --protocol full \
  "$@"
