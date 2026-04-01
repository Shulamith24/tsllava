#!/usr/bin/env bash
#TODO: --enable_augmentation未测试
#TODO: 目前的shots=full未对齐full脚本，仍然走 shot/run -> support/query -> 两阶段 phase1/phase2 -> phase2_last 这套 few-shot runner，没有验证集、没有 best-model/early-stop，也默认用了另一组训练超参和预处理路径。



: <<'USAGE'
example usage:
  uv run  scripts/run_ucr_classification_pretrained_fewshot_all.sh \
  --job-name fewshot_first   --data-path /mnt/data/qyh/codes/tsllava/data \
  --local_checkpoint /mnt/data/qyh/codes/tsllava/results/curriculum_pretrain_stage12/Llama_3_2_1B/newts_stage12_single_safe/stage2_captioning/checkpoints/best_model.pt \
  --epochs 60   --batch_size 8   --eval_batch_size 8 \
  --protocol fewshot  --shots 1,2,5,10 --num_runs 3 --fewshot_batch_mode manual \
  --gradient_accumulation_steps 4   --device cuda  --gradient_checkpointing --cleanup_checkpoints

torchrun example:
  uv run scripts/run_ucr_classification_pretrained_fewshot_all.sh \
  --torchrun --torchrun-args "--standalone --nproc_per_node=4" \
  --job-name fewshot_ddp   --data-path /mnt/data/qyh/codes/tsllava/data \
  --local_checkpoint /mnt/data/qyh/codes/tsllava/results/curriculum_pretrain_stage12/Llama_3_2_1B/newts_stage12_single_safe/stage2_captioning/checkpoints/best_model.pt \
  --epochs 60   --batch_size 8   --eval_batch_size 8 \
  --shots 1,2,5,10 --num_runs 3 --fewshot_batch_mode manual \
  --gradient_accumulation_steps 4   --device cuda  --gradient_checkpointing --cleanup_checkpoints
USAGE


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python -u "$SCRIPT_DIR/experiments/ucr_batch/run_ucr_batch.py" \
  --experiment m2_pretrained \
  --protocol fewshot \
  "$@"
