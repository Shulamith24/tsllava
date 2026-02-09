#!/bin/bash
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

#
# 双分支 LLM 时序分类 - UCR 全数据集批量训练脚本
#
# 使用方法:
#   1. 基本用法 (遍历所有 128 个 UCR 数据集):
#      ./scripts/run_dual_branch_llm_datasets.sh
#
#   2. 自定义参数:
#      ./scripts/run_dual_branch_llm_datasets.sh --epochs 30 --batch_size 4 --branch_mode both
#
#   3. 单分支训练:
#      ./scripts/run_dual_branch_llm_datasets.sh --branch_mode ts_only --no_resume
#
#   4. 多卡 DDP 训练:
#      NUM_GPUS=4 ./scripts/run_dual_branch_llm_datasets.sh --use_ddp
#

set -e

# ================================
# 颜色定义
# ================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ================================
# 脚本根目录
# ================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}  双分支 LLM 时序分类 - UCR 全数据集批量训练${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}项目目录: ${PROJECT_DIR}${NC}"
echo ""

# ================================
# 默认参数
# ================================
EPOCHS=30
BATCH_SIZE=4
EVAL_BATCH_SIZE=8
LR_ENCODER=2e-4
LR_PROJECTOR=1e-4
LR_LORA=1e-4
WEIGHT_DECAY=1e-2
GRAD_CLIP=1.0
WARMUP_RATIO=0.03

# LLM 配置
LLM_ID="meta-llama/Llama-3.2-1B"
USE_LORA=true
LORA_R=16
LORA_ALPHA=32

# 分支配置
BRANCH_MODE="both"  # both, ts_only, vision_only
VIT_MODEL_NAME="facebook/dinov2-base"

# PatchTST 配置
PATCH_LENGTH=16
STRIDE=8
D_MODEL=128
NUM_ATTENTION_HEADS=8
NUM_HIDDEN_LAYERS=3
FFN_DIM=512
DROPOUT=0.1

# 投影层配置
PROJECTOR_TYPE="mlp"
PROJECTOR_DROPOUT=0.1

# 冻结选项
FREEZE_TS_BACKBONE=false
FREEZE_VISION_BACKBONE=true
FREEZE_ENCODER=false

# 显存优化
FP16=true
GRADIENT_ACCUMULATION_STEPS=4
GRADIENT_CHECKPOINTING=false

# DDP 配置
USE_DDP=false
NUM_GPUS=${NUM_GPUS:-1}

# 数据与保存路径
DATA_PATH="./data"
SAVE_DIR="results/dual_branch_llm"

# 其他参数
SEED=42
DEVICE="cuda"
EVAL_EVERY=5
EARLY_STOP=10
MAX_NEW_TOKENS=10

# 恢复训练
RESUME=true
NO_RESUME=false

# ================================
# 解析命令行参数
# ================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --eval_batch_size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --lr_encoder) LR_ENCODER="$2"; shift 2 ;;
        --lr_projector) LR_PROJECTOR="$2"; shift 2 ;;
        --lr_lora) LR_LORA="$2"; shift 2 ;;
        --llm_id) LLM_ID="$2"; shift 2 ;;
        --use_lora) USE_LORA=true; shift ;;
        --no_lora) USE_LORA=false; shift ;;
        --lora_r) LORA_R="$2"; shift 2 ;;
        --lora_alpha) LORA_ALPHA="$2"; shift 2 ;;
        --branch_mode) BRANCH_MODE="$2"; shift 2 ;;
        --vit_model_name) VIT_MODEL_NAME="$2"; shift 2 ;;
        --patch_length) PATCH_LENGTH="$2"; shift 2 ;;
        --stride) STRIDE="$2"; shift 2 ;;
        --d_model) D_MODEL="$2"; shift 2 ;;
        --num_attention_heads) NUM_ATTENTION_HEADS="$2"; shift 2 ;;
        --num_hidden_layers) NUM_HIDDEN_LAYERS="$2"; shift 2 ;;
        --ffn_dim) FFN_DIM="$2"; shift 2 ;;
        --dropout) DROPOUT="$2"; shift 2 ;;
        --projector_type) PROJECTOR_TYPE="$2"; shift 2 ;;
        --freeze_ts_backbone) FREEZE_TS_BACKBONE=true; shift ;;
        --freeze_vision_backbone) FREEZE_VISION_BACKBONE=true; shift ;;
        --no_freeze_vision_backbone) FREEZE_VISION_BACKBONE=false; shift ;;
        --freeze_encoder) FREEZE_ENCODER=true; shift ;;
        --fp16) FP16=true; shift ;;
        --no_fp16) FP16=false; shift ;;
        --gradient_accumulation_steps) GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
        --gradient_checkpointing) GRADIENT_CHECKPOINTING=true; shift ;;
        --use_ddp) USE_DDP=true; shift ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --data_path) DATA_PATH="$2"; shift 2 ;;
        --save_dir) SAVE_DIR="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --eval_every) EVAL_EVERY="$2"; shift 2 ;;
        --early_stop) EARLY_STOP="$2"; shift 2 ;;
        --no_resume) NO_RESUME=true; shift ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --epochs N            训练轮数 (default: $EPOCHS)"
            echo "  --batch_size N        批次大小 (default: $BATCH_SIZE)"
            echo "  --branch_mode MODE    分支模式: both/ts_only/vision_only (default: $BRANCH_MODE)"
            echo "  --use_lora            启用 LoRA (default: true)"
            echo "  --no_lora             禁用 LoRA"
            echo "  --lora_r N            LoRA rank (default: $LORA_R)"
            echo "  --fp16                启用 FP16 (default: true)"
            echo "  --gradient_accumulation_steps N  梯度累积步数 (default: $GRADIENT_ACCUMULATION_STEPS)"
            echo "  --use_ddp             启用 DDP 分布式训练"
            echo "  --no_resume           重新训练所有数据集"
            echo "  --help                显示帮助信息"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ================================
# 打印配置
# ================================
echo -e "${YELLOW}配置参数:${NC}"
echo "  LLM: $LLM_ID"
echo "  USE_LORA: $USE_LORA (r=$LORA_R, alpha=$LORA_ALPHA)"
echo "  BRANCH_MODE: $BRANCH_MODE"
echo "  VIT_MODEL: $VIT_MODEL_NAME"
echo "  EPOCHS: $EPOCHS"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  FP16: $FP16"
echo "  GRADIENT_ACCUMULATION: $GRADIENT_ACCUMULATION_STEPS"
echo "  USE_DDP: $USE_DDP (NUM_GPUS=$NUM_GPUS)"
echo ""

# ================================
# UCR 数据集列表 (128 个)
# ================================
UCR_DATASETS=(
    "ACSF1" "Adiac" "AllGestureWiimoteX" "AllGestureWiimoteY" "AllGestureWiimoteZ"
    "ArrowHead" "Beef" "BeetleFly" "BirdChicken" "BME"
    "Car" "CBF" "Chinatown" "ChlorineConcentration" "CinCECGTorso"
    "Coffee" "Computers" "CricketX" "CricketY" "CricketZ"
    "Crop" "DiatomSizeReduction" "DistalPhalanxOutlineAgeGroup" "DistalPhalanxOutlineCorrect" "DistalPhalanxTW"
    "DodgerLoopDay" "DodgerLoopGame" "DodgerLoopWeekend" "Earthquakes" "ECG200"
    "ECG5000" "ECGFiveDays" "ElectricDevices" "EOGHorizontalSignal" "EOGVerticalSignal"
    "EthanolLevel" "FaceAll" "FaceFour" "FacesUCR" "FiftyWords"
    "Fish" "FordA" "FordB" "FreezerRegularTrain" "FreezerSmallTrain"
    "Fungi" "GestureMidAirD1" "GestureMidAirD2" "GestureMidAirD3" "GesturePebbleZ1"
    "GesturePebbleZ2" "GunPoint" "GunPointAgeSpan" "GunPointMaleVersusFemale" "GunPointOldVersusYoung"
    "Ham" "HandOutlines" "Haptics" "Herring" "HouseTwenty"
    "InlineSkate" "InsectEPGRegularTrain" "InsectEPGSmallTrain" "InsectWingbeatSound" "ItalyPowerDemand"
    "LargeKitchenAppliances" "Lightning2" "Lightning7" "Mallat" "Meat"
    "MedicalImages" "MelbournePedestrian" "MiddlePhalanxOutlineAgeGroup" "MiddlePhalanxOutlineCorrect" "MiddlePhalanxTW"
    "MixedShapesRegularTrain" "MixedShapesSmallTrain" "MoteStrain" "NonInvasiveFetalECGThorax1" "NonInvasiveFetalECGThorax2"
    "OliveOil" "OSULeaf" "PhalangesOutlinesCorrect" "Phoneme" "PickupGestureWiimoteZ"
    "PigAirwayPressure" "PigArtPressure" "PigCVP" "PLAID" "Plane"
    "PowerCons" "ProximalPhalanxOutlineAgeGroup" "ProximalPhalanxOutlineCorrect" "ProximalPhalanxTW" "RefrigerationDevices"
    "Rock" "ScreenType" "SemgHandGenderCh2" "SemgHandMovementCh2" "SemgHandSubjectCh2"
    "ShakeGestureWiimoteZ" "ShapeletSim" "ShapesAll" "SmallKitchenAppliances" "SmoothSubspace"
    "SonyAIBORobotSurface1" "SonyAIBORobotSurface2" "StarLightCurves" "Strawberry" "SwedishLeaf"
    "Symbols" "SyntheticControl" "ToeSegmentation1" "ToeSegmentation2" "Trace"
    "TwoLeadECG" "TwoPatterns" "UMD" "UWaveGestureLibraryAll" "UWaveGestureLibraryX"
    "UWaveGestureLibraryY" "UWaveGestureLibraryZ" "Wafer" "Wine" "WordSynonyms"
    "Worms" "WormsTwoClass" "Yoga"
)

# ================================
# 结果统计文件
# ================================
VIT_SHORT_NAME=$(echo "$VIT_MODEL_NAME" | awk -F'/' '{print $NF}' | tr '-' '_')
RESULTS_FILE="${SAVE_DIR}/all_results_${BRANCH_MODE}_${VIT_SHORT_NAME}"
if [ "$USE_LORA" = true ]; then
    RESULTS_FILE="${RESULTS_FILE}_lora"
fi
RESULTS_FILE="${RESULTS_FILE}.csv"

mkdir -p "$SAVE_DIR"

# 初始化结果文件头
if [ ! -f "$RESULTS_FILE" ] || [ "$NO_RESUME" = true ]; then
    echo "dataset,num_classes,context_length,branch_mode,vit_model,use_lora,best_val_acc,test_loss,test_accuracy,epochs_trained,status" > "$RESULTS_FILE"
fi

# ================================
# 训练函数
# ================================
train_dataset() {
    local DATASET_NAME=$1
    local DATASET_INDEX=$2
    local TOTAL_DATASETS=$3

    echo ""
    echo -e "${BLUE}=================================================${NC}"
    echo -e "${BLUE}[$DATASET_INDEX/$TOTAL_DATASETS] 数据集: $DATASET_NAME${NC}"
    echo -e "${BLUE}=================================================${NC}"

    # 检查是否已完成
    if [ "$NO_RESUME" = false ]; then
        local SAVE_SUBDIR="${BRANCH_MODE}_${VIT_SHORT_NAME}"
        if [ "$USE_LORA" = true ]; then
            SAVE_SUBDIR="${SAVE_SUBDIR}_lora"
        fi
        local RESULT_DIR="${SAVE_DIR}/${DATASET_NAME}/${SAVE_SUBDIR}"
        
        if [ -f "${RESULT_DIR}/final_results.json" ]; then
            echo -e "${GREEN}✓ 已存在结果，跳过: ${RESULT_DIR}${NC}"
            return 0
        fi
    fi

    # 构建训练命令
    local CMD="cd $PROJECT_DIR && uv run -m src.train_dual_branch_llm"
    CMD="$CMD --dataset $DATASET_NAME"
    CMD="$CMD --data_path $DATA_PATH"
    CMD="$CMD --llm_id $LLM_ID"
    CMD="$CMD --branch_mode $BRANCH_MODE"
    CMD="$CMD --vit_model_name $VIT_MODEL_NAME"
    CMD="$CMD --epochs $EPOCHS"
    CMD="$CMD --batch_size $BATCH_SIZE"
    CMD="$CMD --eval_batch_size $EVAL_BATCH_SIZE"
    CMD="$CMD --lr_encoder $LR_ENCODER"
    CMD="$CMD --lr_projector $LR_PROJECTOR"
    CMD="$CMD --lr_lora $LR_LORA"
    CMD="$CMD --weight_decay $WEIGHT_DECAY"
    CMD="$CMD --grad_clip $GRAD_CLIP"
    CMD="$CMD --warmup_ratio $WARMUP_RATIO"
    CMD="$CMD --patch_length $PATCH_LENGTH"
    CMD="$CMD --stride $STRIDE"
    CMD="$CMD --d_model $D_MODEL"
    CMD="$CMD --num_attention_heads $NUM_ATTENTION_HEADS"
    CMD="$CMD --num_hidden_layers $NUM_HIDDEN_LAYERS"
    CMD="$CMD --ffn_dim $FFN_DIM"
    CMD="$CMD --dropout $DROPOUT"
    CMD="$CMD --projector_type $PROJECTOR_TYPE"
    CMD="$CMD --projector_dropout $PROJECTOR_DROPOUT"
    CMD="$CMD --seed $SEED"
    CMD="$CMD --eval_every $EVAL_EVERY"
    CMD="$CMD --early_stop $EARLY_STOP"
    CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
    CMD="$CMD --save_dir $SAVE_DIR"
    CMD="$CMD --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"

    if [ "$USE_LORA" = true ]; then
        CMD="$CMD --use_lora --lora_r $LORA_R --lora_alpha $LORA_ALPHA"
    fi
    if [ "$FREEZE_TS_BACKBONE" = true ]; then
        CMD="$CMD --freeze_ts_backbone"
    fi
    if [ "$FREEZE_VISION_BACKBONE" = true ]; then
        CMD="$CMD --freeze_vision_backbone"
    else
        CMD="$CMD --no_freeze_vision_backbone"
    fi
    if [ "$FREEZE_ENCODER" = true ]; then
        CMD="$CMD --freeze_encoder"
    fi
    if [ "$FP16" = true ]; then
        CMD="$CMD --fp16"
    fi
    if [ "$GRADIENT_CHECKPOINTING" = true ]; then
        CMD="$CMD --gradient_checkpointing"
    fi
    if [ "$USE_DDP" = true ]; then
        CMD="torchrun --nproc_per_node=$NUM_GPUS -m src.train_dual_branch_llm $CMD --use_ddp"
    fi

    # 执行训练
    echo -e "${YELLOW}执行命令: $CMD${NC}"
    
    if eval "$CMD"; then
        echo -e "${GREEN}✓ $DATASET_NAME 训练成功${NC}"
        
        # 从 final_results.json 提取结果
        local SAVE_SUBDIR="${BRANCH_MODE}_${VIT_SHORT_NAME}"
        if [ "$USE_LORA" = true ]; then
            SAVE_SUBDIR="${SAVE_SUBDIR}_lora"
        fi
        local RESULT_FILE="${SAVE_DIR}/${DATASET_NAME}/${SAVE_SUBDIR}/final_results.json"
        
        if [ -f "$RESULT_FILE" ]; then
            # 使用 Python 解析 JSON
            python3 -c "
import json
with open('$RESULT_FILE') as f:
    r = json.load(f)
print(f\"{r['dataset']},{r['num_classes']},{r['context_length']},{r['branch_mode']},{r['vit_model_name']},{r['use_lora']},{r['best_val_acc']:.4f},{r['test_loss']:.4f},{r['test_accuracy']:.4f},{r['epochs_trained']},success\")
" >> "$RESULTS_FILE"
        fi
    else
        echo -e "${RED}✗ $DATASET_NAME 训练失败${NC}"
        echo "$DATASET_NAME,,,,,,,,,,failed" >> "$RESULTS_FILE"
    fi
}

# ================================
# 主循环
# ================================
TOTAL=${#UCR_DATASETS[@]}
INDEX=0

for DATASET in "${UCR_DATASETS[@]}"; do
    INDEX=$((INDEX + 1))
    train_dataset "$DATASET" "$INDEX" "$TOTAL"
done

# ================================
# 汇总结果
# ================================
echo ""
echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}训练完成！结果汇总${NC}"
echo -e "${BLUE}=================================================${NC}"

# 统计成功和失败数量
SUCCESS_COUNT=$(grep -c ",success$" "$RESULTS_FILE" 2>/dev/null || echo 0)
FAILED_COUNT=$(grep -c ",failed$" "$RESULTS_FILE" 2>/dev/null || echo 0)

echo "成功: $SUCCESS_COUNT / $TOTAL"
echo "失败: $FAILED_COUNT"

# 计算平均准确率
if [ "$SUCCESS_COUNT" -gt 0 ]; then
    AVG_ACC=$(python3 -c "
import csv
accs = []
with open('$RESULTS_FILE') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('status') == 'success' and row.get('test_accuracy'):
            accs.append(float(row['test_accuracy']))
if accs:
    print(f'{sum(accs)/len(accs):.4f}')
else:
    print('N/A')
")
    echo "平均测试准确率: $AVG_ACC"
fi

echo ""
echo "详细结果: $RESULTS_FILE"
echo -e "${BLUE}=================================================${NC}"
