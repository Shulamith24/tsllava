#!/bin/bash
# ==============================================================================
# PatchTST + VisionEncoder 双分支 UCR 全数据集批量训练脚本
# 
# 用法：可在任意路径下执行此脚本
#   bash /path/to/run_dual_branch_tivit_datasets.sh
#
# 多卡 DDP 运行：
#   bash /path/to/run_dual_branch_tivit_datasets.sh --ddp --gpus 2
# ==============================================================================

set -e  # 遇到错误立即退出

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 推导项目根目录 (scripts -> root)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "============================================================"
echo "PatchTST + VisionEncoder 双分支 UCR 批量训练"
echo "============================================================"
echo "项目根目录: $PROJECT_ROOT"
echo "脚本目录: $SCRIPT_DIR"
echo "开始时间: $(date)"
echo "============================================================"

# 默认配置参数
DATA_PATH="${PROJECT_ROOT}/data"
UCR_DIR="${DATA_PATH}/UCRArchive_2018"
RESULTS_DIR="${PROJECT_ROOT}/results/patchtst_dual_branch_tivit"

# 训练超参数
EPOCHS=100
BATCH_SIZE=16
LR=1e-3
EVAL_BATCH_SIZE=32
EVAL_EVERY=5
EARLY_STOP=15

# 双分支配置
VIT_MODEL_NAME="facebook/dinov2-base"
BRANCH_MODE="both"  # both, ts_only, vision_only
AGGREGATOR_LAYERS=1

# 显存优化选项
FP16=true
GRADIENT_ACCUMULATION_STEPS=4
FREEZE_VISION_BACKBONE=true
FREEZE_TS_BACKBONE=false

# DDP 配置
USE_DDP=false
NUM_GPUS=1

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --ddp)
            USE_DDP=true
            shift
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --branch_mode)
            BRANCH_MODE="$2"
            shift 2
            ;;
        --vit_model)
            VIT_MODEL_NAME="$2"
            shift 2
            ;;
        --no_fp16)
            FP16=false
            shift
            ;;
        --unfreeze_vision)
            FREEZE_VISION_BACKBONE=false
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 生成结果文件名
VIT_SHORT_NAME=$(echo "$VIT_MODEL_NAME" | sed 's/.*\///' | sed 's/-/_/g')
OUTPUT_FILE="${RESULTS_DIR}/all_datasets_accuracy_${BRANCH_MODE}_${VIT_SHORT_NAME}.txt"

# 创建结果目录
mkdir -p "$RESULTS_DIR"

# 检查UCR数据目录是否存在
if [ ! -d "$UCR_DIR" ]; then
    echo "错误: UCR数据目录不存在: $UCR_DIR"
    echo "请先运行一次训练脚本以下载数据，或手动下载UCRArchive_2018数据集"
    exit 1
fi

# 获取所有数据集名称
DATASETS=($(find "$UCR_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort))

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "错误: 未找到任何UCR数据集"
    exit 1
fi

echo "找到 ${#DATASETS[@]} 个数据集"
echo "分支模式: $BRANCH_MODE"
echo "ViT 模型: $VIT_MODEL_NAME"
echo "DDP: $USE_DDP (GPUs: $NUM_GPUS)"
echo "FP16: $FP16"
echo "梯度累积: $GRADIENT_ACCUMULATION_STEPS"
echo ""

# 初始化结果文件
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "创建新结果文件: $OUTPUT_FILE"
    echo "============================================================" > "$OUTPUT_FILE"
    echo "PatchTST + VisionEncoder 双分支 UCR 分类结果汇总" >> "$OUTPUT_FILE"
    echo "时间: $(date)" >> "$OUTPUT_FILE"
    echo "分支模式: $BRANCH_MODE" >> "$OUTPUT_FILE"
    echo "ViT 模型: $VIT_MODEL_NAME" >> "$OUTPUT_FILE"
    echo "============================================================" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    printf "%-40s %s\n" "Dataset" "Test Accuracy" >> "$OUTPUT_FILE"
    echo "------------------------------------------------------------" >> "$OUTPUT_FILE"
else
    echo "发现已有结果文件，启用断点续训模式: $OUTPUT_FILE"
    echo "------------------------------------------------------------" >> "$OUTPUT_FILE"
    echo "断点续训开始时间: $(date)" >> "$OUTPUT_FILE"
fi

# 记录成功和失败的数据集
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_DATASETS=()

# 遍历所有数据集进行训练
for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    DATASET_NUM=$((i+1))
    TOTAL_NUM=${#DATASETS[@]}
    
    echo ""
    echo "============================================================"
    echo "[$DATASET_NUM/$TOTAL_NUM] 训练数据集: $DATASET"
    echo "============================================================"
    
    # 断点续训：检查是否已经完成
    if grep -q "^$DATASET " "$OUTPUT_FILE" 2>/dev/null; then
        echo "✓ $DATASET 已完成，跳过"
        PREV_ACC=$(grep "^$DATASET " "$OUTPUT_FILE" | awk '{print $NF}')
        echo "  (历史记录 Accuracy: $PREV_ACC)"
        continue
    fi
    
    # 构建基础训练命令
    BASE_CMD="uv run -m src.train_dual_branch_tivit \
        --dataset $DATASET \
        --data_path $DATA_PATH \
        --save_dir $RESULTS_DIR \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --eval_batch_size $EVAL_BATCH_SIZE \
        --eval_every $EVAL_EVERY \
        --early_stop $EARLY_STOP \
        --vit_model_name $VIT_MODEL_NAME \
        --branch_mode $BRANCH_MODE \
        --aggregator_layers $AGGREGATOR_LAYERS \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"
    
    # 添加显存优化选项
    if [ "$FP16" = true ]; then
        BASE_CMD="$BASE_CMD --fp16"
    fi
    
    if [ "$FREEZE_VISION_BACKBONE" = true ]; then
        BASE_CMD="$BASE_CMD --freeze_vision_backbone"
    else
        BASE_CMD="$BASE_CMD --no_freeze_vision_backbone"
    fi
    
    if [ "$FREEZE_TS_BACKBONE" = true ]; then
        BASE_CMD="$BASE_CMD --freeze_ts_backbone"
    fi
    
    # 构建最终命令（DDP 或单卡）
    if [ "$USE_DDP" = true ]; then
        CMD="torchrun --nproc_per_node=$NUM_GPUS $BASE_CMD --use_ddp"
    else
        CMD="$BASE_CMD"
    fi
    
    # 运行训练脚本
    if eval $CMD; then
        
        # 提取测试准确率
        RESULT_JSON="${RESULTS_DIR}/${DATASET}/*/final_results.json"
        LATEST_RESULT=$(ls -t $RESULT_JSON 2>/dev/null | head -1)
        
        if [ -n "$LATEST_RESULT" ] && [ -f "$LATEST_RESULT" ]; then
            TEST_ACC=$(python -c "import json; f=open('$LATEST_RESULT'); d=json.load(f); print(f\"{d['test_accuracy']:.4f}\")")
            printf "%-40s %s\n" "$DATASET" "$TEST_ACC" >> "$OUTPUT_FILE"
            echo "✓ $DATASET 完成: Test Accuracy = $TEST_ACC"
            SUCCESS_COUNT=$((SUCCESS_COUNT+1))
        else
            printf "%-40s %s\n" "$DATASET" "结果文件未找到" >> "$OUTPUT_FILE"
            echo "⚠ $DATASET 完成，但未找到结果文件"
            FAIL_COUNT=$((FAIL_COUNT+1))
            FAILED_DATASETS+=("$DATASET")
        fi
    else
        printf "%-40s %s\n" "$DATASET" "训练失败" >> "$OUTPUT_FILE"
        echo "✗ $DATASET 训练失败"
        FAIL_COUNT=$((FAIL_COUNT+1))
        FAILED_DATASETS+=("$DATASET")
    fi
done

# 写入汇总信息
echo "" >> "$OUTPUT_FILE"
echo "============================================================" >> "$OUTPUT_FILE"
echo "汇总" >> "$OUTPUT_FILE"
echo "成功: $SUCCESS_COUNT / ${#DATASETS[@]}" >> "$OUTPUT_FILE"
echo "失败: $FAIL_COUNT / ${#DATASETS[@]}" >> "$OUTPUT_FILE"
if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo "失败的数据集: ${FAILED_DATASETS[*]}" >> "$OUTPUT_FILE"
fi
echo "完成时间: $(date)" >> "$OUTPUT_FILE"
echo "============================================================" >> "$OUTPUT_FILE"

echo ""
echo "============================================================"
echo "全部训练完成!"
echo "成功: $SUCCESS_COUNT / ${#DATASETS[@]}"
echo "失败: $FAIL_COUNT / ${#DATASETS[@]}"
if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo "失败的数据集: ${FAILED_DATASETS[*]}"
fi
echo "结果已保存到: $OUTPUT_FILE"
echo "完成时间: $(date)"
echo "============================================================"
