#!/bin/bash
# ==============================================================================
# PatchTST + VLM双分支融合 UCR全数据集批量训练脚本
# 
# 用法：
#   bash src/patchtst_ucr/run_patchtst_dual_branch_datasets.sh [image_encoder] [fusion_type] [gpu_id]
# 
# 示例：
#   bash src/patchtst_ucr/run_patchtst_dual_branch_datasets.sh vit concat 0
#   bash src/patchtst_ucr/run_patchtst_dual_branch_datasets.sh resnet cross_attention 1
# ==============================================================================

set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 推导项目根目录 (src/patchtst_ucr -> src -> root)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 默认参数
IMAGE_ENCODER=${1:-"vit"}
FUSION_TYPE=${2:-"concat"}
GPU_ID=${3:-"0"}

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "============================================================"
echo "PatchTST + VLM双分支 UCR批量训练"
echo "============================================================"
echo "项目根目录: $PROJECT_ROOT"
echo "图像编码器: $IMAGE_ENCODER"
echo "融合方式: $FUSION_TYPE"
echo "GPU ID: $GPU_ID"
echo "开始时间: $(date)"
echo "============================================================"

# 配置参数
DATA_PATH="${PROJECT_ROOT}/data"
UCR_DIR="${DATA_PATH}/UCRArchive_2018"
RESULTS_DIR="${PROJECT_ROOT}/results/patchtst_dual_branch_${IMAGE_ENCODER}_${FUSION_TYPE}"
OUTPUT_FILE="${RESULTS_DIR}/all_datasets_accuracy.txt"

# 训练超参数 (针对24GB显存优化)
EPOCHS=50
BATCH_SIZE=4             # 减小batch size
GRAD_ACCUM_STEPS=4       # 梯度累积4步 = 有效batch_size 16
LR=1e-3
EVAL_BATCH_SIZE=16
EVAL_EVERY=5
EARLY_STOP=15
USE_FP16="--fp16"        # 启用FP16混合精度

# 创建结果目录
mkdir -p "$RESULTS_DIR"

# 检查UCR数据目录
if [ ! -d "$UCR_DIR" ]; then
    echo "错误: UCR数据目录不存在: $UCR_DIR"
    exit 1
fi

# 获取所有数据集名称
DATASETS=($(find "$UCR_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort))

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "错误: 未找到任何UCR数据集"
    exit 1
fi

echo "找到 ${#DATASETS[@]} 个数据集"
echo ""

# 初始化结果文件
echo "============================================================" > "$OUTPUT_FILE"
echo "PatchTST + VLM双分支 (${IMAGE_ENCODER} + ${FUSION_TYPE}) 结果汇总" >> "$OUTPUT_FILE"
echo "时间: $(date)" >> "$OUTPUT_FILE"
echo "============================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
printf "%-40s %s\n" "Dataset" "Test Accuracy" >> "$OUTPUT_FILE"
echo "------------------------------------------------------------" >> "$OUTPUT_FILE"

# 统计变量
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_DATASETS=()

# 遍历所有数据集
for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    DATASET_NUM=$((i+1))
    TOTAL_NUM=${#DATASETS[@]}
    
    echo ""
    echo "============================================================"
    echo "[$DATASET_NUM/$TOTAL_NUM] 训练数据集: $DATASET"
    echo "============================================================"
    
    # 构造额外参数
    EXTRA_ARGS=""
    if [ "$IMAGE_ENCODER" == "resnet" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --image_encoder_type resnet"
    elif [ "$IMAGE_ENCODER" == "cnn" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --image_encoder_type cnn"
    else
        EXTRA_ARGS="$EXTRA_ARGS --image_encoder_type vit"
    fi
    
    if [ "$FUSION_TYPE" == "cross_attention" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --fusion_type cross_attention"
    else
        EXTRA_ARGS="$EXTRA_ARGS --fusion_type concat"
    fi
    
    # 运行训练脚本
    # 注意：使用 python -m patchtst_ucr.train_dual_branch 方式运行
    if python -m patchtst_ucr.train_dual_branch \
        --dataset "$DATASET" \
        --data_path "$DATA_PATH" \
        --save_dir "$RESULTS_DIR" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --grad_accum_steps $GRAD_ACCUM_STEPS \
        $USE_FP16 \
        --lr $LR \
        --eval_batch_size $EVAL_BATCH_SIZE \
        --eval_every $EVAL_EVERY \
        --early_stop $EARLY_STOP \
        --device "cuda" \
        $EXTRA_ARGS; then
        
        # 提取测试准确率
        # 结果路径类似于: results/patchtst_dual_branch_vit_concat/Adiac/vit_concat/final_results.json
        # 或者更简单的目录结构，取决于train_dual_branch.py中的实现
        # 目前脚本中save_dir是传入参数，脚本内部会再加上dataset和config子目录
        # 我们需要查找该dataset下的最新结果
        
        RESULT_JSON_PATTERN="${RESULTS_DIR}/${DATASET}/*/final_results.json"
        LATEST_RESULT=$(ls -t $RESULT_JSON_PATTERN 2>/dev/null | head -1)
        
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

# 写入汇总
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
