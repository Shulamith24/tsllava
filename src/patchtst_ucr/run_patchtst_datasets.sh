#!/bin/bash
# ==============================================================================
# PatchTST UCR 全数据集批量训练脚本
# 
# 用法：可在任意路径下执行此脚本
#   bash /path/to/run_patchtst_datasets.sh
#   或
#   cd /path/to/tsllava && bash scripts/run_patchtst_datasets.sh
# ==============================================================================

set -e  # 遇到错误立即退出

# 获取脚本所在目录的绝对路径，从而推导项目根目录
# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 推导项目根目录 (src/patchtst_ucr -> src -> root)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "============================================================"
echo "PatchTST UCR 批量训练"
echo "============================================================"
echo "项目根目录: $PROJECT_ROOT"
echo "脚本目录: $SCRIPT_DIR"
echo "开始时间: $(date)"
echo "============================================================"

# 配置参数
DATA_PATH="${PROJECT_ROOT}/data"
UCR_DIR="${DATA_PATH}/UCRArchive_2018"
RESULTS_DIR="${PROJECT_ROOT}/results/patchtst"
OUTPUT_FILE="${RESULTS_DIR}/all_datasets_accuracy.txt"

# 训练超参数（可根据需要修改）
EPOCHS=100
BATCH_SIZE=32
LR=1e-3
EVAL_BATCH_SIZE=64
EVAL_EVERY=5
EARLY_STOP=15

# 创建结果目录
mkdir -p "$RESULTS_DIR"

# 检查UCR数据目录是否存在
if [ ! -d "$UCR_DIR" ]; then
    echo "错误: UCR数据目录不存在: $UCR_DIR"
    echo "请先运行一次训练脚本以下载数据，或手动下载UCRArchive_2018数据集"
    exit 1
fi

# 获取所有数据集名称（即UCR_DIR下的所有子目录）
DATASETS=($(find "$UCR_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort))

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "错误: 未找到任何UCR数据集"
    exit 1
fi

echo "找到 ${#DATASETS[@]} 个数据集"
echo ""

# 初始化结果文件
echo "============================================================" > "$OUTPUT_FILE"
echo "PatchTST UCR 分类结果汇总" >> "$OUTPUT_FILE"
echo "时间: $(date)" >> "$OUTPUT_FILE"
echo "============================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
printf "%-40s %s\n" "Dataset" "Test Accuracy" >> "$OUTPUT_FILE"
echo "------------------------------------------------------------" >> "$OUTPUT_FILE"

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
    
    # 运行训练脚本 (直接引用同一目录下的 train.py)
    if python "${SCRIPT_DIR}/train.py" \
        --dataset "$DATASET" \
        --data_path "$DATA_PATH" \
        --save_dir "$RESULTS_DIR" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --eval_batch_size $EVAL_BATCH_SIZE \
        --eval_every $EVAL_EVERY \
        --early_stop $EARLY_STOP; then
        
        # 提取测试准确率
        RESULT_JSON="${RESULTS_DIR}/${DATASET}/*/final_results.json"
        # 使用最新的结果文件
        LATEST_RESULT=$(ls -t $RESULT_JSON 2>/dev/null | head -1)
        
        if [ -n "$LATEST_RESULT" ] && [ -f "$LATEST_RESULT" ]; then
            # 使用python提取JSON中的准确率
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
