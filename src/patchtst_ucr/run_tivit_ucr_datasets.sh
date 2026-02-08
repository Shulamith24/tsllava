#!/bin/bash
# ==============================================================================
# TiViT UCR 全数据集批量训练脚本
# 
# 使用 Linear Probing 方式：冻结 ViT 提取特征 + sklearn LogisticRegression
#
# 用法：可在任意路径下执行此脚本
#   bash /path/to/run_tivit_ucr_datasets.sh
#
# 指定 ViT 模型：
#   bash /path/to/run_tivit_ucr_datasets.sh --vit_model facebook/dinov2-large
#
# 使用不同分类器：
#   bash /path/to/run_tivit_ucr_datasets.sh --classifier logistic_regression
# ==============================================================================

set -e  # 遇到错误立即退出

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 推导项目根目录 (src/patchtst_ucr -> src -> root)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "============================================================"
echo "TiViT UCR 批量训练 (Linear Probing)"
echo "============================================================"
echo "项目根目录: $PROJECT_ROOT"
echo "脚本目录: $SCRIPT_DIR"
echo "开始时间: $(date)"
echo "============================================================"

# 默认配置参数
DATA_PATH="${PROJECT_ROOT}/data"
UCR_DIR="${DATA_PATH}/UCRArchive_2018"
RESULTS_DIR="${PROJECT_ROOT}/results/tivit"

# ViT 模型配置
VIT_MODEL_NAME="facebook/dinov2-base"
VIT_LAYER=""  # 空表示使用最后一层
AGGREGATION="mean"

# 时序转图像配置
PATCH_SIZE_MODE="sqrt"
STRIDE="0.1"

# 分类器配置
CLASSIFIER_TYPE="logistic_regression"
MAX_ITER=500

# 其他配置
BATCH_SIZE=64
SEED=42

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --vit_model)
            VIT_MODEL_NAME="$2"
            shift 2
            ;;
        --vit_layer)
            VIT_LAYER="$2"
            shift 2
            ;;
        --aggregation)
            AGGREGATION="$2"
            shift 2
            ;;
        --patch_size_mode)
            PATCH_SIZE_MODE="$2"
            shift 2
            ;;
        --stride)
            STRIDE="$2"
            shift 2
            ;;
        --classifier)
            CLASSIFIER_TYPE="$2"
            shift 2
            ;;
        --max_iter)
            MAX_ITER="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            UCR_DIR="${DATA_PATH}/UCRArchive_2018"
            shift 2
            ;;
        --save_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 生成结果文件名
VIT_SHORT_NAME=$(echo "$VIT_MODEL_NAME" | sed 's/.*\///' | sed 's/-/_/g')
OUTPUT_FILE="${RESULTS_DIR}/all_datasets_accuracy_${VIT_SHORT_NAME}_${CLASSIFIER_TYPE}.txt"

# 创建结果目录
mkdir -p "$RESULTS_DIR"

# 检查UCR数据目录是否存在
if [ ! -d "$UCR_DIR" ]; then
    echo "错误: UCR数据目录不存在: $UCR_DIR"
    echo "请先下载 UCRArchive_2018 数据集"
    exit 1
fi

# 获取所有数据集名称
DATASETS=($(find "$UCR_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort))

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "错误: 未找到任何UCR数据集"
    exit 1
fi

echo "找到 ${#DATASETS[@]} 个数据集"
echo "ViT 模型: $VIT_MODEL_NAME"
echo "ViT 层: ${VIT_LAYER:-last}"
echo "聚合方式: $AGGREGATION"
echo "Patch Size: $PATCH_SIZE_MODE"
echo "Stride: $STRIDE"
echo "分类器: $CLASSIFIER_TYPE"
echo ""

# 初始化结果文件
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "创建新结果文件: $OUTPUT_FILE"
    echo "============================================================" > "$OUTPUT_FILE"
    echo "TiViT UCR 分类结果汇总 (Linear Probing)" >> "$OUTPUT_FILE"
    echo "时间: $(date)" >> "$OUTPUT_FILE"
    echo "ViT 模型: $VIT_MODEL_NAME" >> "$OUTPUT_FILE"
    echo "分类器: $CLASSIFIER_TYPE" >> "$OUTPUT_FILE"
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
    
    # 构建训练命令
    CMD="python -m src.patchtst_ucr.train_tivit \
        --dataset $DATASET \
        --data_path $DATA_PATH \
        --save_dir $RESULTS_DIR \
        --vit_model_name $VIT_MODEL_NAME \
        --aggregation $AGGREGATION \
        --patch_size_mode $PATCH_SIZE_MODE \
        --stride $STRIDE \
        --classifier_type $CLASSIFIER_TYPE \
        --max_iter $MAX_ITER \
        --batch_size $BATCH_SIZE \
        --seed $SEED"
    
    # 添加可选的 vit_layer 参数
    if [ -n "$VIT_LAYER" ]; then
        CMD="$CMD --vit_layer $VIT_LAYER"
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
