#!/bin/bash
# SpecVisNet UCR 数据集批量训练脚本
#
# 用法:
#   bash run_specvisnet_datasets.sh [backbone] [num_gpus]
#
# 参数:
#   backbone: swin_tiny (默认), swin_small, convnext_tiny
#   num_gpus: GPU 数量 (目前仅支持单 GPU)
#
# 示例:
#   bash run_specvisnet_datasets.sh                    # swin_tiny, 单 GPU
#   bash run_specvisnet_datasets.sh swin_small         # swin_small
#   bash run_specvisnet_datasets.sh convnext_tiny      # convnext_tiny

set -e

# ========== 配置 ==========
BACKBONE=${1:-"swin_tiny"}
DATA_PATH="./data"
SAVE_DIR="results/specvisnet"

# 训练参数
EPOCHS=50
BATCH_SIZE=16
EVAL_BATCH_SIZE=32
LR=1e-4
EARLY_STOP=15

# 显存优化: 启用 BF16
USE_BF16=true
GRAD_ACCUM_STEPS=1

# UCR 数据集列表 (可根据需要修改)
DATASETS=(
    # 小型数据集 (快速测试)
    "ECG200"
    "GunPoint"
    "Wafer"
    "FaceFour"
    "Lightning2"
    "Lightning7"
    "Adiac"
    "Beef"
    "CBF"
    "ChlorineConcentration"
    "Coffee"
    "DiatomSizeReduction"
    "ECGFiveDays"
    "FaceAll"
    "FacesUCR"
    "Fish"
    "Haptics"
    "InlineSkate"
    "ItalyPowerDemand"
    "MedicalImages"
    "MoteStrain"
    "OliveOil"
    "OSULeaf"
    "SonyAIBORobotSurface1"
    "SonyAIBORobotSurface2"
    "SwedishLeaf"
    "Symbols"
    "SyntheticControl"
    "Trace"
    "TwoLeadECG"
    "TwoPatterns"
    "WordSynonyms"
    "Yoga"
)

# ========== 检查环境 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "SpecVisNet UCR 批量训练"
echo "=========================================="
echo "骨干网络: $BACKBONE"
echo "数据路径: $DATA_PATH"
echo "保存目录: $SAVE_DIR"
echo "数据集数量: ${#DATASETS[@]}"
echo "=========================================="

# 检查数据目录
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ 数据目录不存在: $DATA_PATH"
    exit 1
fi

# ========== 训练循环 ==========
TOTAL=${#DATASETS[@]}
SUCCESS=0
FAILED=0
SKIPPED=0

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    IDX=$((i + 1))
    
    echo ""
    echo "=========================================="
    echo "[$IDX/$TOTAL] 训练: $DATASET"
    echo "=========================================="
    
    # 构造额外参数
    EXTRA_ARGS=""
    if $USE_BF16; then
        EXTRA_ARGS="$EXTRA_ARGS --bf16"
    fi
    if [ "$GRAD_ACCUM_STEPS" -gt 1 ]; then
        EXTRA_ARGS="$EXTRA_ARGS --grad_accum_steps $GRAD_ACCUM_STEPS"
    fi
    
    # 运行训练
    if python -m patchtst_ucr.train_specvisnet \
        --dataset "$DATASET" \
        --data_path "$DATA_PATH" \
        --backbone "$BACKBONE" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --eval_batch_size $EVAL_BATCH_SIZE \
        --lr $LR \
        --early_stop $EARLY_STOP \
        --save_dir "$SAVE_DIR" \
        $EXTRA_ARGS; then
        echo "✅ $DATASET 训练完成"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "❌ $DATASET 训练失败"
        FAILED=$((FAILED + 1))
    fi
done

# ========== 汇总结果 ==========
echo ""
echo "=========================================="
echo "训练完成汇总"
echo "=========================================="
echo "成功: $SUCCESS / $TOTAL"
echo "失败: $FAILED / $TOTAL"
echo "跳过: $SKIPPED / $TOTAL"
echo "=========================================="

# 汇总所有结果到一个文件
SUMMARY_FILE="$SAVE_DIR/summary_${BACKBONE}.json"
echo "正在生成汇总报告: $SUMMARY_FILE"

python -c "
import os
import json
import glob

save_dir = '$SAVE_DIR'
backbone = '$BACKBONE'
results = []

for dataset_dir in sorted(glob.glob(os.path.join(save_dir, '*', backbone + '*'))):
    result_file = os.path.join(dataset_dir, 'final_results.json')
    if os.path.exists(result_file):
        with open(result_file) as f:
            data = json.load(f)
            results.append(data)

if results:
    # 计算平均准确率
    avg_acc = sum(r['test_accuracy'] for r in results) / len(results)
    
    summary = {
        'backbone': backbone,
        'num_datasets': len(results),
        'average_accuracy': avg_acc,
        'results': results,
    }
    
    with open('$SUMMARY_FILE', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f'汇总完成: {len(results)} 个数据集, 平均准确率: {avg_acc:.4f}')
else:
    print('没有找到结果文件')
"

echo ""
echo "🎉 全部完成!"
