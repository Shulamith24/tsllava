# PatchTST + VisionEncoder 双分支时序分类模型

本项目实现了一个用于时间序列分类的双分支模型，结合了：
- **PatchTST**: 基于 Transformer 的时序编码器
- **VisionEncoder**: 将时序转换为图像并使用预训练 ViT（如 DINOv2）编码

## 项目结构

```
patchtst_dual_branch/
├── src/
│   ├── __init__.py              # 包初始化
│   ├── dual_branch_model.py     # 双分支模型主体
│   ├── vision_encoder.py        # 视觉编码器（TiViT风格）
│   ├── aggregator.py            # Transformer聚合器
│   ├── projector.py             # 投影层模块
│   ├── ucr_dataset.py           # UCR数据集加载
│   ├── ucr_loader.py            # UCR数据加载器
│   └── train_dual_branch_tivit.py  # 训练脚本
├── scripts/
│   └── run_dual_branch_tivit_datasets.sh  # 批量训练脚本
├── data/                        # UCR数据目录
├── results/                     # 结果保存目录
├── requirements.txt             # 依赖包
└── README.md                    # 本文件
```

## 安装

```bash
# 克隆项目
cd patchtst_dual_branch

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 单数据集训练

```bash
# 基础训练
python -m src.train_dual_branch_tivit --dataset Adiac --epochs 50

# 启用 FP16 和梯度累积
python -m src.train_dual_branch_tivit --dataset Adiac --fp16 --gradient_accumulation_steps 4

# 仅使用时序分支
python -m src.train_dual_branch_tivit --dataset Adiac --branch_mode ts_only

# 仅使用视觉分支
python -m src.train_dual_branch_tivit --dataset Adiac --branch_mode vision_only
```

### 批量训练（全部 UCR 数据集）

```bash
# Linux/macOS
bash scripts/run_dual_branch_tivit_datasets.sh

# 使用多 GPU
bash scripts/run_dual_branch_tivit_datasets.sh --ddp --gpus 2

# 自定义参数
bash scripts/run_dual_branch_tivit_datasets.sh --epochs 50 --branch_mode both
```

## 核心参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | Adiac | UCR 数据集名称 |
| `--branch_mode` | both | 分支模式: both/ts_only/vision_only |
| `--vit_model_name` | facebook/dinov2-base | ViT 模型 |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 16 | 批次大小 |
| `--fp16` | False | 启用 FP16 混合精度 |
| `--gradient_accumulation_steps` | 1 | 梯度累积步数 |
| `--freeze_vision_backbone` | True | 冻结视觉 backbone |
| `--aggregator_layers` | 1 | 聚合头层数 |

## 模型架构

```
时间序列 [B, T, 1]
     │
     ├── PatchTST Branch ──────────────────┐
     │   └── [B, num_ts_patches, d_model]  │
     │                                      │
     └── Vision Branch ────────────────────┤
         ├── ts2image: [B, 3, 224, 224]    │
         └── ViT: [B, num_vit_patches, H]  │
                                           │
              Concat: [B, total_patches, H]
                         │
                    [ANS] Token
                         │
                   Aggregator
                         │
                  Classifier Head
                         │
                   Logits [B, C]
```

## 许可证

MIT License
