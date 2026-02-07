#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
SpecVisNet 独立训练脚本 - UCR 时序分类

专注于 SpecVisNet 模型的训练、优化和分析。

使用方法：
    # 基础用法
    python -m patchtst_ucr.train_specvisnet --dataset ECG200 --epochs 50

    # 使用 BF16 混合精度
    python -m patchtst_ucr.train_specvisnet --dataset ECG200 --bf16

    # 禁用 FAM 和 ASB（消融实验）
    python -m patchtst_ucr.train_specvisnet --dataset ECG200 --no_fam --no_asb

    # 使用 ConvNeXt 骨干
    python -m patchtst_ucr.train_specvisnet --dataset ECG200 --backbone convnext_tiny
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

# 添加 src 目录到路径
script_dir = Path(__file__).parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from patchtst_ucr.specvisnet import SpecVisNetEncoder
from patchtst_ucr.ucr_dataset import UCRDatasetForPatchTST, get_dataset_info


class SpecVisNetClassifier(nn.Module):
    """
    SpecVisNet 分类模型
    
    将 SpecVisNetEncoder 与分类头结合，用于时序分类任务。
    
    Args:
        num_classes: 类别数量
        backbone: 骨干网络类型
        num_scales: 小波尺度数
        learnable_wavelet: 是否学习小波参数
        use_fam: 是否使用频率注意力模块
        use_asb: 是否使用自适应频谱块
        finetune: 是否微调骨干网络
        dropout: 分类头 Dropout
        pooling: 池化方式 ('cls', 'mean', 'max')
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = "swin_tiny",
        num_scales: int = 64,
        learnable_wavelet: bool = True,
        use_fam: bool = True,
        use_asb: bool = True,
        finetune: bool = False,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        
        self.pooling = pooling
        
        # SpecVisNet 编码器
        self.encoder = SpecVisNetEncoder(
            backbone=backbone,
            num_scales=num_scales,
            learnable_wavelet=learnable_wavelet,
            use_fam=use_fam,
            use_asb=use_asb,
            finetune=finetune,
        )
        
        self.hidden_size = self.encoder.hidden_size
        self.num_patches = self.encoder.num_patches
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_classes),
        )
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, L, 1] 输入时序
            labels: [B] 标签（可选）
            
        Returns:
            包含 logits 和可选 loss 的字典
        """
        # 编码
        features = self.encoder(x)  # [B, N, C]
        
        # 池化
        if self.pooling == "mean":
            pooled = features.mean(dim=1)  # [B, C]
        elif self.pooling == "max":
            pooled = features.max(dim=1)[0]  # [B, C]
        elif self.pooling == "cls":
            pooled = features[:, 0, :]  # [B, C]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # 分类
        logits = self.classifier(pooled)  # [B, num_classes]
        
        outputs = {"logits": logits}
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            outputs["loss"] = loss
        
        return outputs
    
    def count_parameters(self) -> Dict[str, int]:
        """统计参数量"""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "encoder": encoder_params,
            "classifier": classifier_params,
            "trainable": trainable_params,
            "total": total_params,
        }
    
    def get_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            "hidden_size": self.hidden_size,
            "num_patches": self.num_patches,
            "pooling": self.pooling,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="SpecVisNet UCR 时序分类训练")

    # 数据相关
    parser.add_argument("--dataset", type=str, default="ECG200", help="UCR 数据集名称")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR 数据根目录")
    
    # SpecVisNet 模型配置
    parser.add_argument("--backbone", type=str, default="swin_tiny",
                       choices=["swin_tiny", "swin_small", "convnext_tiny"],
                       help="骨干网络类型")
    parser.add_argument("--num_scales", type=int, default=64,
                       help="小波尺度数（时频图高度）")
    parser.add_argument("--learnable_wavelet", action="store_true", default=True,
                       help="使用可学习小波参数")
    parser.add_argument("--no_learnable_wavelet", dest="learnable_wavelet", action="store_false")
    parser.add_argument("--use_fam", action="store_true", default=True,
                       help="使用频率注意力模块")
    parser.add_argument("--no_fam", dest="use_fam", action="store_false")
    parser.add_argument("--use_asb", action="store_true", default=True,
                       help="使用自适应频谱块")
    parser.add_argument("--no_asb", dest="use_asb", action="store_false")
    parser.add_argument("--finetune", action="store_true",
                       help="微调骨干网络")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="分类头 Dropout")
    parser.add_argument("--pooling", type=str, default="mean",
                       choices=["mean", "max", "cls"],
                       help="特征池化方式")
    
    # 训练相关
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    
    # 显存优化
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                       help="梯度累积步数")
    parser.add_argument("--fp16", action="store_true",
                       help="使用 FP16 混合精度训练")
    parser.add_argument("--bf16", action="store_true",
                       help="使用 BF16 混合精度训练")
    
    # 保存与评估
    parser.add_argument("--save_dir", type=str, default="results/specvisnet",
                       help="结果保存目录")
    parser.add_argument("--eval_every", type=int, default=5, help="每 N 轮评估一次")
    parser.add_argument("--early_stop", type=int, default=15, help="早停耐心值")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="评估批次大小")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    import random
    import numpy as np
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def prepare_batch(batch: List[Dict], device: str):
    """
    将 UCR 批次转换为模型输入格式
    
    Returns:
        past_values: [B, L, 1]
        labels: [B]
    """
    past_values_list = []
    labels = []
    
    for sample in batch:
        ts = sample["time_series"][0]
        if not isinstance(ts, torch.Tensor):
            ts = torch.tensor(ts, dtype=torch.float32)
        past_values_list.append(ts.unsqueeze(-1))  # [L, 1]
        labels.append(sample["int_label"])
    
    # 找到最长序列并 padding
    max_len = max(v.shape[0] for v in past_values_list)
    padded = []
    for v in past_values_list:
        if v.shape[0] < max_len:
            pad = torch.zeros(max_len - v.shape[0], 1)
            v = torch.cat([v, pad], dim=0)
        padded.append(v)
    
    past_values = torch.stack(padded, dim=0).to(device)  # [B, L, 1]
    labels = torch.tensor(labels, device=device, dtype=torch.long)
    
    return past_values, labels


def create_data_loaders(args):
    """创建数据加载器"""
    train_dataset = UCRDatasetForPatchTST(
        dataset_name=args.dataset,
        split="train",
        raw_data_path=args.data_path,
    )
    
    val_dataset = UCRDatasetForPatchTST(
        dataset_name=args.dataset,
        split="validation",
        raw_data_path=args.data_path,
    )
    
    test_dataset = UCRDatasetForPatchTST(
        dataset_name=args.dataset,
        split="test",
        raw_data_path=args.data_path,
    )
    
    def collate_fn(batch):
        return batch
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    grad_clip: float,
    device: str,
    epoch: int,
    num_epochs: int,
    grad_accum_steps: int = 1,
    scaler = None,
    use_amp: bool = False,
    amp_dtype = torch.float16,
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    for batch_idx, batch in enumerate(pbar):
        past_values, labels = prepare_batch(batch, device)
        
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            outputs = model(past_values, labels=labels)
            loss = outputs["loss"] / grad_accum_steps
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{loss.item() * grad_accum_steps:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model,
    data_loader: DataLoader,
    device: str,
    desc: str = "Evaluating",
) -> Dict[str, Any]:
    """评估模型"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(data_loader, desc=desc):
        past_values, labels = prepare_batch(batch, device)
        
        outputs = model(past_values, labels=labels)
        
        total_loss += outputs["loss"].item()
        num_batches += 1
        
        predictions = torch.argmax(outputs["logits"], dim=-1)
        
        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    avg_loss = total_loss / max(num_batches, 1)
    correct = sum(p == l for p, l in zip(all_predictions, all_labels))
    accuracy = correct / len(all_labels) if all_labels else 0.0
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def main():
    args = parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("SpecVisNet UCR 时序分类训练")
    print("=" * 70)
    print(f"时间: {datetime.datetime.now()}")
    print(f"数据集: {args.dataset}")
    print(f"骨干网络: {args.backbone}")
    print(f"可学习小波: {args.learnable_wavelet}")
    print(f"FAM: {args.use_fam}")
    print(f"ASB: {args.use_asb}")
    print(f"微调骨干: {args.finetune}")
    print("=" * 70)
    
    set_seed(args.seed)
    
    print(f"\n使用设备: {device}")
    print("\n📂 分析数据集...")
    
    num_classes, max_length = get_dataset_info(args.dataset, args.data_path)
    
    print(f"   类别数: {num_classes}")
    print(f"   最大长度: {max_length}")
    
    # 创建保存目录
    config_str = f"{args.backbone}"
    if not args.use_fam:
        config_str += "_nofam"
    if not args.use_asb:
        config_str += "_noasb"
    if args.finetune:
        config_str += "_ft"
    
    save_dir = os.path.join(args.save_dir, args.dataset, config_str)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("\n🔧 创建模型...")
    
    model = SpecVisNetClassifier(
        num_classes=num_classes,
        backbone=args.backbone,
        num_scales=args.num_scales,
        learnable_wavelet=args.learnable_wavelet,
        use_fam=args.use_fam,
        use_asb=args.use_asb,
        finetune=args.finetune,
        dropout=args.dropout,
        pooling=args.pooling,
    ).to(device)
    
    param_counts = model.count_parameters()
    print(f"参数量统计:")
    print(f"  - 编码器: {param_counts['encoder']:,}")
    print(f"  - 分类头: {param_counts['classifier']:,}")
    print(f"  - 可训练: {param_counts['trainable']:,}")
    print(f"  - 总计: {param_counts['total']:,}")
    
    print("\n📂 加载数据...")
    
    train_loader, val_loader, test_loader = create_data_loaders(args)
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    print("\n⚙️  创建优化器...")
    
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    
    # 混合精度设置
    use_amp = args.fp16 or args.bf16
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    if use_amp:
        print(f"   ⚡ 混合精度: {'BF16' if args.bf16 else 'FP16'}")
    if args.grad_accum_steps > 1:
        print(f"   📊 梯度累积: {args.grad_accum_steps} 步")
    
    print("\n🚀 开始训练...")
    
    best_val_acc = 0.0
    patience_counter = 0
    loss_history = []
    
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                args.grad_clip, device, epoch, args.epochs,
                grad_accum_steps=args.grad_accum_steps,
                scaler=scaler,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                print(f"\n📊 Epoch {epoch} 评估...")
                
                val_results = evaluate(model, val_loader, device, "Validating")
                val_loss = val_results["loss"]
                val_acc = val_results["accuracy"]
                
                print(f"   Train Loss: {train_loss:.4f}")
                print(f"   Val Loss: {val_loss:.4f}")
                print(f"   Val Accuracy: {val_acc:.4f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    
                    checkpoint = {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "config": model.get_config(),
                        "args": vars(args),
                    }
                    torch.save(checkpoint, os.path.join(save_dir, "best_model.pt"))
                    print(f"   💾 保存最佳模型")
                else:
                    patience_counter += 1
                    print(f"   (无改进, patience: {patience_counter}/{args.early_stop})")
                
                loss_history.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                })
                with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
                    json.dump(loss_history, f, indent=2)
            else:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            if patience_counter >= args.early_stop:
                print(f"\n⏹️  早停! 验证准确率 {args.early_stop} 轮未改进")
                break
        
        # 最终测试评估
        print("\n" + "=" * 70)
        print("📋 最终测试评估...")
        
        best_ckpt = torch.load(
            os.path.join(save_dir, "best_model.pt"),
            map_location=device,
            weights_only=False
        )
        model.load_state_dict(best_ckpt["model_state"])
        
        test_results = evaluate(model, test_loader, device, "Testing")
        
        print(f"\n✅ 测试结果:")
        print(f"   Test Loss: {test_results['loss']:.4f}")
        print(f"   Test Accuracy: {test_results['accuracy']:.4f}")
        
        final_results = {
            "dataset": args.dataset,
            "num_classes": num_classes,
            "backbone": args.backbone,
            "use_fam": args.use_fam,
            "use_asb": args.use_asb,
            "finetune": args.finetune,
            "total_params": param_counts["total"],
            "trainable_params": param_counts["trainable"],
            "best_val_acc": best_val_acc,
            "test_loss": test_results["loss"],
            "test_accuracy": test_results["accuracy"],
            "epochs_trained": epoch,
        }
        
        with open(os.path.join(save_dir, "final_results.json"), "w") as f:
            json.dump(final_results, f, indent=2)
        
        with open(os.path.join(save_dir, "test_predictions.json"), "w") as f:
            json.dump({
                "predictions": test_results["predictions"],
                "labels": test_results["labels"],
            }, f, indent=2)
        
        print("=" * 70)
        print(f"结果保存到: {save_dir}")
        print("=" * 70)
    
    except KeyboardInterrupt:
        print("\n⚠️  训练被中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
