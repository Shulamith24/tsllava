#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
M1: UEA多变量数据集分类训练

验证OpenTSLMSP架构在UEA多变量数据集上的有监督分类能力。
使用LLaVA范式（Soft Prompt）进行指令式分类。

使用方法：
    python scripts/train_uea_classification.py \
        --dataset Epilepsy \
        --encoder_type tslanet \
        --epochs 30 \
        --batch_size 4 \
        --use_lora

训练配置：
- LoRA: r=16, alpha=32 (可选)
- Encoder LR: 2e-4
- Projector LR: 1e-4
- LoRA LR: 1e-4
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.time_series_datasets.uea.UEAClassificationDataset import UEAClassificationDataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE


def parse_args():
    parser = argparse.ArgumentParser(description="M1: UEA多变量数据集分类训练")
    
    # 数据相关
    parser.add_argument(
        "--dataset",
        type=str,
        default="Epilepsy",
        help="UEA数据集名称",
    )
    
    # 模型相关
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="tslanet",
        choices=["transformer_cnn", "tslanet"],
        help="编码器类型",
    )
    parser.add_argument(
        "--encoder_pretrained",
        type=str,
        default=None,
        help="TSLANet预训练权重路径",
    )
    parser.add_argument(
        "--llm_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="LLM模型ID",
    )
    
    # LoRA相关
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    # 训练相关
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--lr_encoder", type=float, default=2e-4, help="编码器学习率")
    parser.add_argument("--lr_projector", type=float, default=1e-4, help="投影层学习率")
    parser.add_argument("--lr_lora", type=float, default=1e-4, help="LoRA学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热比例")
    
    # 保存相关
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/m1_uea_classification",
        help="结果保存目录",
    )
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--eval_every", type=int, default=5, help="每N轮评估一次")
    parser.add_argument("--early_stop", type=int, default=10, help="早停耐心值")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="生成最大token数")
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_accuracy(predictions: List[str], labels: List[str]) -> float:
    """计算分类准确率"""
    correct = 0
    for pred, label in zip(predictions, labels):
        pred_clean = pred.strip()
        pred_label = None
        
        if len(pred_clean) == 1 and pred_clean.isalpha():
            pred_label = pred_clean.upper()
        elif pred_clean:
            words = pred_clean.split()
            if words:
                last_word = words[-1].strip(".,!?:;")
                if last_word and last_word[0].isalpha():
                    pred_label = last_word[0].upper()
        
        label_clean = label.strip().upper()
        if pred_label == label_clean:
            correct += 1
    
    return correct / len(predictions) if predictions else 0.0


def create_data_loaders(args, eos_token: str):
    """创建数据加载器"""
    train_dataset = UEAClassificationDataset(
        split="train",
        EOS_TOKEN=eos_token,
        dataset_name=args.dataset,
    )
    
    val_dataset = UEAClassificationDataset(
        split="validation",
        EOS_TOKEN=eos_token,
        dataset_name=args.dataset,
    )
    
    test_dataset = UEAClassificationDataset(
        split="test",
        EOS_TOKEN=eos_token,
        dataset_name=args.dataset,
    )
    
    def collate_fn(batch):
        return extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: OpenTSLMSP,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    grad_clip: float,
    device: str,
    epoch: int,
    num_epochs: int,
) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    for batch in pbar:
        loss = model.compute_loss(batch)
        
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: OpenTSLMSP,
    data_loader: DataLoader,
    max_new_tokens: int,
    desc: str = "Evaluating",
) -> Dict[str, Any]:
    """评估模型"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(data_loader, desc=desc):
        loss = model.compute_loss(batch)
        total_loss += loss.item()
        num_batches += 1
        
        predictions = model.generate(batch, max_new_tokens=max_new_tokens)
        
        for sample, pred in zip(batch, predictions):
            all_predictions.append(pred)
            all_labels.append(sample["answer"].replace(model.get_eos_token(), "").strip())
    
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = calculate_accuracy(all_predictions, all_labels)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def save_checkpoint(
    model: OpenTSLMSP,
    optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    val_acc: float,
    save_path: str,
    args,
):
    """保存checkpoint"""
    checkpoint = {
        "encoder_state": model.encoder.state_dict(),
        "projector_state": model.projector.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "args": vars(args),
    }
    
    model.save_lora_state_to_checkpoint(checkpoint)
    torch.save(checkpoint, save_path)
    print(f"💾 Saved checkpoint to: {save_path}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("M1: UEA多变量数据集分类训练")
    print("=" * 60)
    print(f"时间: {datetime.datetime.now()}")
    print(f"数据集: {args.dataset}")
    print(f"编码器: {args.encoder_type}")
    print(f"LoRA: {args.use_lora}")
    print("=" * 60)
    
    set_seed(args.seed)
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，使用CPU")
        args.device = "cpu"
    device = args.device
    
    save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("\n🔧 创建模型...")
    tslanet_config = {"patch_size": 4}
    
    model = OpenTSLMSP(
        llm_id=args.llm_id,
        device=device,
        encoder_type=args.encoder_type,
        encoder_pretrained_path=args.encoder_pretrained,
        tslanet_config=tslanet_config if args.encoder_type == "tslanet" else None,
    )
    
    if args.use_lora:
        print("📎 启用LoRA...")
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    
    print("\n📂 加载数据...")
    eos_token = model.get_eos_token()
    train_loader, val_loader, test_loader = create_data_loaders(args, eos_token)
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    print("\n⚙️ 创建优化器...")
    param_groups = [
        {"params": model.encoder.parameters(), "lr": args.lr_encoder},
        {"params": model.projector.parameters(), "lr": args.lr_projector},
    ]
    
    if args.use_lora:
        lora_params = model.get_lora_parameters()
        if lora_params:
            param_groups.append({"params": lora_params, "lr": args.lr_lora})
    
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    
    print("\n🚀 开始训练...")
    best_val_acc = 0.0
    patience_counter = 0
    loss_history = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            args.grad_clip, device, epoch, args.epochs
        )
        
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print(f"\n📊 Epoch {epoch} 评估...")
            
            val_results = evaluate(model, val_loader, args.max_new_tokens, "Validating")
            val_loss = val_results["loss"]
            val_acc = val_results["accuracy"]
            
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val Accuracy: {val_acc:.4f}")
            
            print("   Sample predictions:")
            for i in range(min(3, len(val_results["predictions"]))):
                pred = val_results["predictions"][i]
                label = val_results["labels"][i]
                pred_short = pred[-50:] if len(pred) > 50 else pred
                print(f"     Pred: '{pred_short}' | Label: '{label}'")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_loss, val_acc,
                    os.path.join(save_dir, "best_model.pt"),
                    args
                )
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
            print(f"\n⏹️ 早停! 验证准确率 {args.early_stop} 轮未改进")
            break
    
    print("\n" + "=" * 60)
    print("📋 最终测试评估...")
    
    best_ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device, weights_only=False)
    model.encoder.load_state_dict(best_ckpt["encoder_state"])
    model.projector.load_state_dict(best_ckpt["projector_state"])
    model.load_lora_state_from_checkpoint(best_ckpt, allow_missing=True)
    
    test_results = evaluate(model, test_loader, args.max_new_tokens, "Testing")
    
    print(f"\n✅ 测试结果:")
    print(f"   Test Loss: {test_results['loss']:.4f}")
    print(f"   Test Accuracy: {test_results['accuracy']:.4f}")
    
    final_results = {
        "dataset": args.dataset,
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
    
    print("=" * 60)
    print(f"结果保存到: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
