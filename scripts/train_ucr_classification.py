#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
M1实验：UCR单数据集分类训练脚本

验证"时间序列→TSLANet编码器→投影器→LLM"通路的分类能力。
使用soft prompt风格（LLaVA范式）进行指令式分类。

使用方法:
    python scripts/train_ucr_classification.py \
        --dataset ECG5000 \
        --encoder_type tslanet \
        --encoder_pretrained pretrained/tslanet_ucr98.pt \
        --epochs 30 \
        --batch_size 8 \
        --use_lora

特性:
- 支持TSLANet/TransformerCNN编码器切换
- 自动添加类别专用token（<cls_0>, <cls_1>, ...）
- 约束解码确保只输出有效标签
- LoRA微调（可选）
- 编码器和投影器也参与训练
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import (
    UCRClassificationDataset,
    collate_fn_classification,
)


def parse_args():
    parser = argparse.ArgumentParser(description="M1实验：UCR单数据集分类训练")
    
    # 数据相关
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="UCR数据集名称，如ECG5000",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="UCR数据根目录",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/m1_classification",
        help="结果保存目录",
    )
    
    # 模型配置
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="tslanet",
        choices=["tslanet", "transformer_cnn"],
        help="编码器类型",
    )
    parser.add_argument(
        "--encoder_pretrained",
        type=str,
        default=None,
        help="编码器预训练权重路径",
    )
    parser.add_argument(
        "--llm_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="LLM模型ID",
    )
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr_encoder", type=float, default=2e-4, help="编码器学习率")
    parser.add_argument("--lr_projector", type=float, default=1e-4, help="投影器学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="warmup比例")
    parser.add_argument("--early_stop", type=int, default=5, help="早停耐心值")
    
    # LoRA配置
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--dry_run", action="store_true", help="干运行模式")
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_optimizer(model: OpenTSLMSP, args) -> optim.Optimizer:
    """创建优化器，为不同组件设置不同学习率"""
    param_groups = [
        {
            "params": model.encoder.parameters(),
            "lr": args.lr_encoder,
            "weight_decay": args.weight_decay,
            "name": "encoder",
        },
        {
            "params": model.projector.parameters(),
            "lr": args.lr_projector,
            "weight_decay": args.weight_decay,
            "name": "projector",
        },
    ]
    
    # 如果使用LoRA，添加LoRA参数
    if args.use_lora and model.lora_enabled:
        lora_params = model.get_lora_parameters()
        if lora_params:
            param_groups.append({
                "params": lora_params,
                "lr": args.lr_projector,  # 使用投影器学习率
                "weight_decay": args.weight_decay,
                "name": "lora",
            })
    
    return optim.AdamW(param_groups)


def train_one_epoch(
    model: OpenTSLMSP,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    args,
) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        optimizer.zero_grad()
        
        # 计算损失
        loss = model.compute_loss(batch)
        loss.backward()
        
        # 梯度裁剪
        clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
        if args.dry_run and num_batches >= 2:
            break
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: OpenTSLMSP,
    data_loader: DataLoader,
    cls_token_ids: List[int],
    args,
) -> Dict[str, Any]:
    """评估模型"""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    labels = []
    
    for batch in tqdm(data_loader, desc="Evaluating"):
        # 计算损失
        loss = model.compute_loss(batch)
        total_loss += loss.item()
        
        # 生成预测（使用约束解码）
        outputs = model.generate(
            batch,
            max_new_tokens=1,
            allowed_token_ids=cls_token_ids,
        )
        
        # 解析预测结果
        for i, (output, sample) in enumerate(zip(outputs, batch)):
            true_label = sample["label_idx"]
            
            # 从输出中提取预测的类别token
            pred_label = -1
            for j, token_id in enumerate(cls_token_ids):
                token = model.tokenizer.convert_ids_to_tokens(token_id)
                if token in output:
                    pred_label = j
                    break
            
            predictions.append(pred_label)
            labels.append(true_label)
            
            if pred_label == true_label:
                correct += 1
            total += 1
        
        if args.dry_run:
            break
    
    accuracy = correct / max(total, 1)
    avg_loss = total_loss / max(len(data_loader), 1)
    
    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "predictions": predictions,
        "labels": labels,
        "correct": correct,
        "total": total,
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("M1实验：UCR单数据集分类训练")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"编码器: {args.encoder_type}")
    print(f"LLM: {args.llm_id}")
    print(f"使用LoRA: {args.use_lora}")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，使用CPU")
        args.device = "cpu"
    device = args.device
    
    # 获取数据集类别数
    num_classes = UCRClassificationDataset.get_num_classes(
        args.dataset, 
        raw_data_path=args.data_path
    )
    cls_tokens = UCRClassificationDataset.get_class_tokens(num_classes)
    print(f"\n📊 数据集信息:")
    print(f"   类别数: {num_classes}")
    print(f"   类别tokens: {cls_tokens[:5]}..." if len(cls_tokens) > 5 else f"   类别tokens: {cls_tokens}")
    
    # 创建模型
    print("\n🔧 创建模型...")
    model = OpenTSLMSP(
        llm_id=args.llm_id,
        device=device,
        encoder_type=args.encoder_type,
        encoder_pretrained_path=args.encoder_pretrained,
    )
    
    # 添加分类token
    cls_token_ids = model.add_classification_tokens(cls_tokens)
    
    # 启用LoRA（如果需要）
    if args.use_lora:
        print("\n🔧 启用LoRA...")
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    
    # 创建数据集
    print("\n📂 加载数据...")
    eos_token = model.get_eos_token()
    
    train_dataset = UCRClassificationDataset(
        dataset_name=args.dataset,
        split="train",
        EOS_TOKEN=eos_token,
        cls_tokens=cls_tokens,
        raw_data_path=args.data_path,
    )
    
    val_dataset = UCRClassificationDataset(
        dataset_name=args.dataset,
        split="validation",
        EOS_TOKEN=eos_token,
        cls_tokens=cls_tokens,
        raw_data_path=args.data_path,
    )
    
    test_dataset = UCRClassificationDataset(
        dataset_name=args.dataset,
        split="test",
        EOS_TOKEN=eos_token,
        cls_tokens=cls_tokens,
        raw_data_path=args.data_path,
    )
    
    # 确定patch_size
    patch_size = model.patch_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn_classification(b, patch_size=patch_size),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn_classification(b, patch_size=patch_size),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn_classification(b, patch_size=patch_size),
    )
    
    # 创建优化器和调度器
    optimizer = create_optimizer(model, args)
    
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # 创建保存目录
    save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练循环
    print("\n🚀 开始训练...")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, args)
        
        # 验证
        val_results = evaluate(model, val_loader, cls_token_ids, args)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_results['loss']:.4f} | Val Acc: {val_results['accuracy']:.4f}")
        
        # 保存最佳模型
        if val_results["accuracy"] > best_val_acc:
            best_val_acc = val_results["accuracy"]
            best_epoch = epoch
            patience_counter = 0
            
            # 保存checkpoint
            checkpoint = {
                "epoch": epoch,
                "encoder_state": model.encoder.state_dict(),
                "projector_state": model.projector.state_dict(),
                "val_accuracy": val_results["accuracy"],
                "val_loss": val_results["loss"],
                "train_loss": train_loss,
                "cls_tokens": cls_tokens,
                "args": vars(args),
            }
            
            # 保存LoRA状态
            if args.use_lora:
                model.save_lora_state_to_checkpoint(checkpoint)
            
            checkpoint_path = os.path.join(save_dir, "best_model.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 保存最佳模型到: {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"   (无改进, patience: {patience_counter}/{args.early_stop})")
        
        # 早停
        if patience_counter >= args.early_stop:
            print(f"\n⏹️ 早停! 验证准确率 {args.early_stop} 轮未改进")
            break
        
        if args.dry_run:
            print("\n🧪 干运行模式，提前退出")
            break
    
    # 加载最佳模型进行测试
    print("\n📊 加载最佳模型进行测试...")
    checkpoint_path = os.path.join(save_dir, "best_model.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.encoder.load_state_dict(checkpoint["encoder_state"])
        model.projector.load_state_dict(checkpoint["projector_state"])
        if args.use_lora and "lora_state" in checkpoint:
            model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)
    
    # 测试
    test_results = evaluate(model, test_loader, cls_token_ids, args)
    
    print("\n" + "=" * 60)
    print("📊 最终结果")
    print("=" * 60)
    print(f"最佳Epoch: {best_epoch}")
    print(f"验证准确率: {best_val_acc:.4f}")
    print(f"测试准确率: {test_results['accuracy']:.4f}")
    print(f"测试样本: {test_results['correct']}/{test_results['total']}")
    
    # 保存结果
    results = {
        "dataset": args.dataset,
        "encoder_type": args.encoder_type,
        "use_lora": args.use_lora,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_results["accuracy"],
        "test_correct": test_results["correct"],
        "test_total": test_results["total"],
        "num_classes": num_classes,
        "args": vars(args),
    }
    
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 结果保存到: {results_path}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
