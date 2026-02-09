#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
PatchTST + VisionEncoder 双分支 LLM 分类训练脚本

使用 LLM (Llama-3.2-1B) 进行时间序列分类，支持：
- 双分支融合：PatchTST 时序编码 + ViT 图像编码
- DDP 分布式训练
- LoRA 微调
- 显存优化：FP16 混合精度、梯度累积、梯度检查点

使用方法：
    # 单卡训练
    uv run -m src.train_dual_branch_llm --dataset Adiac --epochs 30 --use_lora

    # 启用显存优化
    uv run -m src.train_dual_branch_llm --dataset Adiac --fp16 --gradient_accumulation_steps 4 --use_lora

    # 多卡 DDP 训练
    torchrun --nproc_per_node=2 -m src.train_dual_branch_llm --dataset Adiac --use_ddp --fp16 --use_lora
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .dual_branch_llm_model import DualBranchLLMModel
from .ucr_llm_dataset import UCRLLMClassificationDataset
from .ucr_dataset import get_dataset_info
from .model_config import PATCH_SIZE


def parse_args():
    parser = argparse.ArgumentParser(description="双分支 LLM 时序分类训练")

    # 数据相关
    parser.add_argument("--dataset", type=str, default="Adiac", help="UCR数据集名称")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR数据根目录")
    
    # LLM 相关
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B", help="LLM模型ID")
    
    # LoRA 相关
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    # 分支控制
    parser.add_argument("--branch_mode", type=str, default="both",
                       choices=["both", "ts_only", "vision_only"],
                       help="分支模式: both(双分支), ts_only(仅时序), vision_only(仅视觉)")
    
    # PatchTST 时序分支配置
    parser.add_argument("--context_length", type=int, default=None,
                       help="上下文长度（None则自动设置）")
    parser.add_argument("--patch_length", type=int, default=16, help="Patch 长度")
    parser.add_argument("--stride", type=int, default=8, help="Patch 步长")
    parser.add_argument("--d_model", type=int, default=128, help="PatchTST 模型维度")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="PatchTST Attention heads")
    parser.add_argument("--num_hidden_layers", type=int, default=3, help="PatchTST Transformer 层数")
    parser.add_argument("--ffn_dim", type=int, default=512, help="PatchTST FFN 维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    
    # Vision 分支配置
    parser.add_argument("--vit_model_name", type=str, default="facebook/dinov2-base",
                       help="ViT模型名称")
    parser.add_argument("--vit_layer_idx", type=int, default=-1, help="ViT 特征提取层索引")
    parser.add_argument("--vit_patch_size", type=int, default=16, help="时序图像化 patch 大小")
    parser.add_argument("--vit_stride", type=float, default=0.5, help="时序图像化步长比例")
    
    # 投影层配置
    parser.add_argument("--projector_type", type=str, default="mlp",
                       choices=["mlp", "linear"], help="投影层类型")
    parser.add_argument("--projector_dropout", type=float, default=0.1, help="投影层Dropout")
    
    # 冻结选项
    parser.add_argument("--freeze_ts_backbone", action="store_true", help="冻结 PatchTST backbone")
    parser.add_argument("--freeze_vision_backbone", action="store_true", default=True,
                       help="冻结 Vision backbone（默认开启）")
    parser.add_argument("--no_freeze_vision_backbone", action="store_true",
                       help="不冻结 Vision backbone")
    parser.add_argument("--freeze_encoder", action="store_true", help="冻结所有编码器")
    
    # 训练相关
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--lr_encoder", type=float, default=2e-4, help="编码器学习率")
    parser.add_argument("--lr_projector", type=float, default=1e-4, help="投影层学习率")
    parser.add_argument("--lr_lora", type=float, default=1e-4, help="LoRA学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热比例")
    
    # DDP 分布式训练
    parser.add_argument("--use_ddp", action="store_true", help="启用 DDP 分布式训练")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP local rank")
    
    # 显存优化
    parser.add_argument("--fp16", action="store_true", help="启用 FP16 混合精度")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="梯度累积步数")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="启用梯度检查点")
    
    # 保存相关
    parser.add_argument("--save_dir", type=str, default="results/dual_branch_llm",
                       help="结果保存目录")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--eval_every", type=int, default=5, help="每N轮评估一次")
    parser.add_argument("--early_stop", type=int, default=10, help="早停耐心值")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="生成最大token数")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="评估批次大小")
    
    args = parser.parse_args()
    
    # 处理冻结选项冲突
    if args.no_freeze_vision_backbone:
        args.freeze_vision_backbone = False
    
    return args


def set_seed(seed: int, rank: int = 0):
    """设置随机种子"""
    seed = seed + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def setup_ddp(args):
    """初始化 DDP"""
    if args.use_ddp:
        if args.local_rank == -1:
            args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        args.device = f"cuda:{args.local_rank}"
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        
        if args.rank == 0:
            print(f"🌐 DDP 初始化完成，world_size={args.world_size}")
    else:
        args.world_size = 1
        args.rank = 0


def cleanup_ddp(args):
    """清理 DDP"""
    if args.use_ddp:
        dist.destroy_process_group()


def is_main_process(args):
    """判断是否为主进程"""
    return args.rank == 0


def get_model(model):
    """获取底层模型（兼容DDP包装）"""
    return model.module if hasattr(model, "module") else model


def collate_fn(batch):
    """Collate 函数：处理变长时间序列"""
    # 直接返回 batch，在模型内部处理 padding
    return batch


def calculate_accuracy(predictions: List[str], labels: List[str]) -> float:
    """
    计算分类准确率
    
    对生成文本进行后处理，提取预测标签并与真实标签比较
    """
    correct = 0
    for pred, label in zip(predictions, labels):
        pred_clean = pred.strip()
        
        # 尝试提取 <cN> 格式的标签
        pred_label = None
        
        # 1. 查找 <cN> 模式
        import re
        match = re.search(r'<c(\d+)>', pred_clean)
        if match:
            pred_label = f"<c{match.group(1)}>"
        # 2. 如果预测就是单个字母
        elif len(pred_clean) == 1 and pred_clean.isalpha():
            pred_label = pred_clean.upper()
        # 3. 取最后一个 <cN> 或字母
        elif pred_clean:
            words = pred_clean.split()
            if words:
                last_word = words[-1].strip(".,!?:;")
                if last_word.startswith("<c") and last_word.endswith(">"):
                    pred_label = last_word
                elif last_word and last_word[0].isalpha():
                    pred_label = last_word[0].upper()
        
        # 比较
        label_clean = label.strip()
        if pred_label == label_clean:
            correct += 1
    
    return correct / len(predictions) if predictions else 0.0


def create_data_loaders(args, eos_token: str):
    """创建数据加载器"""
    train_dataset = UCRLLMClassificationDataset(
        split="train",
        EOS_TOKEN=eos_token,
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )
    
    val_dataset = UCRLLMClassificationDataset(
        split="validation",
        EOS_TOKEN=eos_token,
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )
    
    test_dataset = UCRLLMClassificationDataset(
        split="test",
        EOS_TOKEN=eos_token,
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )
    
    # DDP 采样器
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.use_ddp else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,
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
    
    return train_loader, val_loader, test_loader, train_sampler


def train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    grad_clip: float,
    epoch: int,
    num_epochs: int,
    args,
    scaler: GradScaler = None,
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()
    
    # DDP 设置 epoch
    if args.use_ddp and hasattr(train_loader, 'sampler') and train_loader.sampler is not None:
        train_loader.sampler.set_epoch(epoch)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", disable=not is_main_process(args))
    
    for step, batch in enumerate(pbar):
        # 混合精度训练
        if args.fp16:
            with autocast():
                loss = model(batch)
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:
            loss = model(batch)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
        
        # 梯度累积
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * args.gradient_accumulation_steps
        num_batches += 1
        
        if is_main_process(args):
            pbar.set_postfix({
                "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
    
    # 处理剩余梯度
    if num_batches % args.gradient_accumulation_steps != 0:
        if args.fp16:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model,
    data_loader: DataLoader,
    max_new_tokens: int,
    args,
    desc: str = "Evaluating",
) -> Dict[str, Any]:
    """评估模型"""
    model.eval()
    underlying_model = get_model(model)
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(data_loader, desc=desc, disable=not is_main_process(args)):
        # 计算损失
        loss = underlying_model.compute_loss(batch)
        total_loss += loss.item()
        num_batches += 1
        
        # 生成预测
        predictions = underlying_model.generate(batch, max_new_tokens=max_new_tokens)
        
        # 收集结果
        for sample, pred in zip(batch, predictions):
            all_predictions.append(pred)
            all_labels.append(sample["class_token"])
    
    # 计算指标
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = calculate_accuracy(all_predictions, all_labels)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def main():
    args = parse_args()
    
    # 初始化 DDP
    setup_ddp(args)
    
    if is_main_process(args):
        print("=" * 60)
        print("双分支 LLM 时序分类训练")
        print("=" * 60)
        print(f"时间: {datetime.datetime.now()}")
        print(f"数据集: {args.dataset}")
        print(f"分支模式: {args.branch_mode}")
        print(f"LLM: {args.llm_id}")
        print(f"LoRA: {args.use_lora}")
        print(f"DDP: {args.use_ddp}, FP16: {args.fp16}")
        print(f"梯度累积: {args.gradient_accumulation_steps}")
        print("=" * 60)
    
    set_seed(args.seed, args.rank)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    if is_main_process(args):
        print(f"\n使用设备: {device}")
    
    # 获取数据集信息
    if is_main_process(args):
        print("\n📂 分析数据集...")
    num_classes, max_length = get_dataset_info(args.dataset, args.data_path)
    
    # 设置 context_length
    if args.context_length is None:
        context_length = ((max_length - 1) // args.patch_length + 1) * args.patch_length
    else:
        context_length = args.context_length
    
    if is_main_process(args):
        print(f"   类别数: {num_classes}")
        print(f"   最大长度: {max_length}")
        print(f"   Context length: {context_length}")
    
    # 创建保存目录
    vit_short_name = args.vit_model_name.split("/")[-1].replace("-", "_")
    save_subdir = f"{args.branch_mode}_{vit_short_name}"
    if args.use_lora:
        save_subdir += "_lora"
    
    save_dir = os.path.join(args.save_dir, args.dataset, save_subdir)
    
    if is_main_process(args):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    # 创建模型
    if is_main_process(args):
        print("\n🔧 创建模型...")
    
    model = DualBranchLLMModel(
        llm_id=args.llm_id,
        branch_mode=args.branch_mode,
        context_length=context_length,
        patch_length=args.patch_length,
        stride=args.stride,
        d_model=args.d_model,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        vit_model_name=args.vit_model_name,
        vit_layer_idx=args.vit_layer_idx,
        vit_patch_size=args.vit_patch_size,
        vit_stride=args.vit_stride,
        projector_type=args.projector_type,
        projector_dropout=args.projector_dropout,
        freeze_ts_backbone=args.freeze_ts_backbone or args.freeze_encoder,
        freeze_vision_backbone=args.freeze_vision_backbone or args.freeze_encoder,
        device=device,
    )
    
    # 梯度检查点
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    
    # 启用 LoRA
    if args.use_lora:
        if is_main_process(args):
            print("📎 启用LoRA...")
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    
    # DDP 包装
    if args.use_ddp:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
        if is_main_process(args):
            print(f"✅ 模型已用DDP包装")
    
    # 创建数据加载器
    if is_main_process(args):
        print("\n📂 加载数据...")
    eos_token = get_model(model).get_eos_token()
    train_loader, val_loader, test_loader, train_sampler = create_data_loaders(args, eos_token)
    
    if is_main_process(args):
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
    
    # 创建优化器
    if is_main_process(args):
        print("\n⚙️ 创建优化器...")
    underlying_model = get_model(model)
    
    param_groups = []
    
    # 编码器参数
    if not args.freeze_encoder:
        if underlying_model.ts_backbone is not None:
            param_groups.append({
                "params": underlying_model.ts_backbone.parameters(),
                "lr": args.lr_encoder
            })
        if underlying_model.vision_encoder is not None:
            param_groups.append({
                "params": underlying_model.vision_encoder.parameters(),
                "lr": args.lr_encoder
            })
    
    # 投影器参数
    if underlying_model.ts_projector is not None:
        param_groups.append({
            "params": underlying_model.ts_projector.parameters(),
            "lr": args.lr_projector
        })
    if underlying_model.vision_projector is not None:
        param_groups.append({
            "params": underlying_model.vision_projector.parameters(),
            "lr": args.lr_projector
        })
    
    # LoRA 参数
    if args.use_lora:
        lora_params = underlying_model.get_lora_parameters()
        if lora_params:
            param_groups.append({"params": lora_params, "lr": args.lr_lora})
    
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    
    # 学习率调度器
    import math
    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.gradient_accumulation_steps))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_ratio * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    if is_main_process(args):
        print(f"   Total steps: {total_steps}")
        print(f"   Warmup steps: {warmup_steps}")
    
    # FP16 混合精度
    scaler = GradScaler() if args.fp16 else None
    if args.fp16 and is_main_process(args):
        print("⚡ FP16 混合精度已启用")
    
    # 训练循环
    if is_main_process(args):
        print("\n🚀 开始训练...")
    
    best_val_acc = 0.0
    patience_counter = 0
    loss_history = []
    epoch = 0
    
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                args.grad_clip, epoch, args.epochs, args, scaler
            )
            
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                if is_main_process(args):
                    print(f"\n📊 Epoch {epoch} 评估...")
                
                val_results = evaluate(
                    model, val_loader, args.max_new_tokens, args, "Validating"
                )
                val_loss = val_results["loss"]
                val_acc = val_results["accuracy"]
                
                if is_main_process(args):
                    print(f"   Train Loss: {train_loss:.4f}")
                    print(f"   Val Loss: {val_loss:.4f}")
                    print(f"   Val Accuracy: {val_acc:.4f}")
                    
                    # 显示预测样本
                    print("   Sample predictions:")
                    for i in range(min(3, len(val_results["predictions"]))):
                        pred = val_results["predictions"][i]
                        label = val_results["labels"][i]
                        pred_short = pred[-50:] if len(pred) > 50 else pred
                        print(f"     Pred: '{pred_short}' | Label: '{label}'")
                    
                    # 保存最佳模型
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                        
                        checkpoint = {
                            "ts_backbone_state": underlying_model.ts_backbone.state_dict() if underlying_model.ts_backbone else None,
                            "vision_encoder_state": underlying_model.vision_encoder.state_dict() if underlying_model.vision_encoder else None,
                            "ts_projector_state": underlying_model.ts_projector.state_dict() if underlying_model.ts_projector else None,
                            "vision_projector_state": underlying_model.vision_projector.state_dict() if underlying_model.vision_projector else None,
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "args": vars(args),
                        }
                        underlying_model.save_lora_state_to_checkpoint(checkpoint)
                        
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
                if is_main_process(args):
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            if patience_counter >= args.early_stop:
                if is_main_process(args):
                    print(f"\n⏹️ 早停! 验证准确率 {args.early_stop} 轮未改进")
                break
        
        # 最终测试
        if is_main_process(args):
            print("\n" + "=" * 60)
            print("📋 最终测试评估...")
            
            # 加载最佳模型
            best_ckpt = torch.load(
                os.path.join(save_dir, "best_model.pt"),
                map_location=device,
                weights_only=False
            )
            
            if best_ckpt.get("ts_backbone_state") and underlying_model.ts_backbone:
                underlying_model.ts_backbone.load_state_dict(best_ckpt["ts_backbone_state"])
            if best_ckpt.get("ts_projector_state") and underlying_model.ts_projector:
                underlying_model.ts_projector.load_state_dict(best_ckpt["ts_projector_state"])
            if best_ckpt.get("vision_projector_state") and underlying_model.vision_projector:
                underlying_model.vision_projector.load_state_dict(best_ckpt["vision_projector_state"])
            underlying_model.load_lora_state_from_checkpoint(best_ckpt, allow_missing=True)
            
            test_results = evaluate(
                model, test_loader, args.max_new_tokens, args, "Testing"
            )
            
            print(f"\n✅ 测试结果:")
            print(f"   Test Loss: {test_results['loss']:.4f}")
            print(f"   Test Accuracy: {test_results['accuracy']:.4f}")
            
            final_results = {
                "dataset": args.dataset,
                "num_classes": num_classes,
                "context_length": context_length,
                "branch_mode": args.branch_mode,
                "vit_model_name": args.vit_model_name,
                "use_lora": args.use_lora,
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
    
    except KeyboardInterrupt:
        if is_main_process(args):
            print("\n⚠️ 训练被中断")
    except Exception as e:
        if is_main_process(args):
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
        cleanup_ddp(args)
        return 1
    
    cleanup_ddp(args)
    return 0


if __name__ == "__main__":
    exit(main())
