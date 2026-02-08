#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
PatchTST + VisionEncoder 双分支时序分类训练脚本

特性：
- 双分支融合：PatchTST 时序编码 + TiViT 风格图像编码
- DDP 分布式训练支持
- 显存优化：FP16 混合精度、梯度累积、梯度检查点

使用方法：
    # 单卡训练
    python -m src.patchtst_ucr.train_dual_branch_tivit --dataset Adiac --epochs 50

    # 启用显存优化
    python -m src.patchtst_ucr.train_dual_branch_tivit --dataset Adiac --fp16 --gradient_accumulation_steps 4

    # 多卡 DDP 训练
    torchrun --nproc_per_node=2 -m src.patchtst_ucr.train_dual_branch_tivit --dataset Adiac --use_ddp --fp16
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
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

# 添加 src 目录到路径
script_dir = Path(__file__).parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from patchtst_ucr.dual_branch_model import PatchTSTWithVisionBranch
from patchtst_ucr.ucr_dataset import UCRDatasetForPatchTST, get_dataset_info


def parse_args():
    parser = argparse.ArgumentParser(description="PatchTST + VisionEncoder 双分支分类")

    # 数据相关
    parser.add_argument("--dataset", type=str, default="Adiac", help="UCR数据集名称")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR数据根目录")
    
    # PatchTST 时序分支配置
    parser.add_argument("--context_length", type=int, default=None, 
                       help="上下文长度（None则自动设置为数据集最大长度）")
    parser.add_argument("--patch_length", type=int, default=16, help="Patch 长度")
    parser.add_argument("--stride", type=int, default=8, help="Patch 步长")
    parser.add_argument("--d_model", type=int, default=128, help="PatchTST 模型维度")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="PatchTST Attention heads")
    parser.add_argument("--num_hidden_layers", type=int, default=3, help="PatchTST Transformer 层数")
    parser.add_argument("--ffn_dim", type=int, default=512, help="PatchTST FFN 维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    
    # Vision 分支配置（支持多种 ViT 模型）
    parser.add_argument("--vit_model_name", type=str, default="facebook/dinov2-base",
                       help="ViT模型名称，支持 dinov2/clip/siglip/mae 等")
    parser.add_argument("--vit_layer_idx", type=int, default=-1, help="ViT 特征提取层索引")
    parser.add_argument("--vit_patch_size", type=int, default=16, help="时序图像化 patch 大小")
    parser.add_argument("--vit_stride", type=float, default=0.5, help="时序图像化步长比例")
    
    # 分支控制
    parser.add_argument("--branch_mode", type=str, default="both",
                       choices=["both", "ts_only", "vision_only"],
                       help="分支模式: both(双分支), ts_only(仅时序), vision_only(仅视觉)")
    
    # 聚合头配置
    parser.add_argument("--aggregator_layers", type=int, default=1, help="聚合头 Transformer 层数")
    parser.add_argument("--aggregator_hidden_size", type=int, default=None, 
                       help="聚合头 hidden size（None则与d_model相同）")
    parser.add_argument("--aggregator_num_heads", type=int, default=8, help="聚合头 attention heads")
    parser.add_argument("--aggregator_ffn_dim", type=int, default=None, 
                       help="聚合头 FFN 维度（None则自动计算）")
    
    # 投影层配置
    parser.add_argument("--projector_type", type=str, default="mlp", 
                       choices=["mlp", "linear", "none"],
                       help="投影层类型")
    parser.add_argument("--projector_dropout", type=float, default=0.1, 
                       help="MLP投影层的Dropout概率")
    
    # 冻结选项
    parser.add_argument("--freeze_ts_backbone", action="store_true", help="冻结 PatchTST backbone")
    parser.add_argument("--freeze_vision_backbone", action="store_true", default=True,
                       help="冻结 Vision backbone（默认开启）")
    parser.add_argument("--no_freeze_vision_backbone", action="store_true",
                       help="不冻结 Vision backbone")
    
    # 训练相关
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    
    # DDP 分布式训练
    parser.add_argument("--use_ddp", action="store_true", help="启用 DDP 分布式训练")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP local rank")
    
    # 显存优化
    parser.add_argument("--fp16", action="store_true", help="启用 FP16 混合精度")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="梯度累积步数")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="启用梯度检查点（节省显存但降低速度）")
    
    # 保存相关
    parser.add_argument("--save_dir", type=str, default="results/patchtst_dual_branch_tivit", 
                       help="结果保存目录")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--eval_every", type=int, default=5, help="每N轮评估一次")
    parser.add_argument("--early_stop", type=int, default=15, help="早停耐心值")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="评估批次大小")
    
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


def prepare_batch(
    batch: List[Dict],
    context_length: int,
    device: str,
):
    """将 UCR 批次转换为模型输入"""
    past_values_list = []
    labels = []
    
    for sample in batch:
        ts = sample["time_series"][0]
        
        if not isinstance(ts, torch.Tensor):
            ts = torch.tensor(ts, dtype=torch.float32)
        
        if len(ts) < context_length:
            padded = torch.zeros(context_length, device=device)
            padded[:len(ts)] = ts.to(device)
        else:
            padded = ts[:context_length].to(device)
        
        past_values_list.append(padded.unsqueeze(-1))
        labels.append(sample["int_label"])
    
    past_values = torch.stack(past_values_list, dim=0)
    labels = torch.tensor(labels, device=device, dtype=torch.long)
    
    return past_values, labels


def create_data_loaders(args, num_classes: int, context_length: int):
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
    
    # DDP 采样器
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.use_ddp else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,  # 避免多进程问题
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
    context_length: int,
    grad_clip: float,
    device: str,
    epoch: int,
    num_epochs: int,
    args,
    scaler: GradScaler = None,
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # DDP 设置 epoch（用于 shuffle）
    if args.use_ddp and hasattr(train_loader, 'sampler'):
        train_loader.sampler.set_epoch(epoch)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", disable=not is_main_process(args))
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        past_values, labels = prepare_batch(batch, context_length, device)
        
        # 混合精度训练
        if args.fp16:
            with autocast():
                outputs = model(past_values=past_values, labels=labels)
                loss = outputs["loss"]
                loss = loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
        else:
            outputs = model(past_values=past_values, labels=labels)
            loss = outputs["loss"]
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
        
        # 梯度累积
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
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
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model,
    data_loader: DataLoader,
    context_length: int,
    device: str,
    args,
    desc: str = "Evaluating",
) -> Dict[str, Any]:
    """评估模型"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(data_loader, desc=desc, disable=not is_main_process(args)):
        past_values, labels = prepare_batch(batch, context_length, device)
        
        if args.fp16:
            with autocast():
                outputs = model(past_values=past_values, labels=labels)
        else:
            outputs = model(past_values=past_values, labels=labels)
        
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
    
    # 初始化 DDP
    setup_ddp(args)
    
    if is_main_process(args):
        print("=" * 60)
        print("PatchTST + VisionEncoder 双分支时序分类")
        print("=" * 60)
        print(f"时间: {datetime.datetime.now()}")
        print(f"数据集: {args.dataset}")
        print(f"分支模式: {args.branch_mode}")
        print(f"ViT 模型: {args.vit_model_name}")
        print(f"DDP: {args.use_ddp}, FP16: {args.fp16}")
        print(f"梯度累积: {args.gradient_accumulation_steps}")
        print("=" * 60)
    
    set_seed(args.seed, args.rank)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    if is_main_process(args):
        print(f"\n使用设备: {device}")
    
    # 分析数据集
    if is_main_process(args):
        print("\n📂 分析数据集...")
    num_classes, max_length = get_dataset_info(args.dataset, args.data_path)
    
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
    save_subdir = f"{args.branch_mode}_{vit_short_name}_L{args.aggregator_layers}"
    if args.freeze_ts_backbone:
        save_subdir += "_tsFrozen"
    if args.freeze_vision_backbone:
        save_subdir += "_vFrozen"
    
    save_dir = os.path.join(args.save_dir, args.dataset, save_subdir)
    
    if is_main_process(args):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    # 创建模型
    if is_main_process(args):
        print("\n🔧 创建模型...")
    
    model = PatchTSTWithVisionBranch(
        num_classes=num_classes,
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
        aggregator_layers=args.aggregator_layers,
        aggregator_hidden_size=args.aggregator_hidden_size,
        aggregator_num_heads=args.aggregator_num_heads,
        aggregator_ffn_dim=args.aggregator_ffn_dim,
        projector_type=args.projector_type,
        projector_dropout=args.projector_dropout,
        branch_mode=args.branch_mode,
        freeze_ts_backbone=args.freeze_ts_backbone,
        freeze_vision_backbone=args.freeze_vision_backbone,
        device=device,
    ).to(device)
    
    # 梯度检查点
    if args.gradient_checkpointing:
        if hasattr(model.ts_backbone, 'gradient_checkpointing_enable'):
            model.ts_backbone.gradient_checkpointing_enable()
        if is_main_process(args):
            print("🔄 梯度检查点已启用")
    
    # DDP 包装
    if args.use_ddp:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process(args):
        print(f"可训练参数: {trainable_params:,}")
    
    # 创建数据加载器
    if is_main_process(args):
        print("\n📂 加载数据...")
    train_loader, val_loader, test_loader, train_sampler = create_data_loaders(
        args, num_classes, context_length
    )
    
    if is_main_process(args):
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
    
    # 创建优化器
    if is_main_process(args):
        print("\n⚙️  创建优化器...")
    
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 计算总步数（考虑梯度累积）
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_ratio * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
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
    
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                context_length, args.grad_clip, device,
                epoch, args.epochs, args, scaler
            )
            
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                if is_main_process(args):
                    print(f"\n📊 Epoch {epoch} 评估...")
                
                # 获取原始模型（DDP 包装下）
                eval_model = model.module if args.use_ddp else model
                
                val_results = evaluate(
                    eval_model, val_loader, context_length, device, args, "Validating"
                )
                val_loss = val_results["loss"]
                val_acc = val_results["accuracy"]
                
                if is_main_process(args):
                    print(f"   Train Loss: {train_loss:.4f}")
                    print(f"   Val Loss: {val_loss:.4f}")
                    print(f"   Val Accuracy: {val_acc:.4f}")
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                        
                        checkpoint = {
                            "model_state": eval_model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "config": eval_model.get_config(),
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
                if is_main_process(args):
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            if patience_counter >= args.early_stop:
                if is_main_process(args):
                    print(f"\n⏹️  早停! 验证准确率 {args.early_stop} 轮未改进")
                break
        
        # 最终测试
        if is_main_process(args):
            print("\n" + "=" * 60)
            print("📋 最终测试评估...")
            
            eval_model = model.module if args.use_ddp else model
            
            # 加载最佳模型
            best_ckpt = torch.load(
                os.path.join(save_dir, "best_model.pt"),
                map_location=device,
                weights_only=False
            )
            eval_model.load_state_dict(best_ckpt["model_state"])
            
            test_results = evaluate(
                eval_model, test_loader, context_length, device, args, "Testing"
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
                "aggregator_layers": args.aggregator_layers,
                "aggregator_hidden_size": args.aggregator_hidden_size or args.d_model,
                "freeze_ts_backbone": args.freeze_ts_backbone,
                "freeze_vision_backbone": args.freeze_vision_backbone,
                "total_params": sum(p.numel() for p in eval_model.parameters()),
                "trainable_params": eval_model.count_parameters(),
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
            print("\n⚠️  训练被中断")
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
