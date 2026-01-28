#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
实验 A: UCR分类训练（使用分类头而非生成式head）

目标：验证LLM作为序列聚合器的作用，隔离生成式目标的影响。

架构：
    [PrePrompt] + [TS Tokens] + [PostPrompt] + [ANS]
                                                 ↓
                                       classification_head
                                                 ↓
                                            num_classes

使用方法：
    # 从零开始训练（不使用预训练权重）
    python scripts/train_ucr_classification_head.py \
        --dataset ECG5000 \
        --encoder_type transformer_cnn \
        --epochs 30 \
        --batch_size 32

训练配置：
- LoRA: r=16, alpha=32 (默认启用)
- Encoder LR: 2e-4
- Projector LR: 1e-4
- LoRA LR: 1e-4
- 使用 CrossEntropyLoss 直接分类
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model.llm.OpenTSLMClassifier import OpenTSLMClassifier
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import UCRClassificationDataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE, ENCODER_OUTPUT_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="实验A: UCR分类训练（分类头模式）")

    # 必须指定
    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点")
    parser.add_argument("--freeze_encoder", action="store_true", help="冻结编码器参数")

    # 数据相关
    parser.add_argument("--dataset", type=str, default="CricketZ", help="UCR数据集名称")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR数据根目录")
    
    # 模型相关
    parser.add_argument("--encoder_type", type=str, default="transformer_cnn",
                        choices=["transformer_cnn", "tslanet"],
                        help="编码器类型")
    parser.add_argument("--encoder_pretrained", type=str, default=None,
                        help="编码器预训练权重路径（可选）")
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B",
                        help="LLM模型ID")
    parser.add_argument("--tslanet_patch_size", type=int, default=8,
                        help="TSLANet的patch_size（使用tslanet编码器时）")
    
    # LoRA相关 (默认启用)
    parser.add_argument("--no_lora", action="store_true", help="禁用LoRA（不推荐）")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    # 训练相关
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr_encoder", type=float, default=2e-4, help="编码器学习率")
    parser.add_argument("--lr_projector", type=float, default=1e-4, help="投影层学习率")
    parser.add_argument("--lr_lora", type=float, default=1e-4, help="LoRA学习率")
    parser.add_argument("--lr_cls_head", type=float, default=1e-3, help="分类头学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热比例")
    
    # 保存相关
    parser.add_argument("--save_dir", type=str, default="results/exp_a_classification_head", help="结果保存目录")
    
    # DDP和梯度相关
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--eval_every", type=int, default=5, help="每N轮评估一次")
    parser.add_argument("--early_stop", type=int, default=10, help="早停耐心值")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="评估批次大小")
    
    return parser.parse_args()


def setup_distributed():
    """初始化分布式训练环境（用于torchrun）"""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        
        return local_rank, world_size, rank
    return 0, 1, 0


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_model(model):
    """获取底层模型（兼容DDP包装）"""
    return model.module if hasattr(model, "module") else model


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    计算分类准确率
    
    Args:
        predictions: Predicted class indices [B]
        labels: True class labels [B]
    
    Returns:
        Accuracy as float
    """
    correct = (predictions == labels).sum().item()
    return correct / len(predictions) if len(predictions) > 0 else 0.0


def create_data_loaders(args, eos_token: str, world_size: int = 1, rank: int = 0):
    """创建数据加载器"""
    # 创建数据集
    train_dataset = UCRClassificationDataset(
        split="train",
        EOS_TOKEN=eos_token,
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )
    
    val_dataset = UCRClassificationDataset(
        split="validation",
        EOS_TOKEN=eos_token,
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )
    
    test_dataset = UCRClassificationDataset(
        split="test",
        EOS_TOKEN=eos_token,
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )
    
    # Collate函数
    def collate_fn(batch):
        return extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        )
    
    # 分布式采样器
    train_sampler = None
    val_sampler = None
    test_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    
    eval_batch_size = getattr(args, 'eval_batch_size', 8)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=test_sampler,
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
    gradient_accumulation_steps: int = 1,
    rank: int = 0,
) -> float:
    """训练一个epoch（支持梯度累积和DDP）"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", disable=(rank != 0))
    for step, batch in enumerate(pbar):
        # 计算损失（使用 forward 方法，batch 中包含 label）
        loss = model(batch)
        loss = loss / gradient_accumulation_steps
        
        # 反向传播
        loss.backward()
        
        # 梯度累积完成后更新
        if (step + 1) % gradient_accumulation_steps == 0:
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        if rank == 0:
            pbar.set_postfix({
                "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
    
    # 处理最后不足accumulation_steps的batch
    if num_batches % gradient_accumulation_steps != 0:
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model,
    data_loader: DataLoader,
    desc: str = "Evaluating",
    rank: int = 0,
) -> Dict[str, Any]:
    """评估模型"""
    model.eval()
    underlying_model = get_model(model)
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(data_loader, desc=desc, disable=(rank != 0)):
        # 提取标签索引 (0 到 K-1)
        labels = torch.tensor([item["label_index"] for item in batch], device=underlying_model.device, dtype=torch.long)
        
        # 计算损失
        loss = underlying_model.compute_loss(batch, labels)
        total_loss += loss.item()
        num_batches += 1
        
        # 生成预测
        predictions = underlying_model.predict(batch)  # [B]
        
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())
    
    # 合并所有batch的结果
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 计算指标
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = calculate_accuracy(all_predictions, all_labels)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    val_acc: float,
    save_path: str,
    args,
    rank: int = 0,
):
    """保存checkpoint（仅rank=0执行）"""
    if rank != 0:
        return
    
    underlying_model = get_model(model)
    checkpoint = {
        "encoder_state": underlying_model.encoder.state_dict(),
        "projector_state": underlying_model.projector.state_dict(),
        "ans_token": underlying_model.ans_token.data,
        "classification_head_state": underlying_model.classification_head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "args": vars(args),
    }
    
    # 保存LoRA权重
    underlying_model.save_lora_state_to_checkpoint(checkpoint)
    
    torch.save(checkpoint, save_path)
    print(f"💾 Saved checkpoint to: {save_path}")


def main():
    args = parse_args()
    
    # 初始化分布式环境
    local_rank, world_size, rank = setup_distributed()
    
    # 仅rank=0打印信息
    if rank == 0:
        print("=" * 60)
        print("实验 A: UCR分类训练（分类头模式）")
        print("=" * 60)
        print(f"时间: {datetime.datetime.now()}")
        print(f"数据集: {args.dataset}")
        print(f"编码器: {args.encoder_type}")
        print(f"LoRA: {not args.no_lora}")
        print(f"DDP: world_size={world_size}")
        print(f"梯度累积: {args.gradient_accumulation_steps}")
        print(f"梯度检查点: {args.gradient_checkpointing}")
        print("=" * 60)
    
    # 设置随机种子
    set_seed(args.seed + rank)
    
    # 设置设备
    if world_size > 1:
        device = f"cuda:{local_rank}"
    elif args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        if rank == 0:
            print("⚠️ CUDA不可用，使用CPU")
        device = "cpu"
    
    # 创建保存目录（仅rank=0）
    save_dir = os.path.join(args.save_dir, args.dataset)
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        # 保存配置
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    # 同步所有进程
    if world_size > 1:
        dist.barrier()
    
    # 获取类别数量
    num_classes = UCRClassificationDataset.get_num_classes()
    
    # 创建模型
    if rank == 0:
        print("\n🔧 创建模型...")
        print(f"   类别数量: {num_classes}")
    
    tslanet_config = {
        "patch_size": args.tslanet_patch_size,
        "output_dim": ENCODER_OUTPUT_DIM,
    }
    
    model = OpenTSLMClassifier(
        llm_id=args.llm_id,
        num_classes=num_classes,
        device=device,
        encoder_type=args.encoder_type,
        encoder_pretrained_path=args.encoder_pretrained,
        tslanet_config=tslanet_config if args.encoder_type == "tslanet" else None,
    )
    
    # 启用梯度检查点
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    
    # 冻结编码器（可选）
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        if rank == 0:
            print("🧊 编码器参数已冻结")
    
    # 启用LoRA
    use_lora = not args.no_lora
    if use_lora:
        if rank == 0:
            print("📎 启用LoRA...")
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    
    # DDP包装
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
        if rank == 0:
            print(f"✅ 模型已用DDP包装 (world_size={world_size})")
    
    # 创建数据加载器
    if rank == 0:
        print("\n📂 加载数据...")
    eos_token = get_model(model).get_eos_token()
    train_loader, val_loader, test_loader, train_sampler = create_data_loaders(
        args, eos_token, world_size, rank
    )
    
    if rank == 0:
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
    
    # 创建优化器
    if rank == 0:
        print("\n⚙️ 创建优化器...")
    underlying_model = get_model(model)
    
    # 参数组
    param_groups = []
    if not args.freeze_encoder:
        param_groups.append({"params": underlying_model.encoder.parameters(), "lr": args.lr_encoder})
    param_groups.append({"params": underlying_model.projector.parameters(), "lr": args.lr_projector})
    param_groups.append({"params": [underlying_model.ans_token], "lr": args.lr_cls_head})
    param_groups.append({"params": underlying_model.classification_head.parameters(), "lr": args.lr_cls_head})
    
    if use_lora:
        lora_params = underlying_model.get_lora_parameters()
        if lora_params:
            param_groups.append({"params": lora_params, "lr": args.lr_lora})
    
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    
    # 创建学习率调度器
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    if rank == 0:
        print(f"   Effective batch size: {effective_batch_size}")
        print(f"   Total steps: {total_steps}")
        print(f"   Warmup steps: {warmup_steps}")
    
    # 训练循环
    if rank == 0:
        print("\n🚀 开始训练...")
    best_val_acc = 0.0
    patience_counter = 0
    loss_history = []
    epoch = 0
    
    try:
        for epoch in range(1, args.epochs + 1):
            # 设置sampler的epoch（DDP必需）
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            # 训练
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                args.grad_clip, epoch, args.epochs,
                args.gradient_accumulation_steps, rank
            )
            
            # 定期评估
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                if rank == 0:
                    print(f"\n📊 Epoch {epoch} 评估...")
                
                # 验证集评估
                val_results = evaluate(model, val_loader, "Validating", rank)
                val_loss = val_results["loss"]
                val_acc = val_results["accuracy"]
                
                if rank == 0:
                    print(f"   Train Loss: {train_loss:.4f}")
                    print(f"   Val Loss: {val_loss:.4f}")
                    print(f"   Val Accuracy: {val_acc:.4f}")
                
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    save_checkpoint(
                        model, optimizer, scheduler, epoch,
                        val_loss, val_acc,
                        os.path.join(save_dir, "best_model.pt"),
                        args, rank
                    )
                else:
                    patience_counter += 1
                    if rank == 0:
                        print(f"   (无改进, patience: {patience_counter}/{args.early_stop})")
                
                # 记录历史
                if rank == 0:
                    loss_history.append({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    })
                    with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
                        json.dump(loss_history, f, indent=2)
            else:
                if rank == 0:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # 早停
            if patience_counter >= args.early_stop:
                if rank == 0:
                    print(f"\n⏹️ 早停! 验证准确率 {args.early_stop} 轮未改进")
                break
        
        # 最终测试（仅rank=0）
        if rank == 0:
            print("\n" + "=" * 60)
            print("📋 最终测试评估...")
            
            # 加载最佳模型
            best_ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device, weights_only=False)
            underlying_model.encoder.load_state_dict(best_ckpt["encoder_state"])
            underlying_model.projector.load_state_dict(best_ckpt["projector_state"])
            underlying_model.ans_token.data = best_ckpt["ans_token"]
            underlying_model.classification_head.load_state_dict(best_ckpt["classification_head_state"])
            underlying_model.load_lora_state_from_checkpoint(best_ckpt, allow_missing=True)
            
            test_results = evaluate(model, test_loader, "Testing", rank)
            
            print(f"\n✅ 测试结果:")
            print(f"   Test Loss: {test_results['loss']:.4f}")
            print(f"   Test Accuracy: {test_results['accuracy']:.4f}")
            
            # 保存测试结果
            final_results = {
                "dataset": args.dataset,
                "best_val_acc": best_val_acc,
                "test_loss": test_results["loss"],
                "test_accuracy": test_results["accuracy"],
                "epochs_trained": epoch,
            }
            
            with open(os.path.join(save_dir, "final_results.json"), "w") as f:
                json.dump(final_results, f, indent=2)
            
            print("=" * 60)
            print(f"结果保存到: {save_dir}")
            print("=" * 60)
    
    finally:
        # 清理分布式环境
        cleanup_distributed()


if __name__ == "__main__":
    main()
