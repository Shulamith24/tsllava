#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
M2: UEA多变量数据集分类训练（基于Stage2预训练模型）

加载curriculum learning的stage2预训练模型进行分类微调。
编码器和投影层解冻，LLM使用LoRA训练。

使用方法：
    # 单GPU训练
    python scripts/train_uea_classification_pretrained.py \
        --pretrained_model OpenTSLM/llama-3.2-1b-m4-sp \
        --dataset Epilepsy \
        --epochs 30 \
        --batch_size 4
    
    # DDP多GPU训练
    torchrun --nproc_per_node=2 scripts/train_uea_classification_pretrained.py \
        --pretrained_model OpenTSLM/llama-3.2-1b-m4-sp \
        --dataset Epilepsy \
        --gradient_accumulation_steps 4 \
        --gradient_checkpointing

训练配置：
- LoRA: r=16, alpha=32 (默认启用)
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

from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.time_series_datasets.uea.UEAClassificationDataset import UEAClassificationDataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE, ENCODER_OUTPUT_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="M2: UEA多变量数据集分类训练（基于Stage2预训练模型）")
    
    # 数据相关
    parser.add_argument("--dataset", type=str, default="Epilepsy", help="UEA数据集名称")
    
    # 模型相关 - 使用HuggingFace预训练模型
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="预训练模型ID (HuggingFace repo_id，如 OpenTSLM/llama-3.2-1b-m4-sp)")
    
    # 模型相关 - 使用本地checkpoint（如train_curriculum_pretrain.py产生的）
    parser.add_argument("--local_checkpoint", type=str, default=None,
                        help="本地checkpoint路径 (如 results/curriculum_pretrain/.../best_model.pt)")
    parser.add_argument("--encoder_type", type=str, default="transformer_cnn",
                        choices=["transformer_cnn", "tslanet"],
                        help="编码器类型（使用local_checkpoint时必须指定）")
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B",
                        help="LLM模型ID（使用local_checkpoint时需要）")
    parser.add_argument("--tslanet_patch_size", type=int, default=8,
                        help="TSLANet的patch_size（使用tslanet编码器时）")
    
    # LoRA相关 (默认启用)
    parser.add_argument("--no_lora", action="store_true", help="禁用LoRA（不推荐）")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    # 训练相关
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr_encoder", type=float, default=2e-4, help="编码器学习率")
    parser.add_argument("--lr_projector", type=float, default=1e-4, help="投影层学习率")
    parser.add_argument("--lr_lora", type=float, default=1e-4, help="LoRA学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热比例")
    
    # 保存相关
    parser.add_argument("--save_dir", type=str, default="results/m2_uea_pretrained", help="结果保存目录")
    
    # DDP和梯度相关
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点")
    parser.add_argument("--freeze_encoder", action="store_true", help="冻结编码器参数")
    parser.add_argument(
        "--runtime_branch_mode",
        type=str,
        default="both",
        choices=["both", "ts_only", "vision_only"],
        help="Dual-branch checkpoint runtime masking mode.",
    )
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--eval_every", type=int, default=5, help="每N轮评估一次")
    parser.add_argument("--early_stop", type=int, default=10, help="早停耐心值")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="生成最大token数")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="评估批次大小")
    
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


def create_data_loaders(args, eos_token: str, world_size: int = 1, rank: int = 0):
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
    
    # 分布式采样器（仅训练集）
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    
    # 评估用DataLoader（支持批量评估）
    eval_batch_size = getattr(args, 'eval_batch_size', 8)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
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
    gradient_accumulation_steps: int = 1,
    rank: int = 0,
) -> float:
    """训练一个epoch（支持梯度累积和DDP）"""
    # 1. 训练模式+初始化
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", disable=(rank != 0))
    for step, batch in enumerate(pbar):
        # 2. 计算损失（缩放用于梯度累积）
        loss = model(batch)
        loss = loss / gradient_accumulation_steps
        
        # 3. 反向传播,ddp会在gpu之间自动同步梯度
        loss.backward()
        
        # 4. 梯度累积完成后更新
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
    max_new_tokens: int,
    desc: str = "Evaluating",
    rank: int = 0,
) -> Dict[str, Any]:
    
    #1. 设置模型评估和初始化
    model.eval()
    underlying_model = get_model(model)
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    #2. 遍历数据
    for batch in tqdm(data_loader, desc=desc, disable=(rank != 0)):
        # 评估时不需要梯度同步，所以可以直接调用底层模型
        loss = underlying_model.compute_loss(batch)
        total_loss += loss.item()
        num_batches += 1
        
        predictions = underlying_model.generate(batch, max_new_tokens=max_new_tokens)
        
        for sample, pred in zip(batch, predictions):
            all_predictions.append(pred)
            all_labels.append(sample["answer"].replace(underlying_model.get_eos_token(), "").strip())
    
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
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "args": vars(args),
    }
    
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
        print("M2: UEA多变量数据集分类训练（基于Stage2预训练模型）")
        print("=" * 60)
        print(f"时间: {datetime.datetime.now()}")
        print(f"数据集: {args.dataset}")
        print(f"预训练模型: {args.pretrained_model}")
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
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    # 同步所有进程
    if world_size > 1:
        dist.barrier()
    
    # 加载模型
    if rank == 0:
        print("\n🔧 加载模型...")
    
    use_lora = not args.no_lora
    
    if args.local_checkpoint:
        # 使用本地checkpoint加载（如train_curriculum_pretrain.py产生的）
        if rank == 0:
            print(f"📂 从本地checkpoint加载: {args.local_checkpoint}")
            print(f"   编码器类型: {args.encoder_type}")
            print(f"   LLM: {args.llm_id}")
        
        # 创建模型
        tslanet_config = {
            "patch_size": args.tslanet_patch_size,
            "output_dim": ENCODER_OUTPUT_DIM,
        }
        model = OpenTSLMSP(
            llm_id=args.llm_id,
            device=device,
            encoder_type=args.encoder_type,
            tslanet_config=tslanet_config if args.encoder_type == "tslanet" else None,
        )
        
        # 加载checkpoint权重
        checkpoint = torch.load(args.local_checkpoint, map_location=device, weights_only=False)
        model.encoder.load_state_dict(checkpoint["encoder_state"])
        model.projector.load_state_dict(checkpoint["projector_state"])
        if rank == 0:
            print(f"✅ 已加载encoder和projector权重")
        
        # 启用LoRA
        if use_lora:
            model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
            # 尝试加载checkpoint中的LoRA权重（如果有）
            model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)
    
    elif args.pretrained_model:
        # 使用HuggingFace预训练模型
        if rank == 0:
            print(f"📂 从HuggingFace加载: {args.pretrained_model}")
        
        model = OpenTSLM.load_pretrained(
            repo_id=args.pretrained_model,
            device=device,
            enable_lora=use_lora,
        )
        
        # 如果需要自定义LoRA参数
        if use_lora and (args.lora_r != 16 or args.lora_alpha != 32):
            model.disable_lora()
            model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
            if rank == 0:
                print(f"📎 重新配置LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    
    else:
        raise ValueError("必须指定 --pretrained_model 或 --local_checkpoint 之一")
    
    # 启用梯度检查点
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    
    # 冻结编码器
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        if rank == 0:
            print("🧊 编码器参数已冻结")

    if hasattr(model, "set_runtime_branch_mode"):
        model.set_runtime_branch_mode(args.runtime_branch_mode)
        if rank == 0:
            print(f"🌿 runtime_branch_mode: {args.runtime_branch_mode}")

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
    
    # 根据是否冻结编码器决定参数组
    param_groups = []
    if not args.freeze_encoder:
        param_groups.append({"params": underlying_model.encoder.parameters(), "lr": args.lr_encoder})
    param_groups.append({"params": underlying_model.projector.parameters(), "lr": args.lr_projector})
    
    if use_lora:
        lora_params = underlying_model.get_lora_parameters()
        if lora_params:
            param_groups.append({"params": lora_params, "lr": args.lr_lora})
    
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    
    # 创建学习率调度器（考虑梯度累积）
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
                
                val_results = evaluate(model, val_loader, args.max_new_tokens, "Validating", rank)
                val_loss = val_results["loss"]
                val_acc = val_results["accuracy"]
                
                if rank == 0:
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
                        args, rank
                    )
                else:
                    patience_counter += 1
                    if rank == 0:
                        print(f"   (无改进, patience: {patience_counter}/{args.early_stop})")
                
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
            
            if patience_counter >= args.early_stop:
                if rank == 0:
                    print(f"\n⏹️ 早停! 验证准确率 {args.early_stop} 轮未改进")
                break
        
        # 最终测试（仅rank=0）
        if rank == 0:
            print("\n" + "=" * 60)
            print("📋 最终测试评估...")
            
            best_ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device, weights_only=False)
            underlying_model.encoder.load_state_dict(best_ckpt["encoder_state"])
            underlying_model.projector.load_state_dict(best_ckpt["projector_state"])
            underlying_model.load_lora_state_from_checkpoint(best_ckpt, allow_missing=True)
            
            test_results = evaluate(model, test_loader, args.max_new_tokens, "Testing", rank)
            
            print(f"\n✅ 测试结果:")
            print(f"   Test Loss: {test_results['loss']:.4f}")
            print(f"   Test Accuracy: {test_results['accuracy']:.4f}")
            
            final_results = {
                "dataset": args.dataset,
                "pretrained_model": args.pretrained_model,
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
    
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
