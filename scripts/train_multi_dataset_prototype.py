#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
多数据集统一Prototype分类训练脚本

特点:
1. 从配置文件加载多个UCR数据集
2. 每数据集独立的 Prompt (PromptBank) + Prototype (PrototypeBank)
3. Episodic采样：一个batch一个dataset，温度采样平衡大小差异
4. 两阶段训练：Stage 0 冻结主干，Stage 1 联合训练
5. 评估指标：Macro average + Worst-10%

使用方法:
    # Stage 0 (对齐 task tokens)
    python scripts/train_multi_dataset_prototype.py \
        --pretrained_model OpenTSLM/llama-3.2-1b-m4-sp \
        --config configs/multi_dataset_ucr.txt \
        --stage 0 \
        --epochs 5

    # Stage 1 (联合训练)
    python scripts/train_multi_dataset_prototype.py \
        --pretrained_model OpenTSLM/llama-3.2-1b-m4-sp \
        --config configs/multi_dataset_ucr.txt \
        --stage 1 \
        --resume_from results/multi_dataset/stage0_best.pt \
        --epochs 30
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
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.model.llm.OpenTSLMMultiDataset import OpenTSLMMultiDataset
from opentslm.time_series_datasets.multi_dataset import (
    MultiDatasetRegistry,
    UnifiedPrototypeDataset,
)
from opentslm.time_series_datasets.episodic_sampler import EpisodicBatchSampler
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE, ENCODER_OUTPUT_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="多数据集统一Prototype分类训练")
    
    # 数据相关
    parser.add_argument("--config", type=str, default="configs/multi_dataset_ucr.txt",
                        help="数据集配置文件路径")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR数据根目录")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="温度采样参数: 0=均匀, 0.3-0.5=折中, 1=按数据量")
    
    # 模型相关
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="预训练模型ID (HuggingFace repo_id)")
    parser.add_argument("--local_checkpoint", type=str, default=None,
                        help="本地checkpoint路径")
    parser.add_argument("--encoder_type", type=str, default="transformer_cnn",
                        choices=["transformer_cnn", "tslanet"])
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B")
    
    # Prototype相关
    parser.add_argument("--prompt_len", type=int, default=10, help="可学习Prompt长度")
    parser.add_argument("--init_temperature", type=float, default=1.0, help="温度初始值")
    
    # LoRA相关
    parser.add_argument("--no_lora", action="store_true", help="禁用LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    # 训练阶段
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1],
                        help="训练阶段: 0=只训练头部, 1=联合训练")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从checkpoint加载模型权重")
    
    # 训练超参
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_prompt_bank", type=float, default=1e-3, help="PromptBank学习率")
    parser.add_argument("--lr_prototype_bank", type=float, default=1e-3, help="PrototypeBank学习率")
    parser.add_argument("--lr_cls", type=float, default=1e-3, help="CLS相关学习率")
    parser.add_argument("--lr_encoder", type=float, default=2e-4, help="Encoder学习率")
    parser.add_argument("--lr_projector", type=float, default=1e-4, help="Projector学习率")
    parser.add_argument("--lr_lora", type=float, default=1e-4, help="LoRA学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    
    # 保存相关
    parser.add_argument("--save_dir", type=str, default="results/multi_dataset")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    return parser.parse_args()


def setup_distributed():
    """初始化分布式环境"""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return local_rank, world_size, rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_model(model):
    return model.module if hasattr(model, "module") else model


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """整理批次数据"""
    processed = extend_time_series_to_match_patch_size_and_aggregate(
        batch, patch_size=PATCH_SIZE
    )
    return processed


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
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", disable=(rank != 0))
    for step, batch in enumerate(pbar):
        loss = model(batch)
        loss = loss / gradient_accumulation_steps
        
        loss.backward()
        
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
def evaluate_multi_dataset(
    model,
    registry: MultiDatasetRegistry,
    split: str = "test",
    batch_size: int = 32,
    rank: int = 0,
) -> Dict[str, Any]:
    """
    评估多数据集
    
    返回:
        - per_dataset: 每个数据集的accuracy
        - macro_avg: 宏平均accuracy
        - worst_10_pct: 最差10%数据集的平均accuracy
    """
    underlying_model = get_model(model)
    underlying_model.eval()
    
    per_dataset_results = {}
    
    for ds_info in registry.get_all_datasets():
        # 创建该数据集的DataLoader
        dataset = UnifiedPrototypeDataset(registry, split=split)
        indices = dataset.get_indices_for_dataset(ds_info.ds_id)
        
        if len(indices) == 0:
            continue
        
        # 创建子集DataLoader
        subset_data = [dataset[i] for i in indices]
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        num_batches = 0
        
        # 分批处理
        for i in range(0, len(subset_data), batch_size):
            batch = subset_data[i:i+batch_size]
            batch = collate_fn(batch)
            
            loss, logits = underlying_model.forward_prototype(batch)
            predictions = logits.argmax(dim=-1)
            labels = torch.tensor([s["label_index"] for s in batch], device=logits.device)
            
            total_loss += loss.item()
            total_correct += (predictions == labels).sum().item()
            total_samples += len(batch)
            num_batches += 1
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        per_dataset_results[ds_info.name] = {
            "accuracy": accuracy,
            "loss": avg_loss,
            "num_samples": total_samples,
            "temperature": underlying_model.prototype_bank.get_temperature(ds_info.ds_id),
        }
    
    # 计算宏平均
    accuracies = [r["accuracy"] for r in per_dataset_results.values()]
    macro_avg = sum(accuracies) / len(accuracies) if accuracies else 0.0
    
    # 计算最差10%
    sorted_acc = sorted(accuracies)
    num_worst = max(1, len(sorted_acc) // 10)
    worst_10_pct = sum(sorted_acc[:num_worst]) / num_worst if sorted_acc else 0.0
    
    return {
        "per_dataset": per_dataset_results,
        "macro_avg": macro_avg,
        "worst_10_pct": worst_10_pct,
    }


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    macro_acc: float,
    save_path: str,
    args,
    rank: int = 0,
):
    """保存checkpoint"""
    if rank != 0:
        return
    
    underlying_model = get_model(model)
    checkpoint = {
        "encoder_state": underlying_model.encoder.state_dict(),
        "projector_state": underlying_model.projector.state_dict(),
        "prompt_bank_state": underlying_model.prompt_bank.state_dict(),
        "prototype_bank_state": underlying_model.prototype_bank.state_dict(),
        "cls_embed": underlying_model.cls_embed.data,
        "cls_projector_state": underlying_model.cls_projector.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "macro_acc": macro_acc,
        "args": vars(args),
    }
    
    underlying_model.save_lora_state_to_checkpoint(checkpoint)
    torch.save(checkpoint, save_path)
    print(f"💾 Saved checkpoint to: {save_path}")


def main():
    args = parse_args()
    
    # 初始化分布式
    local_rank, world_size, rank = setup_distributed()
    
    if rank == 0:
        print("=" * 60)
        print("多数据集统一Prototype分类训练")
        print("=" * 60)
        print(f"时间: {datetime.datetime.now()}")
        print(f"配置: {args.config}")
        print(f"Stage: {args.stage}")
        print(f"Alpha: {args.alpha}")
        print(f"Prompt长度: {args.prompt_len}")
        print(f"LoRA: {not args.no_lora}")
        print("=" * 60)
    
    set_seed(args.seed + rank)
    
    # 设备
    if world_size > 1:
        device = f"cuda:{local_rank}"
    elif args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        if rank == 0:
            print("⚠️ 使用CPU")
    
    # 保存目录
    save_dir = args.save_dir
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"config_stage{args.stage}.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    if world_size > 1:
        dist.barrier()
    
    # 加载数据集注册表
    if rank == 0:
        print("\n📂 加载数据集...")
    registry = MultiDatasetRegistry(data_path=args.data_path)
    registry.load_from_file(args.config)
    
    # 创建模型
    if rank == 0:
        print("\n🔧 创建模型...")
    
    use_lora = not args.no_lora
    
    if args.pretrained_model:
        # 从HuggingFace加载基础权重
        if rank == 0:
            print(f"📂 从HuggingFace加载: {args.pretrained_model}")
        
        base_model = OpenTSLM.load_pretrained(
            repo_id=args.pretrained_model,
            device=device,
            enable_lora=False,
        )
        
        # 创建多数据集模型
        model = OpenTSLMMultiDataset(
            registry=registry,
            llm_id=base_model.llm.config._name_or_path if hasattr(base_model.llm.config, '_name_or_path') else "meta-llama/Llama-3.2-1B",
            device=device,
            encoder_type="transformer_cnn",
            prompt_len=args.prompt_len,
            init_temperature=args.init_temperature,
        )
        
        # 复制权重
        model.encoder.load_state_dict(base_model.encoder.state_dict())
        model.projector.load_state_dict(base_model.projector.state_dict())
        
        if use_lora:
            model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        
        del base_model
        torch.cuda.empty_cache()
    
    elif args.local_checkpoint:
        # 从本地checkpoint创建
        model = OpenTSLMMultiDataset(
            registry=registry,
            llm_id=args.llm_id,
            device=device,
            encoder_type=args.encoder_type,
            prompt_len=args.prompt_len,
            init_temperature=args.init_temperature,
        )
        
        checkpoint = torch.load(args.local_checkpoint, map_location=device, weights_only=False)
        model.encoder.load_state_dict(checkpoint["encoder_state"])
        model.projector.load_state_dict(checkpoint["projector_state"])
        if rank == 0:
            print(f"✅ 加载encoder/projector from {args.local_checkpoint}")
        
        if use_lora:
            model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
            model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)
    
    else:
        raise ValueError("必须指定 --pretrained_model 或 --local_checkpoint")
    
    # 从resume加载（如果有）
    if args.resume_from:
        if rank == 0:
            print(f"📂 从{args.resume_from}恢复...")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.prompt_bank.load_state_dict(ckpt["prompt_bank_state"])
        model.prototype_bank.load_state_dict(ckpt["prototype_bank_state"])
        model.cls_embed.data = ckpt["cls_embed"].to(device)
        if "cls_projector_state" in ckpt:
            model.cls_projector.load_state_dict(ckpt["cls_projector_state"])
        if rank == 0:
            print(f"✅ 恢复 PromptBank/PrototypeBank/cls")
    
    # 配置训练阶段
    if args.stage == 0:
        model.freeze_backbone()
    else:
        model.unfreeze_for_stage1(unfreeze_encoder=True)
    
    # 梯度检查点
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    
    # DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
        if rank == 0:
            print(f"✅ DDP (world_size={world_size})")
    
    # 创建数据集和DataLoader
    if rank == 0:
        print("\n📂 创建数据加载器...")
    
    train_dataset = UnifiedPrototypeDataset(registry, split="train")
    sampler = EpisodicBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        alpha=args.alpha,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
    )
    
    if rank == 0:
        print(f"   Train: {len(train_loader)} episodes/epoch")
    
    # 优化器
    if rank == 0:
        print("\n⚙️ 创建优化器...")
    underlying_model = get_model(model)
    
    param_groups = []
    if args.stage == 0:
        # Stage 0: 只训练banks/cls
        param_groups.append({"params": list(underlying_model.prompt_bank.parameters()), "lr": args.lr_prompt_bank})
        param_groups.append({"params": list(underlying_model.prototype_bank.parameters()), "lr": args.lr_prototype_bank})
        param_groups.append({"params": [underlying_model.cls_embed] + list(underlying_model.cls_projector.parameters()), "lr": args.lr_cls})
    else:
        # Stage 1: 全部训练
        param_groups.append({"params": list(underlying_model.encoder.parameters()), "lr": args.lr_encoder})
        param_groups.append({"params": list(underlying_model.projector.parameters()), "lr": args.lr_projector})
        param_groups.append({"params": list(underlying_model.prompt_bank.parameters()), "lr": args.lr_prompt_bank})
        param_groups.append({"params": list(underlying_model.prototype_bank.parameters()), "lr": args.lr_prototype_bank})
        param_groups.append({"params": [underlying_model.cls_embed] + list(underlying_model.cls_projector.parameters()), "lr": args.lr_cls})
        
        if use_lora:
            lora_params = underlying_model.get_lora_parameters()
            if lora_params:
                param_groups.append({"params": lora_params, "lr": args.lr_lora})
    
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    
    # 学习率调度
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    if rank == 0:
        print(f"   Total steps: {total_steps}")
        print(f"   Warmup steps: {warmup_steps}")
    
    # 训练循环
    if rank == 0:
        print("\n🚀 开始训练...")
    best_macro_acc = 0.0
    patience_counter = 0
    history = []
    
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                args.grad_clip, epoch, args.epochs,
                args.gradient_accumulation_steps, rank
            )
            
            # 评估
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                if rank == 0:
                    print(f"\n📊 Epoch {epoch} 评估...")
                
                eval_results = evaluate_multi_dataset(
                    model, registry, split="test",
                    batch_size=args.eval_batch_size, rank=rank
                )
                
                if rank == 0:
                    print(f"   Train Loss: {train_loss:.4f}")
                    print(f"   Macro Avg Acc: {eval_results['macro_avg']:.4f}")
                    print(f"   Worst 10% Acc: {eval_results['worst_10_pct']:.4f}")
                    print(f"   Per-dataset:")
                    for ds_name, ds_result in eval_results["per_dataset"].items():
                        print(f"      {ds_name}: acc={ds_result['accuracy']:.4f}, τ={ds_result['temperature']:.3f}")
                
                macro_acc = eval_results["macro_avg"]
                
                if macro_acc > best_macro_acc:
                    best_macro_acc = macro_acc
                    patience_counter = 0
                    save_checkpoint(
                        model, optimizer, scheduler, epoch,
                        macro_acc,
                        os.path.join(save_dir, f"stage{args.stage}_best.pt"),
                        args, rank
                    )
                else:
                    patience_counter += 1
                    if rank == 0:
                        print(f"   (无改进, patience: {patience_counter}/{args.early_stop})")
                
                if rank == 0:
                    history.append({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "macro_avg": eval_results["macro_avg"],
                        "worst_10_pct": eval_results["worst_10_pct"],
                        "per_dataset": eval_results["per_dataset"],
                    })
                    with open(os.path.join(save_dir, f"history_stage{args.stage}.json"), "w") as f:
                        json.dump(history, f, indent=2)
            else:
                if rank == 0:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            if patience_counter >= args.early_stop:
                if rank == 0:
                    print(f"\n⏹️ 早停!")
                break
        
        # 最终结果
        if rank == 0:
            print("\n" + "=" * 60)
            print("📋 最终结果")
            print(f"   Best Macro Avg Acc: {best_macro_acc:.4f}")
            
            final_results = {
                "stage": args.stage,
                "best_macro_acc": best_macro_acc,
                "epochs_trained": epoch,
                "config": vars(args),
            }
            
            with open(os.path.join(save_dir, f"final_results_stage{args.stage}.json"), "w") as f:
                json.dump(final_results, f, indent=2)
            
            print("=" * 60)
            print(f"结果保存到: {save_dir}")
    
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
