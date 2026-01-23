#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
UCR单数据集Prototype分类训练脚本

使用Prototype分类头替代生成式解码，架构:
[Learnable Prompt] + [TS_tokens] + [CLS] → LLM → CLS隐向量 → Prototype头 → logits

两阶段训练:
- Stage 0: 冻结backbone，只训练 prompt + cls + prototypes + temperature
- Stage 1: 解冻 encoder + projector + LoRA，联合训练

使用方法:
    # Stage 0 (快速收敛)
    python scripts/train_ucr_prototype_single.py \
        --pretrained_model OpenTSLM/llama-3.2-1b-m4-sp \
        --dataset ECG200 \
        --stage 0 \
        --epochs 10

    # Stage 1 (联合训练)
    python scripts/train_ucr_prototype_single.py \
        --pretrained_model OpenTSLM/llama-3.2-1b-m4-sp \
        --dataset ECG200 \
        --stage 1 \
        --epochs 30 \
        --resume_from results/prototype_ucr/ECG200/stage0_best.pt
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
from opentslm.model.llm.OpenTSLMPrototype import OpenTSLMPrototype
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import UCRClassificationDataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE, ENCODER_OUTPUT_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="UCR单数据集Prototype分类训练")
    
    # 数据相关
    parser.add_argument("--dataset", type=str, default="ECG200", help="UCR数据集名称")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR数据根目录")
    
    # 模型相关
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="预训练模型ID (HuggingFace repo_id)")
    parser.add_argument("--local_checkpoint", type=str, default=None,
                        help="本地checkpoint路径")
    parser.add_argument("--encoder_type", type=str, default="transformer_cnn",
                        choices=["transformer_cnn", "tslanet"])
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--tslanet_patch_size", type=int, default=8)
    
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
                        help="从Stage0 checkpoint加载模型权重（不恢复训练状态，用于Stage1加载Stage0）")
    parser.add_argument("--continue_training", type=str, default=None,
                        help="从checkpoint完全恢复训练（包括epoch/optimizer/scheduler）")
    
    # 训练超参
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_prompt", type=float, default=1e-3, help="Prompt/CLS学习率")
    parser.add_argument("--lr_head", type=float, default=1e-3, help="Prototype头学习率")
    parser.add_argument("--lr_encoder", type=float, default=2e-4, help="Encoder学习率")
    parser.add_argument("--lr_projector", type=float, default=1e-4, help="Projector学习率")
    parser.add_argument("--lr_lora", type=float, default=1e-4, help="LoRA学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    
    # 保存相关
    parser.add_argument("--save_dir", type=str, default="results/prototype_ucr")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--overfit_test", action="store_true", help="使用小子集测试overfit")
    
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


class PrototypeDataset(torch.utils.data.Dataset):
    """
    适配Prototype模型的数据集包装器
    
    将UCRClassificationDataset的样本转换为Prototype模型需要的格式
    """
    def __init__(self, ucr_dataset: UCRClassificationDataset):
        self.ucr_dataset = ucr_dataset
        # 构建类别token到索引的映射
        self._build_label_mapping()
    
    def _build_label_mapping(self):
        """构建标签映射"""
        label_mapping = UCRClassificationDataset.get_label_mapping()
        self.token_to_index = {token: i for i, token in enumerate(sorted(label_mapping.values()))}
    
    def __len__(self):
        return len(self.ucr_dataset)
    
    def __getitem__(self, idx):
        sample = self.ucr_dataset[idx]
        
        # 将类别token转换为整数索引
        class_token = sample.get("class_token", sample["answer"].replace(self.ucr_dataset.EOS_TOKEN, "").strip())
        label_index = self.token_to_index.get(class_token, 0)
        
        return {
            "time_series": sample["time_series"],
            "label_index": label_index,
            "_sample_idx": idx,
        }


def create_data_loaders(args, eos_token: str, world_size: int = 1, rank: int = 0):
    """创建数据加载器"""
    # 创建原始数据集
    train_dataset_raw = UCRClassificationDataset(
        split="train",
        EOS_TOKEN=eos_token,
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )
    
    val_dataset_raw = UCRClassificationDataset(
        split="validation",
        EOS_TOKEN=eos_token,
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )
    
    test_dataset_raw = UCRClassificationDataset(
        split="test",
        EOS_TOKEN=eos_token,
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )
    
    # 包装为Prototype格式
    train_dataset = PrototypeDataset(train_dataset_raw)
    val_dataset = PrototypeDataset(val_dataset_raw)
    test_dataset = PrototypeDataset(test_dataset_raw)
    
    # Overfit测试：只使用前10个样本
    if args.overfit_test:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(10, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(10, len(val_dataset))))
        test_dataset = Subset(test_dataset, range(min(10, len(test_dataset))))
        if rank == 0:
            print(f"⚠️ Overfit test: 使用前10个样本")
    
    def collate_fn(batch):
        """整理批次数据"""
        # 处理时间序列padding
        processed = extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        )
        return processed
    
    # 采样器
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
def evaluate(
    model,
    data_loader: DataLoader,
    desc: str = "Evaluating",
    rank: int = 0,
) -> Dict[str, Any]:
    """评估模型"""
    underlying_model = get_model(model)
    underlying_model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    
    for batch in tqdm(data_loader, desc=desc, disable=(rank != 0)):
        loss, logits = underlying_model.forward_prototype(batch)
        predictions = logits.argmax(dim=-1)
        labels = torch.tensor([s["label_index"] for s in batch], device=logits.device)
        
        total_loss += loss.item()
        total_correct += (predictions == labels).sum().item()
        total_samples += len(batch)
        
        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
    
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
    """保存checkpoint"""
    if rank != 0:
        return
    
    underlying_model = get_model(model)
    checkpoint = {
        "encoder_state": underlying_model.encoder.state_dict(),
        "projector_state": underlying_model.projector.state_dict(),
        "prompt_embeds": underlying_model.prompt_embeds.data,
        "cls_embed": underlying_model.cls_embed.data,
        "cls_projector_state": underlying_model.cls_projector.state_dict(),
        "cls_head_state": underlying_model.cls_head.state_dict(),
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
    
    # 初始化分布式
    local_rank, world_size, rank = setup_distributed()
    
    if rank == 0:
        print("=" * 60)
        print("UCR单数据集Prototype分类训练")
        print("=" * 60)
        print(f"时间: {datetime.datetime.now()}")
        print(f"数据集: {args.dataset}")
        print(f"Stage: {args.stage}")
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
    save_dir = os.path.join(args.save_dir, args.dataset)
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"config_stage{args.stage}.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    if world_size > 1:
        dist.barrier()
    
    # 临时创建数据集获取类别数
    if rank == 0:
        print("\n📂 加载数据获取类别数...")
    temp_dataset = UCRClassificationDataset(
        split="train",
        EOS_TOKEN="<eos>",
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )
    num_classes = UCRClassificationDataset.get_num_classes()
    if rank == 0:
        print(f"   类别数: {num_classes}")
    
    # 创建模型
    if rank == 0:
        print("\n🔧 创建模型...")
    
    use_lora = not args.no_lora
    
    if args.local_checkpoint:
        # 从本地checkpoint创建
        tslanet_config = {"patch_size": args.tslanet_patch_size, "output_dim": ENCODER_OUTPUT_DIM}
        model = OpenTSLMPrototype(
            llm_id=args.llm_id,
            device=device,
            encoder_type=args.encoder_type,
            tslanet_config=tslanet_config if args.encoder_type == "tslanet" else None,
            prompt_len=args.prompt_len,
            num_classes=num_classes,
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
    
    elif args.pretrained_model:
        # 从HuggingFace创建 - 需要先创建基础模型再转换
        if rank == 0:
            print(f"📂 从HuggingFace加载: {args.pretrained_model}")
        
        # 先加载OpenTSLM获取权重
        base_model = OpenTSLM.load_pretrained(
            repo_id=args.pretrained_model,
            device=device,
            enable_lora=False,  # 先不启用LoRA
        )
        
        # 创建Prototype模型
        model = OpenTSLMPrototype(
            llm_id=base_model.llm.config._name_or_path if hasattr(base_model.llm.config, '_name_or_path') else "meta-llama/Llama-3.2-1B",
            device=device,
            encoder_type="transformer_cnn",
            prompt_len=args.prompt_len,
            num_classes=num_classes,
            init_temperature=args.init_temperature,
        )
        
        # 复制权重
        model.encoder.load_state_dict(base_model.encoder.state_dict())
        model.projector.load_state_dict(base_model.projector.state_dict())
        
        if use_lora:
            model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        
        del base_model
        torch.cuda.empty_cache()
    
    else:
        raise ValueError("必须指定 --pretrained_model 或 --local_checkpoint")
    
    # 从resume加载（如果有）
    if args.resume_from:
        if rank == 0:
            print(f"📂 从{args.resume_from}恢复...")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.prompt_embeds.data = ckpt["prompt_embeds"].to(device)
        model.cls_embed.data = ckpt["cls_embed"].to(device)
        if "cls_projector_state" in ckpt:
            model.cls_projector.load_state_dict(ckpt["cls_projector_state"])
        model.cls_head.load_state_dict(ckpt["cls_head_state"])
        if rank == 0:
            print(f"✅ 恢复prompt/cls/projector/head")
    
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
    
    # 数据加载器
    if rank == 0:
        print("\n📂 创建数据加载器...")
    eos_token = get_model(model).get_eos_token()
    train_loader, val_loader, test_loader, train_sampler = create_data_loaders(
        args, eos_token, world_size, rank
    )
    if rank == 0:
        print(f"   Train: {len(train_loader)} batches")
        print(f"   Val: {len(val_loader)} batches")
        print(f"   Test: {len(test_loader)} batches")
    
    # 优化器
    if rank == 0:
        print("\n⚙️ 创建优化器...")
    underlying_model = get_model(model)
    
    param_groups = []
    if args.stage == 0:
        # Stage 0: 只训练prompt/cls/projector/head
        param_groups.append({"params": [underlying_model.prompt_embeds, underlying_model.cls_embed], "lr": args.lr_prompt})
        param_groups.append({"params": list(underlying_model.cls_projector.parameters()), "lr": args.lr_head})
        param_groups.append({"params": list(underlying_model.cls_head.parameters()), "lr": args.lr_head})
    else:
        # Stage 1: 全部训练
        param_groups.append({"params": list(underlying_model.encoder.parameters()), "lr": args.lr_encoder})
        param_groups.append({"params": list(underlying_model.projector.parameters()), "lr": args.lr_projector})
        param_groups.append({"params": [underlying_model.prompt_embeds, underlying_model.cls_embed], "lr": args.lr_prompt})
        param_groups.append({"params": list(underlying_model.cls_projector.parameters()), "lr": args.lr_head})
        param_groups.append({"params": list(underlying_model.cls_head.parameters()), "lr": args.lr_head})
        
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
    best_val_acc = 0.0
    patience_counter = 0
    loss_history = []
    start_epoch = 1  # 默认从1开始
    
    # 如果是完全恢复训练，加载训练状态
    if args.continue_training:
        ckpt = torch.load(args.continue_training, map_location=device, weights_only=False)
        
        # 恢复模型权重
        underlying_model.prompt_embeds.data = ckpt["prompt_embeds"].to(device)
        underlying_model.cls_embed.data = ckpt["cls_embed"].to(device)
        if "cls_projector_state" in ckpt:
            underlying_model.cls_projector.load_state_dict(ckpt["cls_projector_state"])
        underlying_model.cls_head.load_state_dict(ckpt["cls_head_state"])
        underlying_model.encoder.load_state_dict(ckpt["encoder_state"])
        underlying_model.projector.load_state_dict(ckpt["projector_state"])
        underlying_model.load_lora_state_from_checkpoint(ckpt, allow_missing=True)
        
        # 恢复优化器和调度器状态
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        
        # 恢夌epoch
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("val_acc", 0.0)
        
        if rank == 0:
            print(f"✅ 从epoch {start_epoch} 恢复训练")
            print(f"   上次best_val_acc: {best_val_acc:.4f}")
    
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                args.grad_clip, epoch, args.epochs,
                args.gradient_accumulation_steps, rank
            )
            
            # 评估
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                if rank == 0:
                    print(f"\n📊 Epoch {epoch} 评估...")
                
                val_results = evaluate(model, val_loader, "Validating", rank)
                val_loss = val_results["loss"]
                val_acc = val_results["accuracy"]
                
                if rank == 0:
                    print(f"   Train Loss: {train_loss:.4f}")
                    print(f"   Val Loss: {val_loss:.4f}")
                    print(f"   Val Accuracy: {val_acc:.4f}")
                    
                    # 温度信息
                    temp = underlying_model.cls_head.temperature.item()
                    print(f"   Temperature: {temp:.4f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    save_checkpoint(
                        model, optimizer, scheduler, epoch,
                        val_loss, val_acc,
                        os.path.join(save_dir, f"stage{args.stage}_best.pt"),
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
                    with open(os.path.join(save_dir, f"loss_history_stage{args.stage}.json"), "w") as f:
                        json.dump(loss_history, f, indent=2)
            else:
                if rank == 0:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            if patience_counter >= args.early_stop:
                if rank == 0:
                    print(f"\n⏹️ 早停!")
                break
        
        # 最终测试
        if rank == 0:
            print("\n" + "=" * 60)
            print("📋 最终测试...")
            
            # 加载最佳模型
            best_path = os.path.join(save_dir, f"stage{args.stage}_best.pt")
            if os.path.exists(best_path):
                ckpt = torch.load(best_path, map_location=device, weights_only=False)
                underlying_model.prompt_embeds.data = ckpt["prompt_embeds"].to(device)
                underlying_model.cls_embed.data = ckpt["cls_embed"].to(device)
                if "cls_projector_state" in ckpt:
                    underlying_model.cls_projector.load_state_dict(ckpt["cls_projector_state"])
                underlying_model.cls_head.load_state_dict(ckpt["cls_head_state"])
                underlying_model.encoder.load_state_dict(ckpt["encoder_state"])
                underlying_model.projector.load_state_dict(ckpt["projector_state"])
            
            test_results = evaluate(model, test_loader, "Testing", rank)
            
            print(f"\n✅ 测试结果:")
            print(f"   Test Loss: {test_results['loss']:.4f}")
            print(f"   Test Accuracy: {test_results['accuracy']:.4f}")
            
            final_results = {
                "dataset": args.dataset,
                "stage": args.stage,
                "best_val_acc": best_val_acc,
                "test_loss": test_results["loss"],
                "test_accuracy": test_results["accuracy"],
                "epochs_trained": epoch,
            }
            
            with open(os.path.join(save_dir, f"final_results_stage{args.stage}.json"), "w") as f:
                json.dump(final_results, f, indent=2)
            
            print("=" * 60)
            print(f"结果保存到: {save_dir}")
    
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
