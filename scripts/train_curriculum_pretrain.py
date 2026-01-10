#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Curriculum Pretraining with Configurable Encoder

在stage1 (TSQA-MCQ) 和 stage2 (M4-Captioning) 数据集上预训练可配置编码器的OpenTSLMSP模型。
支持选择不同的编码器类型（tslanet/transformer_cnn）和加载编码器预训练权重。

使用方法：
    # Stage1 + Stage2 全流程
    python scripts/train_curriculum_pretrain.py \\
        --encoder_type tslanet \\
        --encoder_pretrained pretrained/tslanet_ucr98.pt \\
        --stages stage1_mcq,stage2_captioning

    # 仅Stage1
    python scripts/train_curriculum_pretrain.py \\
        --encoder_type tslanet \\
        --stages stage1_mcq

    # 使用LoRA
    python scripts/train_curriculum_pretrain.py \\
        --encoder_type tslanet \\
        --use_lora \\
        --stages stage1_mcq,stage2_captioning
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.time_series_datasets.TSQADataset import TSQADataset
from opentslm.time_series_datasets.m4.M4QADataset import M4QADataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE, ENCODER_OUTPUT_DIM


# ============================================================================
# 配置默认值
# ============================================================================
DEFAULT_EPOCHS_STAGE1 = 30
DEFAULT_EPOCHS_STAGE2 = 20
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR_ENCODER = 2e-4
DEFAULT_LR_PROJECTOR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_EARLY_STOP = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Curriculum Pretraining with Configurable Encoder")

    # 阶段选择
    parser.add_argument("--stages", type=str, default="stage1_mcq,stage2_captioning",
                        help="训练阶段，用逗号分隔 (stage1_mcq, stage2_captioning)")
    
    # 模型相关
    parser.add_argument("--encoder_type", type=str, default="tslanet", 
                        choices=["transformer_cnn", "tslanet"], help="编码器类型")
    parser.add_argument("--encoder_pretrained", type=str, default=None, 
                        help="编码器预训练权重路径（仅tslanet）")
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B", 
                        help="LLM模型ID")
    parser.add_argument("--tslanet_patch_size", type=int, default=8, 
                        help="TSLANet的patch_size")
    
    # LoRA相关
    parser.add_argument("--use_lora", action="store_true", help="是否启用LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    # 训练相关 - Stage1
    parser.add_argument("--epochs_stage1", type=int, default=DEFAULT_EPOCHS_STAGE1, 
                        help="Stage1训练轮数")
    parser.add_argument("--lr_encoder_stage1", type=float, default=2e-4, 
                        help="Stage1编码器学习率")
    parser.add_argument("--lr_projector_stage1", type=float, default=1e-4, 
                        help="Stage1投影层学习率")
    
    # 训练相关 - Stage2
    parser.add_argument("--epochs_stage2", type=int, default=DEFAULT_EPOCHS_STAGE2, 
                        help="Stage2训练轮数")
    parser.add_argument("--lr_encoder_stage2", type=float, default=2e-4, 
                        help="Stage2编码器学习率")
    parser.add_argument("--lr_projector_stage2", type=float, default=1e-4, 
                        help="Stage2投影层学习率")
    
    # 通用训练参数
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="批次大小")
    parser.add_argument("--lr_lora", type=float, default=1e-4, help="LoRA学习率")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=DEFAULT_GRAD_CLIP, help="梯度裁剪")
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO, help="预热比例")
    parser.add_argument("--early_stop", type=int, default=DEFAULT_EARLY_STOP, help="早停耐心值")
    
    # 保存相关
    parser.add_argument("--save_dir", type=str, default="results/curriculum_pretrain", 
                        help="结果保存目录")
    
    # DDP和梯度相关
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点")
    parser.add_argument("--freeze_encoder", action="store_true", help="冻结编码器参数")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="评估批次大小")
    
    return parser.parse_args()


# ============================================================================
# 分布式训练辅助函数
# ============================================================================
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


def sanitize_llm_id(llm_id: str) -> str:
    """将LLM ID转换为文件系统安全的名称"""
    if not llm_id:
        return "unknown_llm"
    name = llm_id.split("/")[-1]
    name = name.replace(".", "_").replace("-", "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name


# ============================================================================
# 数据加载
# ============================================================================
def create_data_loader(
    dataset_class,
    split: str,
    eos_token: str,
    batch_size: int,
    shuffle: bool,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """创建数据加载器"""
    dataset = dataset_class(split, EOS_TOKEN=eos_token)
    
    def collate_fn(batch):
        return extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        )
    
    sampler = None
    if world_size > 1 and shuffle:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
    )


# ============================================================================
# 评估指标
# ============================================================================
def calculate_accuracy(predictions: List[str], gold_answers: List[str]) -> float:
    """计算MCQ准确率"""
    correct = 0
    total = len(predictions)
    
    for pred, gold in zip(predictions, gold_answers):
        pred_clean = pred.strip()
        gold_clean = gold.strip()
        
        if gold_clean.startswith(pred_clean) or pred_clean == gold_clean:
            correct += 1
    
    return correct / total if total > 0 else 0.0


# ============================================================================
# 训练核心函数
# ============================================================================
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
def validate(
    model,
    val_loader: DataLoader,
    rank: int = 0,
) -> float:
    """验证集评估"""
    model.eval()
    underlying_model = get_model(model)
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Validating", disable=(rank != 0)):
        loss = underlying_model.compute_loss(batch)
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_mcq(
    model,
    test_loader: DataLoader,
    rank: int = 0,
) -> Dict[str, Any]:
    """MCQ评估（带准确率）"""
    model.eval()
    underlying_model = get_model(model)
    
    all_predictions = []
    all_golds = []
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(test_loader, desc="Evaluating", disable=(rank != 0)):
        loss = underlying_model.compute_loss(batch)
        total_loss += loss.item()
        num_batches += 1
        
        predictions = underlying_model.generate(batch, max_new_tokens=50)
        for sample, pred in zip(batch, predictions):
            all_predictions.append(pred)
            all_golds.append(sample["answer"].replace(underlying_model.get_eos_token(), "").strip())
    
    accuracy = calculate_accuracy(all_predictions, all_golds)
    
    return {
        "loss": total_loss / max(num_batches, 1),
        "accuracy": accuracy,
        "predictions": all_predictions[:10],  # 保存少量样本
        "golds": all_golds[:10],
    }


# ============================================================================
# Checkpoint管理
# ============================================================================
def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    save_path: str,
    rank: int = 0,
    extra_metrics: Dict = None,
):
    """保存checkpoint"""
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
    }
    
    if extra_metrics:
        checkpoint.update(extra_metrics)
    
    # 保存LoRA权重
    underlying_model.save_lora_state_to_checkpoint(checkpoint)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"💾 Saved checkpoint to: {save_path}")


def load_checkpoint(
    model,
    checkpoint_path: str,
    optimizer=None,
    scheduler=None,
    device: str = "cuda",
) -> Optional[int]:
    """加载checkpoint，返回epoch数"""
    if not os.path.exists(checkpoint_path):
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    underlying_model = get_model(model)
    
    underlying_model.encoder.load_state_dict(checkpoint["encoder_state"])
    underlying_model.projector.load_state_dict(checkpoint["projector_state"])
    
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    
    # 加载LoRA权重
    underlying_model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)
    
    return checkpoint.get("epoch", 0)


def save_loss_history(save_path: str, epoch: int, train_loss: float, val_loss: float):
    """保存loss历史"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            f.write("Epoch\tTrain_Loss\tVal_Loss\n")
            f.write("-" * 40 + "\n")
    
    with open(save_path, "a") as f:
        f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\n")


# ============================================================================
# Stage训练函数
# ============================================================================
def train_stage(
    stage_name: str,
    model,
    dataset_class,
    num_epochs: int,
    lr_encoder: float,
    lr_projector: float,
    lr_lora: float,
    batch_size: int,
    args,
    save_dir: str,
    world_size: int,
    rank: int,
    device: str,
    metric_func: Callable = None,
    train_sampler_ref: List = None,
) -> Dict[str, Any]:
    """通用Stage训练函数"""
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"🚀 Starting {stage_name}")
        print(f"{'='*60}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Encoder LR: {lr_encoder:.2e}")
        print(f"   Projector LR: {lr_projector:.2e}")
        print(f"   Batch size: {batch_size}")
    
    underlying_model = get_model(model)
    eos_token = underlying_model.get_eos_token()
    
    # 创建数据加载器
    train_loader = create_data_loader(
        dataset_class, "train", eos_token, batch_size, 
        shuffle=True, world_size=world_size, rank=rank
    )
    val_loader = create_data_loader(
        dataset_class, "validation", eos_token, args.eval_batch_size,
        shuffle=False, world_size=1, rank=0
    )
    test_loader = create_data_loader(
        dataset_class, "test", eos_token, args.eval_batch_size,
        shuffle=False, world_size=1, rank=0
    )
    
    if train_sampler_ref is not None:
        train_sampler_ref.append(train_loader.sampler if hasattr(train_loader, 'sampler') else None)
    
    if rank == 0:
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
    
    # 创建优化器
    param_groups = []
    if not args.freeze_encoder:
        param_groups.append({"params": underlying_model.encoder.parameters(), "lr": lr_encoder})
    param_groups.append({"params": underlying_model.projector.parameters(), "lr": lr_projector})
    
    if args.use_lora:
        lora_params = underlying_model.get_lora_parameters()
        if lora_params:
            param_groups.append({"params": lora_params, "lr": lr_lora})
    
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    
    # 学习率调度器
    total_steps = num_epochs * len(train_loader) // args.gradient_accumulation_steps
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    if rank == 0:
        print(f"   Total steps: {total_steps}")
        print(f"   Warmup steps: {warmup_steps}")
    
    # 检查是否有已有checkpoint
    stage_save_dir = os.path.join(save_dir, stage_name)
    checkpoint_path = os.path.join(stage_save_dir, "checkpoints", "best_model.pt")
    start_epoch = 1
    best_val_loss = float("inf")
    
    if os.path.exists(checkpoint_path):
        loaded_epoch = load_checkpoint(model, checkpoint_path, optimizer, scheduler, device)
        if loaded_epoch:
            start_epoch = loaded_epoch + 1
            if rank == 0:
                print(f"📂 Resuming from epoch {loaded_epoch}")
    
    # 训练循环
    patience_counter = 0
    loss_history_path = os.path.join(stage_save_dir, "checkpoints", "loss_history.txt")
    
    for epoch in range(start_epoch, num_epochs + 1):
        # 设置sampler的epoch
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            args.grad_clip, epoch, num_epochs,
            args.gradient_accumulation_steps, rank
        )
        
        val_loss = validate(model, val_loader, rank)
        
        # 同步验证损失
        if dist.is_initialized():
            val_loss_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss = val_loss_tensor.item() / world_size
        
        if rank == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            save_loss_history(loss_history_path, epoch, train_loss, val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                checkpoint_path, rank
            )
            if rank == 0:
                print("✔️ New best model saved.")
        else:
            patience_counter += 1
            if rank == 0:
                print(f"   (No improvement for {patience_counter}/{args.early_stop} epochs)")
        
        # 早停
        if patience_counter >= args.early_stop:
            if rank == 0:
                print(f"⏹️ Early stopping triggered!")
            break
    
    # 加载最佳模型进行测试
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path, device=device)
    
    # 测试评估
    if metric_func:
        if rank == 0:
            print(f"\n📊 Evaluating {stage_name}...")
        metrics = metric_func(model, test_loader, rank)
    else:
        # 仅计算loss
        test_loss = validate(model, test_loader, rank)
        metrics = {"test_loss": test_loss}
    
    # 保存metrics
    if rank == 0:
        metrics_path = os.path.join(stage_save_dir, "results", "metrics.json")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ {stage_name} completed!")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"   {k}: {v:.4f}")
    
    return metrics


# ============================================================================
# 主函数
# ============================================================================
def main():
    args = parse_args()
    
    # 初始化分布式环境
    local_rank, world_size, rank = setup_distributed()
    
    # 解析stages
    stages = [s.strip() for s in args.stages.split(",")]
    valid_stages = ["stage1_mcq", "stage2_captioning"]
    for stage in stages:
        if stage not in valid_stages:
            raise ValueError(f"Unknown stage: {stage}. Valid stages: {valid_stages}")
    
    # 设置设备
    if world_size > 1:
        device = f"cuda:{local_rank}"
    elif args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        if rank == 0:
            print("⚠️ CUDA不可用，使用CPU")
    
    # 设置随机种子
    set_seed(args.seed + rank)
    
    # 打印配置
    if rank == 0:
        print("=" * 60)
        print("Curriculum Pretraining with Configurable Encoder")
        print("=" * 60)
        print(f"时间: {datetime.datetime.now()}")
        print(f"编码器: {args.encoder_type}")
        print(f"编码器预训练: {args.encoder_pretrained}")
        print(f"LLM: {args.llm_id}")
        print(f"训练阶段: {stages}")
        print(f"LoRA: {args.use_lora}")
        print(f"DDP: world_size={world_size}")
        print("=" * 60)
    
    # 创建保存目录
    llm_name = sanitize_llm_id(args.llm_id)
    encoder_name = args.encoder_type
    if args.encoder_pretrained:
        encoder_name += "_pretrained"
    save_dir = os.path.join(args.save_dir, llm_name, encoder_name)
    
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    if world_size > 1:
        dist.barrier()
    
    # 创建模型
    if rank == 0:
        print("\n🔧 创建模型...")
    
    tslanet_config = {
        "patch_size": args.tslanet_patch_size,
        "output_dim": ENCODER_OUTPUT_DIM,
    }
    
    model = OpenTSLMSP(
        llm_id=args.llm_id,
        device=device,
        encoder_type=args.encoder_type,
        encoder_pretrained_path=args.encoder_pretrained,
        tslanet_config=tslanet_config if args.encoder_type == "tslanet" else None,
    )
    
    # 启用梯度检查点
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    
    # 冻结编码器
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        if rank == 0:
            print("🧊 编码器参数已冻结")
    
    # 启用LoRA
    if args.use_lora:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        if rank == 0:
            print("📎 LoRA已启用")
    
    # DDP包装
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
        if rank == 0:
            print(f"✅ 模型已用DDP包装 (world_size={world_size})")
    
    # 训练阶段
    results = {}
    
    try:
        # Stage1: MCQ
        if "stage1_mcq" in stages:
            def mcq_eval_func(model, loader, rank):
                return evaluate_mcq(model, loader, rank)
            
            stage1_results = train_stage(
                stage_name="stage1_mcq",
                model=model,
                dataset_class=TSQADataset,
                num_epochs=args.epochs_stage1,
                lr_encoder=args.lr_encoder_stage1,
                lr_projector=args.lr_projector_stage1,
                lr_lora=args.lr_lora,
                batch_size=args.batch_size,
                args=args,
                save_dir=save_dir,
                world_size=world_size,
                rank=rank,
                device=device,
                metric_func=mcq_eval_func,
            )
            results["stage1_mcq"] = stage1_results
        
        # Stage2: Captioning
        if "stage2_captioning" in stages:
            # 如果之前运行了stage1，确保我们使用stage1的最佳模型
            if "stage1_mcq" in stages:
                stage1_ckpt = os.path.join(save_dir, "stage1_mcq", "checkpoints", "best_model.pt")
                if os.path.exists(stage1_ckpt):
                    load_checkpoint(model, stage1_ckpt, device=device)
                    if rank == 0:
                        print(f"📂 Loaded stage1 checkpoint for stage2 training")
            
            stage2_results = train_stage(
                stage_name="stage2_captioning",
                model=model,
                dataset_class=M4QADataset,
                num_epochs=args.epochs_stage2,
                lr_encoder=args.lr_encoder_stage2,
                lr_projector=args.lr_projector_stage2,
                lr_lora=args.lr_lora,
                batch_size=args.batch_size,
                args=args,
                save_dir=save_dir,
                world_size=world_size,
                rank=rank,
                device=device,
                metric_func=None,  # 仅测试loss
            )
            results["stage2_captioning"] = stage2_results
        
        # 保存总体结果
        if rank == 0:
            overall_results_path = os.path.join(save_dir, "curriculum_results.json")
            with open(overall_results_path, "w") as f:
                json.dump(results, f, indent=2)
            
            print("\n" + "=" * 60)
            print("🎉 Curriculum Pretraining Complete!")
            print(f"📁 Results saved to: {save_dir}")
            print("=" * 60)
    
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
