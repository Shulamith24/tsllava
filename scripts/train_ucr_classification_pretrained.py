#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
M2: UCR单数据集分类训练（基于Stage2预训练模型）

加载curriculum learning的stage2预训练模型进行分类微调。
编码器和投影层解冻，LLM使用LoRA训练。
使用特殊类别token: <c0>, <c1>, ... <cK-1>

使用方法：
    python scripts/train_ucr_classification_pretrained.py \
        --pretrained_model OpenTSLM/llama-3.2-1b-m4-sp \
        --dataset ECG5000 \
        --epochs 30 \
        --batch_size 4

训练配置：
- LoRA: r=16, alpha=32 (默认启用)
- Encoder LR: 2e-4
- Projector LR: 1e-4
- LoRA LR: 1e-4
- 使用特殊类别token (<c0>, <c1>, ...) 替代字母标签
- 约束解码：只允许输出类别token + EOS
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
from transformers import get_linear_schedule_with_warmup, LogitsProcessor, LogitsProcessorList

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import UCRClassificationDataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE, ENCODER_OUTPUT_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="M2: UCR单数据集分类训练（基于Stage2预训练模型）")

    # 必须指定
    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点")
    parser.add_argument("--freeze_encoder", action="store_true", help="冻结编码器参数")

    # 数据相关
    parser.add_argument("--dataset", type=str, default="CricketZ", help="UCR数据集名称")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR数据根目录")
    
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
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr_encoder", type=float, default=2e-4, help="编码器学习率")
    parser.add_argument("--lr_projector", type=float, default=1e-4, help="投影层学习率")
    parser.add_argument("--lr_lora", type=float, default=1e-4, help="LoRA学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热比例")
    
    # 保存相关
    parser.add_argument("--save_dir", type=str, default="results/m2_ucr_pretrained", help="结果保存目录")
    
    # DDP和梯度相关
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--eval_every", type=int, default=5, help="每N轮评估一次")
    parser.add_argument("--early_stop", type=int, default=10, help="早停耐心值")
    parser.add_argument("--max_new_tokens", type=int, default=2, help="生成最大token数（类别token + EOS）")
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


def calculate_accuracy(predictions: List[str], labels: List[str]) -> float:
    """
    计算分类准确率 - 适配特殊token格式 (<c0>, <c1>, ...)
    
    直接比较生成的token与真实标签
    """
    import re
    correct = 0
    for pred, label in zip(predictions, labels):
        pred_clean = pred.strip()
        label_clean = label.strip()
        
        # 尝试从预测中提取 <cN> 格式的token
        match = re.search(r'<c\d+>', pred_clean)
        if match:
            pred_token = match.group()
        else:
            # 如果没有找到，使用整个预测
            pred_token = pred_clean
        
        # 直接比较
        if pred_token == label_clean:
            correct += 1
    
    return correct / len(predictions) if predictions else 0.0


def add_class_tokens_to_model(model, num_classes: int, device: str, rank: int = 0):
    """
    添加类别特殊token到tokenizer和embedding层
    
    Args:
        model: OpenTSLMSP 模型
        num_classes: 类别数量
        device: 设备
        rank: DDP rank
    
    Returns:
        class_tokens: 类别token列表 ['<c0>', '<c1>', ...]
        class_token_ids: 对应的token ID列表
    """
    class_tokens = [f"<c{i}>" for i in range(num_classes)]
    
    # 添加到tokenizer
    num_added = model.tokenizer.add_tokens(class_tokens, special_tokens=True)
    if rank == 0:
        print(f"✅ Added {num_added} class tokens to tokenizer")
    
    # 调整embedding大小
    old_vocab_size = model.llm.get_input_embeddings().weight.shape[0]
    model.llm.resize_token_embeddings(len(model.tokenizer))
    new_vocab_size = model.llm.get_input_embeddings().weight.shape[0]
    
    if rank == 0:
        print(f"   Vocabulary size: {old_vocab_size} -> {new_vocab_size}")
    
    # 改进的初始化：每个类别token使用不同的初始化
    # 从已有token中随机采样，并添加小的扰动
    with torch.no_grad():
        embedding = model.llm.get_input_embeddings()
        lm_head = model.llm.lm_head
        
        if num_added > 0:
            # 获取已有embedding的统计信息
            old_embeddings = embedding.weight[:-num_added]
            emb_mean = old_embeddings.mean(dim=0)
            emb_std = old_embeddings.std(dim=0)
            
            # 为每个类别token生成不同的初始化
            for i in range(num_added):
                # 方法：均值 + 随机扰动 (扰动幅度为标准差的10%)
                noise = torch.randn_like(emb_mean) * emb_std * 0.1
                embedding.weight[-num_added + i] = emb_mean + noise
            
            # 同样处理lm_head
            old_head = lm_head.weight[:-num_added]
            head_mean = old_head.mean(dim=0)
            head_std = old_head.std(dim=0)
            
            for i in range(num_added):
                noise = torch.randn_like(head_mean) * head_std * 0.1
                lm_head.weight[-num_added + i] = head_mean + noise
            
            if rank == 0:
                print(f"   Initialized {num_added} class tokens with mean + random perturbation")
    
    # 确保新token的embedding可训练
    embedding.weight.requires_grad = True
    lm_head.weight.requires_grad = True
    
    # 获取token IDs
    class_token_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in class_tokens]
    if rank == 0:
        print(f"   Class token IDs: {class_token_ids[:5]}..." if len(class_token_ids) > 5 else f"   Class token IDs: {class_token_ids}")
    
    return class_tokens, class_token_ids


class AllowedTokensLogitsProcessor(LogitsProcessor):
    """
    约束解码的Logits处理器：只允许特定token被生成
    """
    def __init__(self, allowed_token_ids: List[int]):
        self.allowed_token_ids = set(allowed_token_ids)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 创建mask，只保留允许的token
        mask = torch.full_like(scores, float('-inf'))
        for token_id in self.allowed_token_ids:
            if token_id < scores.shape[-1]:
                mask[:, token_id] = 0
        return scores + mask


class IndexedDataset(torch.utils.data.Dataset):
    """
    为数据集包装一个索引，用于分布式评估时的去重
    """
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # 添加原始索引到样本中
        sample["_sample_idx"] = idx
        return sample


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
    
    # 用IndexedDataset包装评估数据集，为每个样本添加索引
    indexed_val_dataset = IndexedDataset(val_dataset)
    indexed_test_dataset = IndexedDataset(test_dataset)
    
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
        # 评估集使用分布式采样器（shuffle=False保持顺序）
        val_sampler = DistributedSampler(
            indexed_val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_sampler = DistributedSampler(
            indexed_test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    
    # 评估用DataLoader（使用分布式采样+索引跟踪）
    eval_batch_size = getattr(args, 'eval_batch_size', 8)
    
    val_loader = DataLoader(
        indexed_val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        indexed_test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=test_sampler,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader, test_loader, train_sampler, len(val_dataset), len(test_dataset)


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
        # 计算损失（缩放用于梯度累积）
        # 使用model(batch)调用forward方法，DDP梯度同步在backward()时自动进行
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
    max_new_tokens: int,
    class_token_ids: List[int] | None = None,
    desc: str = "Evaluating",
    rank: int = 0,
    world_size: int = 1,
    total_samples: int | None = None,
) -> Dict[str, Any]:
    """
    分布式评估模型（使用样本索引正确去重）
    
    Args:
        model: 模型（DDP 包装或底层模型都可以）
        data_loader: 数据加载器（使用IndexedDataset + DistributedSampler）
        max_new_tokens: 最大生成token数
        class_token_ids: 类别token的ID列表，用于约束解码
        desc: 进度条描述
        rank: DDP rank
        world_size: GPU 数量
        total_samples: 真实样本数，用于验证去重结果
    """
    import re
    import pickle
    
    # 始终使用底层模型评估
    underlying_model = get_model(model)
    underlying_model.eval()
    
    # 使用字典按索引存储结果（自动去重）
    results_by_idx = {}
    total_loss = 0.0
    num_batches = 0
    
    # 设置约束解码处理器
    logits_processor = None
    if class_token_ids is not None:
        eos_token_id = underlying_model.tokenizer.eos_token_id
        allowed_ids = class_token_ids + [eos_token_id]
        logits_processor = LogitsProcessorList([AllowedTokensLogitsProcessor(allowed_ids)])
    
    for batch in tqdm(data_loader, desc=desc, disable=(rank != 0)):
        # 使用底层模型
        loss = underlying_model.compute_loss(batch)
        total_loss += loss.item()
        num_batches += 1
        
        # 生成预测（使用约束解码）
        if logits_processor is not None:
            inputs_embeds, attention_mask = underlying_model.pad_and_apply_batch(batch)
            gen_ids = underlying_model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_processor,
                do_sample=False,
            )
            predictions = underlying_model.tokenizer.batch_decode(gen_ids, skip_special_tokens=False)
            # 清理多余的特殊token，保留<cN>格式
            cleaned_predictions = []
            for p in predictions:
                match = re.search(r'<c\d+>', p)
                if match:
                    cleaned_predictions.append(match.group())
                else:
                    cleaned_predictions.append(p.strip())
            predictions = cleaned_predictions
        else:
            predictions = underlying_model.generate(batch, max_new_tokens=max_new_tokens)
        
        # 收集结果（使用样本索引作为key）
        for sample, pred in zip(batch, predictions):
            idx = sample.get("_sample_idx", -1)
            label = sample["answer"].replace(underlying_model.get_eos_token(), "").strip()
            results_by_idx[idx] = {"prediction": pred, "label": label}
    
    # 分布式聚合：收集所有 rank 的结果
    if world_size > 1:
        # 序列化本地结果
        local_data = pickle.dumps({
            "results_by_idx": results_by_idx,
            "loss": total_loss,
            "num_batches": num_batches,
        })
        local_size = torch.tensor([len(local_data)], device=underlying_model.device)
        
        # 收集所有 rank 的数据大小
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        max_size = max(s.item() for s in all_sizes)
        
        # 填充到相同大小
        local_tensor = torch.zeros(int(max_size), dtype=torch.uint8, device=underlying_model.device)
        local_tensor[:len(local_data)] = torch.tensor(list(local_data), dtype=torch.uint8, device=underlying_model.device)
        
        # 收集所有数据
        all_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(all_tensors, local_tensor)
        
        # 反序列化并合并（字典自动去重：相同索引只保留一份）
        merged_results = {}
        total_loss = 0.0
        num_batches = 0
        
        for tensor, size in zip(all_tensors, all_sizes):
            data = pickle.loads(bytes(tensor[:size.item()].cpu().tolist()))
            merged_results.update(data["results_by_idx"])  # 自动去重
            total_loss += data["loss"]
            num_batches += data["num_batches"]
        
        results_by_idx = merged_results
    
    # 按索引排序并提取结果
    sorted_indices = sorted(results_by_idx.keys())
    all_predictions = [results_by_idx[idx]["prediction"] for idx in sorted_indices]
    all_labels = [results_by_idx[idx]["label"] for idx in sorted_indices]
    
    # 验证样本数量
    if total_samples is not None and len(all_predictions) != total_samples:
        if rank == 0:
            print(f"⚠️ 警告: 期望 {total_samples} 个样本，实际 {len(all_predictions)} 个")
    
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
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "args": vars(args),
    }
    
    # 保存LoRA权重
    underlying_model.save_lora_state_to_checkpoint(checkpoint)
    
    # 保存 class token 的 embedding 和 lm_head 权重
    # 这些是训练时新添加的特殊 token，必须保存
    checkpoint["embedding_weight"] = underlying_model.llm.get_input_embeddings().weight.detach().cpu()
    checkpoint["lm_head_weight"] = underlying_model.llm.lm_head.weight.detach().cpu()
    checkpoint["tokenizer_vocab_size"] = len(underlying_model.tokenizer)
    
    torch.save(checkpoint, save_path)
    print(f"💾 Saved checkpoint to: {save_path}")


def main():
    args = parse_args()
    
    # 初始化分布式环境
    local_rank, world_size, rank = setup_distributed()
    
    # 仅rank=0打印信息
    if rank == 0:
        print("=" * 60)
        print("M2: UCR单数据集分类训练（基于Stage2预训练模型）")
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
    set_seed(args.seed + rank)  # 每个rank使用不同的随机种子
    
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
    
    # 冻结编码器（可选）
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        if rank == 0:
            print("🧊 编码器参数已冻结")
    
    # DDP包装
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
        if rank == 0:
            print(f"✅ 模型已用DDP包装 (world_size={world_size})")
    
    # 创建数据加载器
    if rank == 0:
        print("\n📂 加载数据...")
    eos_token = get_model(model).get_eos_token()
    train_loader, val_loader, test_loader, train_sampler, val_size, test_size = create_data_loaders(
        args, eos_token, world_size, rank
    )
    
    if rank == 0:
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
    
    # 添加类别特殊token到模型
    if rank == 0:
        print("\n🎯 添加类别token...")
    num_classes = UCRClassificationDataset.get_num_classes()
    underlying_model_for_tokens = get_model(model)
    class_tokens, class_token_ids = add_class_tokens_to_model(
        underlying_model_for_tokens, num_classes, device, rank
    )
    
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
    
    # 添加新增的类别token的embedding和lm_head权重到优化器
    # 这些权重需要更高的学习率来快速学习
    embedding_weight = underlying_model.llm.get_input_embeddings().weight
    lm_head_weight = underlying_model.llm.lm_head.weight
    param_groups.append({
        "params": [embedding_weight, lm_head_weight], 
        "lr": args.lr_lora * 2  # 使用更高的学习率
    })
    if rank == 0:
        print(f"   Added embedding and lm_head to optimizer (lr={args.lr_lora * 2:.2e})")
    
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
    epoch = 0  # 初始化防止unbound
    
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
            # 所有rank参与分布式评估，使用索引自动去重
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                if rank == 0:
                    print(f"\n📊 Epoch {epoch} 评估...")
                
                # 分布式评估：所有rank参与，结果通过索引自动去重
                val_results = evaluate(
                    model, val_loader, args.max_new_tokens, 
                    class_token_ids=class_token_ids, desc="Validating",
                    rank=rank, world_size=world_size, total_samples=val_size
                )
                val_loss = val_results["loss"]
                val_acc = val_results["accuracy"]
                
                if rank == 0:
                    print(f"   Train Loss: {train_loss:.4f}")
                    print(f"   Val Loss: {val_loss:.4f}")
                    print(f"   Val Accuracy: {val_acc:.4f}")
                    
                    # 显示一些预测样本
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
                        save_checkpoint(
                            model, optimizer, scheduler, epoch,
                            val_loss, val_acc,
                            os.path.join(save_dir, "best_model.pt"),
                            args, rank
                        )
                    else:
                        patience_counter += 1
                        print(f"   (无改进, patience: {patience_counter}/{args.early_stop})")
                    
                    # 记录历史
                    loss_history.append({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    })
                    with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
                        json.dump(loss_history, f, indent=2)
                
                # 同步 patience_counter 和 best_val_acc 给所有 rank
                if world_size > 1:
                    patience_tensor = torch.tensor([patience_counter], device=device)
                    best_val_acc_tensor = torch.tensor([best_val_acc], device=device)
                    dist.broadcast(patience_tensor, src=0)
                    dist.broadcast(best_val_acc_tensor, src=0)
                    patience_counter = int(patience_tensor.item())
                    best_val_acc = float(best_val_acc_tensor.item())
            else:
                if rank == 0:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # 早停（所有 rank 同步检查）
            if patience_counter >= args.early_stop:
                if rank == 0:
                    print(f"\n⏹️ 早停! 验证准确率 {args.early_stop} 轮未改进")
                break
        
        # 最终测试（所有rank参与分布式测试）
        if rank == 0:
            print("\n" + "=" * 60)
            print("📋 最终测试评估...")
        
        # 所有rank加载最佳模型
        best_ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device, weights_only=False)
        underlying_model.encoder.load_state_dict(best_ckpt["encoder_state"])
        underlying_model.projector.load_state_dict(best_ckpt["projector_state"])
        underlying_model.load_lora_state_from_checkpoint(best_ckpt, allow_missing=True)
        
        # 恢复 class token 的 embedding 和 lm_head 权重
        if "embedding_weight" in best_ckpt:
            with torch.no_grad():
                underlying_model.llm.get_input_embeddings().weight.copy_(
                    best_ckpt["embedding_weight"].to(device)
                )
                underlying_model.llm.lm_head.weight.copy_(
                    best_ckpt["lm_head_weight"].to(device)
                )
            if rank == 0:
                print("📥 Loaded embedding and lm_head weights")
        
        # 同步所有rank，确保都加载完成权重后再开始测试
        if world_size > 1:
            dist.barrier()
        
        # 分布式测试评估
        test_results = evaluate(
            model, test_loader, args.max_new_tokens,
            class_token_ids=class_token_ids, desc="Testing",
            rank=rank, world_size=world_size, total_samples=test_size
        )
        
        if rank == 0:
            print(f"\n✅ 测试结果:")
            print(f"   Test Loss: {test_results['loss']:.4f}")
            print(f"   Test Accuracy: {test_results['accuracy']:.4f}")
            
            # 保存测试结果
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
            
            # 保存测试预测
            with open(os.path.join(save_dir, "test_predictions.json"), "w") as f:
                json.dump({
                    "predictions": test_results["predictions"],
                    "labels": test_results["labels"],
                }, f, indent=2)
            
            print("=" * 60)
            print(f"结果保存到: {save_dir}")
            print("=" * 60)
    
    finally:
        # 清理分布式环境
        cleanup_distributed()


if __name__ == "__main__":
    main()
