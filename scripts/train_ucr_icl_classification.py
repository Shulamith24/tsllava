#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
UCR ICL分类训练脚本

使用In-Context Learning范式进行时间序列分类训练。

训练流程：
1. 加载预训练的OpenTSLM SP模型
2. 加载训练好的TSLANet encoder用于检索
3. 构建检索索引
4. 使用ICL格式进行训练/测试

使用方法：
    # 首先训练TSLANet
    python scripts/train_tslanet_ucr.py --dataset ECG5000
    
    # 然后进行ICL分类训练
    python scripts/train_ucr_icl_classification.py \\
        --dataset ECG5000 \\
        --pretrained_model OpenTSLM/llama-3.2-1b-m4-sp \\
        --tslanet_checkpoint results/tslanet_ucr/ECG5000/best_model.pt \\
        --k_shot 1 \\
        --epochs 10
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
from opentslm.model.encoder.TSLANetEncoder import TSLANetEncoder
from opentslm.retrieval.TSLANetRetriever import TSLANetRetriever
from opentslm.time_series_datasets.ucr.UCRICLClassificationDataset import (
    UCRICLClassificationDataset,
    create_icl_collate_fn
)
from opentslm.time_series_datasets.ucr.ucr_loader import load_ucr_dataset, ensure_ucr_data
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import index_to_excel_label
from opentslm.model_config import PATCH_SIZE, ENCODER_OUTPUT_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="UCR ICL分类训练")

    # 数据相关
    parser.add_argument("--dataset", type=str, default="ECG5000", help="UCR数据集名称")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR数据根目录")
    
    # 模型相关 - OpenTSLM
    parser.add_argument("--pretrained_model", type=str, default=None, 
                        help="预训练模型ID (HuggingFace repo_id)")
    parser.add_argument("--local_checkpoint", type=str, default=None,
                        help="本地checkpoint路径")
    parser.add_argument("--encoder_type", type=str, default="tslanet",
                        choices=["transformer_cnn", "tslanet"],
                        help="编码器类型")
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B",
                        help="LLM模型ID")
    
    # 模型相关 - TSLANet检索器
    parser.add_argument("--tslanet_checkpoint", type=str, required=True,
                        help="TSLANet分类器checkpoint路径 (用于检索)")
    
    # ICL相关
    parser.add_argument("--k_shot", type=int, default=1, 
                        help="每个类别的支持样本数")
    parser.add_argument("--top_m", type=int, default=10,
                        help="每个类别检索的候选数量")
    
    # LoRA相关
    parser.add_argument("--no_lora", action="store_true", help="禁用LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    # 训练相关
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--lr_encoder", type=float, default=2e-4, help="编码器学习率")
    parser.add_argument("--lr_projector", type=float, default=1e-4, help="投影层学习率")
    parser.add_argument("--lr_lora", type=float, default=1e-4, help="LoRA学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热比例")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点")
    parser.add_argument("--freeze_encoder", action="store_true", help="冻结编码器参数")
    
    # 保存相关
    parser.add_argument("--save_dir", type=str, default="results/icl_classification", help="结果保存目录")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--eval_every", type=int, default=1, help="每N轮评估一次")
    parser.add_argument("--early_stop", type=int, default=5, help="早停耐心值")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="生成最大token数")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="评估批次大小")
    
    return parser.parse_args()


def setup_distributed():
    """初始化分布式训练环境"""
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
        pred_clean = pred.strip().upper()
        label_clean = label.strip().upper()
        
        # 提取预测标签
        pred_label = None
        if len(pred_clean) == 1 and pred_clean.isalpha():
            pred_label = pred_clean
        elif len(pred_clean) == 2 and pred_clean.isalpha():
            pred_label = pred_clean  # AA, AB等
        elif pred_clean:
            # 取最后一个词
            words = pred_clean.split()
            if words:
                last_word = words[-1].strip(".,!?:;")
                if len(last_word) <= 2 and last_word.isalpha():
                    pred_label = last_word.upper()
        
        if pred_label == label_clean:
            correct += 1
    
    return correct / len(predictions) if predictions else 0.0


def load_tslanet_for_retrieval(checkpoint_path: str, device: str):
    """加载TSLANet用于检索"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    encoder_state = checkpoint["encoder_state"]
    patch_size = config.get("patch_size", 8)
    
    # 获取max_seq_len: 优先使用保存的值，否则从pos_embed推断
    if "max_seq_len" in checkpoint:
        max_seq_len = checkpoint["max_seq_len"]
    else:
        # 从pos_embed形状推断 (兼容旧版本checkpoint)
        pos_embed_shape = encoder_state["pos_embed"].shape  # [1, num_patches, emb_dim]
        num_patches = pos_embed_shape[1]
        stride = patch_size // 2
        max_seq_len = (num_patches - 1) * stride + patch_size
    
    # 创建encoder
    encoder = TSLANetEncoder(
        output_dim=config.get("emb_dim", 128),
        dropout=config.get("dropout", 0.15),
        patch_size=patch_size,
        emb_dim=config.get("emb_dim", 128),
        depth=config.get("depth", 2),
        max_seq_len=max_seq_len
    )
    
    # 加载权重
    encoder.load_state_dict(encoder_state)
    encoder = encoder.to(device)
    encoder.eval()
    
    print(f"✅ 加载TSLANet检索器: {checkpoint_path}")
    print(f"   序列长度: {checkpoint.get('seq_len', 'unknown')}")
    print(f"   类别数: {checkpoint.get('num_classes', 'unknown')}")
    print(f"   max_seq_len: {max_seq_len}")
    
    return encoder, checkpoint


def create_datasets(args, retriever, eos_token: str):
    """创建ICL Dataset"""
    ensure_ucr_data()
    
    # 加载数据
    train_df, test_df = load_ucr_dataset(args.dataset, raw_data_path=args.data_path)
    
    # 获取类别信息
    all_labels = sorted(train_df["label"].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    
    # 提取时间序列和标签
    feature_cols = [col for col in train_df.columns if col != "label"]
    
    def df_to_tensors(df):
        ts = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        labels = torch.tensor([label_to_idx[l] for l in df["label"]], dtype=torch.long)
        return ts, labels
    
    train_ts, train_labels = df_to_tensors(train_df)
    test_ts, test_labels = df_to_tensors(test_df)
    
    print(f"📊 Dataset: {args.dataset}")
    print(f"   Classes: {len(all_labels)}")
    print(f"   Train samples: {len(train_ts)}")
    print(f"   Test samples: {len(test_ts)}")
    
    # 构建检索索引 (只用训练集)
    print("\n🔧 构建检索索引...")
    retriever.build_index(train_ts, train_labels)
    
    # 创建Dataset
    train_dataset = UCRICLClassificationDataset(
        time_series=train_ts,
        labels=train_labels,
        retriever=retriever,
        dataset_name=args.dataset,
        k_shot=args.k_shot,
        top_m=args.top_m,
        eos_token=eos_token,
        split="train",
        exclude_query=True
    )
    
    # 测试集也用训练集的索引进行检索
    test_dataset = UCRICLClassificationDataset(
        time_series=test_ts,
        labels=test_labels,
        retriever=retriever,
        dataset_name=args.dataset,
        k_shot=args.k_shot,
        top_m=args.top_m,
        eos_token=eos_token,
        split="test",
        exclude_query=False
    )
    
    return train_dataset, test_dataset


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
    """评估模型"""
    model.eval()
    underlying_model = get_model(model)
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(data_loader, desc=desc, disable=(rank != 0)):
        loss = underlying_model.compute_loss(batch)
        total_loss += loss.item()
        num_batches += 1
        
        predictions = underlying_model.generate(batch, max_new_tokens=max_new_tokens)
        
        for sample, pred in zip(batch, predictions):
            all_predictions.append(pred)
            all_labels.append(sample["letter_label"])
    
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
    
    # 初始化分布式环境
    local_rank, world_size, rank = setup_distributed()
    
    if rank == 0:
        print("=" * 60)
        print("UCR ICL分类训练")
        print("=" * 60)
        print(f"时间: {datetime.datetime.now()}")
        print(f"数据集: {args.dataset}")
        print(f"K-shot: {args.k_shot}")
        print(f"Top-M: {args.top_m}")
        print("=" * 60)
    
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
    
    # 创建保存目录
    save_dir = os.path.join(args.save_dir, args.dataset)
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    if world_size > 1:
        dist.barrier()
    
    # 加载TSLANet用于检索
    if rank == 0:
        print("\n🔧 加载TSLANet检索器...")
    tslanet_encoder, tslanet_ckpt = load_tslanet_for_retrieval(args.tslanet_checkpoint, device)
    retriever = TSLANetRetriever(tslanet_encoder, device=device)
    
    # 加载OpenTSLM模型
    if rank == 0:
        print("\n🔧 加载OpenTSLM模型...")
    
    use_lora = not args.no_lora
    
    if args.pretrained_model:
        model = OpenTSLM.load_pretrained(
            repo_id=args.pretrained_model,
            device=device,
            enable_lora=use_lora,
        )
        if use_lora and (args.lora_r != 16 or args.lora_alpha != 32):
            model.disable_lora()
            model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    elif args.local_checkpoint:
        tslanet_config = {
            "patch_size": tslanet_ckpt.get("config", {}).get("patch_size", 8),
            "output_dim": ENCODER_OUTPUT_DIM,
        }
        model = OpenTSLMSP(
            llm_id=args.llm_id,
            device=device,
            encoder_type=args.encoder_type,
            tslanet_config=tslanet_config if args.encoder_type == "tslanet" else None,
        )
        checkpoint = torch.load(args.local_checkpoint, map_location=device, weights_only=False)
        model.encoder.load_state_dict(checkpoint["encoder_state"])
        model.projector.load_state_dict(checkpoint["projector_state"])
        if use_lora:
            model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
            model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)
    else:
        raise ValueError("必须指定 --pretrained_model 或 --local_checkpoint")
    
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        if rank == 0:
            print("🧊 编码器参数已冻结")
    
    # 创建数据集
    if rank == 0:
        print("\n📂 创建ICL数据集...")
    eos_token = get_model(model).get_eos_token() if hasattr(model, "module") else model.get_eos_token()
    train_dataset, test_dataset = create_datasets(args, retriever, eos_token)
    
    # 创建DataLoader
    collate_fn = create_icl_collate_fn(patch_size=PATCH_SIZE)
    
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    if rank == 0:
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Test batches: {len(test_loader)}")
    
    # DDP包装
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
        if rank == 0:
            print(f"✅ 模型已用DDP包装 (world_size={world_size})")
    
    # 创建优化器
    underlying_model = get_model(model)
    param_groups = []
    if not args.freeze_encoder:
        param_groups.append({"params": underlying_model.encoder.parameters(), "lr": args.lr_encoder})
    param_groups.append({"params": underlying_model.projector.parameters(), "lr": args.lr_projector})
    
    if use_lora:
        lora_params = underlying_model.get_lora_parameters()
        if lora_params:
            param_groups.append({"params": lora_params, "lr": args.lr_lora})
    
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    
    # 学习率调度器
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    if rank == 0:
        print(f"\n⚙️ 训练配置:")
        print(f"   Effective batch size: {effective_batch_size}")
        print(f"   Total steps: {total_steps}")
        print(f"   Warmup steps: {warmup_steps}")
    
    # 训练循环
    if rank == 0:
        print("\n🚀 开始训练...")
    best_test_acc = 0.0
    patience_counter = 0
    loss_history = []
    
    try:
        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            # 训练
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                args.grad_clip, epoch, args.epochs,
                args.gradient_accumulation_steps, rank
            )
            
            # 评估
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                if rank == 0:
                    print(f"\n📊 Epoch {epoch} 评估...")
                
                test_results = evaluate(model, test_loader, args.max_new_tokens, "Testing", rank)
                test_loss = test_results["loss"]
                test_acc = test_results["accuracy"]
                
                if rank == 0:
                    print(f"   Train Loss: {train_loss:.4f}")
                    print(f"   Test Loss: {test_loss:.4f}")
                    print(f"   Test Accuracy: {test_acc:.4f}")
                    
                    # 显示样本预测
                    print("   Sample predictions:")
                    for i in range(min(3, len(test_results["predictions"]))):
                        pred = test_results["predictions"][i]
                        label = test_results["labels"][i]
                        pred_short = pred[:30] if len(pred) > 30 else pred
                        print(f"     Pred: '{pred_short}' | Label: '{label}'")
                    
                    loss_history.append({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                    })
                    
                    with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
                        json.dump(loss_history, f, indent=2)
                
                # 保存最佳模型
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    patience_counter = 0
                    
                    if rank == 0:
                        checkpoint = {
                            "encoder_state": underlying_model.encoder.state_dict(),
                            "projector_state": underlying_model.projector.state_dict(),
                            "epoch": epoch,
                            "test_acc": best_test_acc,
                            "args": vars(args),
                        }
                        underlying_model.save_lora_state_to_checkpoint(checkpoint)
                        torch.save(checkpoint, os.path.join(save_dir, "best_model.pt"))
                        print(f"💾 Saved best model (test_acc={best_test_acc:.4f})")
                else:
                    patience_counter += 1
                    if rank == 0:
                        print(f"   (无改进, patience: {patience_counter}/{args.early_stop})")
                
                if patience_counter >= args.early_stop:
                    if rank == 0:
                        print(f"\n⏹️ 早停! 测试准确率 {args.early_stop} 轮未改进")
                    break
            else:
                if rank == 0:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
        
        # 最终结果
        if rank == 0:
            print("\n" + "=" * 60)
            print("📋 最终结果")
            print(f"   Best Test Accuracy: {best_test_acc:.4f}")
            
            final_results = {
                "dataset": args.dataset,
                "k_shot": args.k_shot,
                "top_m": args.top_m,
                "best_test_acc": best_test_acc,
                "epochs_trained": epoch,
            }
            
            with open(os.path.join(save_dir, "final_results.json"), "w") as f:
                json.dump(final_results, f, indent=2)
            
            print("=" * 60)
            print(f"结果保存到: {save_dir}")
            print("=" * 60)
    
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
