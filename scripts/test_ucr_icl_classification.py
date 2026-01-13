#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
UCR ICL分类测试脚本

加载训练好的模型权重，在任意UCR数据集的测试集上进行评估。

使用方法：
    python scripts/test_ucr_icl_classification.py \\
        --dataset ECG5000 \\
        --pretrained_model OpenTSLM/llama-3.2-1b-m4-sp \\
        --tslanet_checkpoint results/tslanet_ucr/ECG5000/best_model.pt \\
        --icl_checkpoint results/icl_classification/ECG5000/best_model.pt \\
        --k_shot 1

可选：跨域测试（用A数据集的TSLANet检索器在B数据集上测试）：
    python scripts/test_ucr_icl_classification.py \\
        --dataset Wafer \\
        --tslanet_checkpoint results/tslanet_ucr/ECG5000/best_model.pt \\
        --icl_checkpoint results/icl_classification/ECG5000/best_model.pt \\
        --pretrained_model OpenTSLM/llama-3.2-1b-m4-sp \\
        --k_shot 1
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
    parser = argparse.ArgumentParser(description="UCR ICL分类测试")

    # 数据相关
    parser.add_argument("--dataset", type=str, required=True, help="要测试的UCR数据集名称")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR数据根目录")
    
    # 模型相关 - OpenTSLM
    parser.add_argument("--pretrained_model", type=str, default=None, 
                        help="预训练模型ID (HuggingFace repo_id)")
    parser.add_argument("--icl_checkpoint", type=str, default=None,
                        help="ICL训练后的checkpoint路径 (可选，用于加载fine-tuned权重)")
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
    
    # 测试相关
    parser.add_argument("--batch_size", type=int, default=8, help="测试批次大小")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="生成最大token数")
    
    # 保存相关
    parser.add_argument("--save_dir", type=str, default="results/icl_test", help="结果保存目录")
    parser.add_argument("--save_predictions", action="store_true", help="保存详细预测结果")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    
    return parser.parse_args()


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
        pos_embed_shape = encoder_state["pos_embed"].shape
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


def create_test_dataset(args, retriever, eos_token: str):
    """创建测试Dataset"""
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
    print(f"   Train samples (用于检索): {len(train_ts)}")
    print(f"   Test samples: {len(test_ts)}")
    
    # 构建检索索引 (用训练集)
    print("\n🔧 构建检索索引...")
    retriever.build_index(train_ts, train_labels)
    
    # 创建测试Dataset
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
    
    return test_dataset, len(all_labels)


@torch.no_grad()
def evaluate(
    model,
    data_loader: DataLoader,
    max_new_tokens: int,
    desc: str = "Testing",
) -> Dict[str, Any]:
    """评估模型"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_details = []
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(data_loader, desc=desc):
        # 计算loss
        loss = model.compute_loss(batch)
        total_loss += loss.item()
        num_batches += 1
        
        # 生成预测
        predictions = model.generate(batch, max_new_tokens=max_new_tokens)
        
        for sample, pred in zip(batch, predictions):
            label = sample.get("letter_label", "")
            all_predictions.append(pred)
            all_labels.append(label)
            
            # 保存详细信息
            all_details.append({
                "prediction": pred,
                "label": label,
                "query_idx": sample.get("query_idx", -1),
                "support_labels": sample.get("support_labels", []),
            })
    
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = calculate_accuracy(all_predictions, all_labels)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
        "details": all_details,
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("UCR ICL分类测试")
    print("=" * 60)
    print(f"时间: {datetime.datetime.now()}")
    print(f"数据集: {args.dataset}")
    print(f"K-shot: {args.k_shot}")
    print(f"Top-M: {args.top_m}")
    print("=" * 60)
    
    set_seed(args.seed)
    
    # 设置设备
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        print("⚠️ CUDA不可用，使用CPU")
        device = "cpu"
    
    # 创建保存目录
    save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(save_dir, "test_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # 加载TSLANet用于检索
    print("\n🔧 加载TSLANet检索器...")
    tslanet_encoder, tslanet_ckpt = load_tslanet_for_retrieval(args.tslanet_checkpoint, device)
    retriever = TSLANetRetriever(tslanet_encoder, device=device)
    
    # 加载OpenTSLM模型
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
    else:
        raise ValueError("必须指定 --pretrained_model")
    
    # 加载ICL训练后的权重（如果提供）
    if args.icl_checkpoint:
        print(f"\n📂 加载ICL checkpoint: {args.icl_checkpoint}")
        icl_ckpt = torch.load(args.icl_checkpoint, map_location=device, weights_only=False)
        
        # 加载encoder和projector权重
        model.encoder.load_state_dict(icl_ckpt["encoder_state"])
        model.projector.load_state_dict(icl_ckpt["projector_state"])
        
        # 加载LoRA权重
        if use_lora and "lora_state" in icl_ckpt:
            model.load_lora_state_from_checkpoint(icl_ckpt, allow_missing=True)
        
        print(f"   Epoch: {icl_ckpt.get('epoch', 'unknown')}")
        print(f"   Test Acc (训练时): {icl_ckpt.get('test_acc', 'unknown')}")
    
    # 创建测试数据集
    print("\n📂 创建测试数据集...")
    eos_token = model.get_eos_token()
    test_dataset, num_classes = create_test_dataset(args, retriever, eos_token)
    
    # 创建DataLoader
    collate_fn = create_icl_collate_fn(patch_size=PATCH_SIZE)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    print(f"   Test batches: {len(test_loader)}")
    
    # 测试
    print("\n🚀 开始测试...")
    test_results = evaluate(model, test_loader, args.max_new_tokens)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("📋 测试结果")
    print("=" * 60)
    print(f"   Dataset: {args.dataset}")
    print(f"   Test Loss: {test_results['loss']:.4f}")
    print(f"   Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"   Total Samples: {len(test_results['predictions'])}")
    
    # 显示样本预测
    print("\n   Sample predictions:")
    for i in range(min(5, len(test_results["predictions"]))):
        pred = test_results["predictions"][i]
        label = test_results["labels"][i]
        pred_short = pred[:40] if len(pred) > 40 else pred
        status = "✓" if pred.strip().upper() == label.strip().upper() else "✗"
        print(f"     [{status}] Pred: '{pred_short}' | Label: '{label}'")
    
    # 保存结果
    final_results = {
        "dataset": args.dataset,
        "k_shot": args.k_shot,
        "top_m": args.top_m,
        "test_loss": test_results["loss"],
        "test_accuracy": test_results["accuracy"],
        "num_samples": len(test_results["predictions"]),
        "num_classes": num_classes,
        "tslanet_checkpoint": args.tslanet_checkpoint,
        "icl_checkpoint": args.icl_checkpoint,
        "timestamp": str(datetime.datetime.now()),
    }
    
    with open(os.path.join(save_dir, "test_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    
    # 保存详细预测（可选）
    if args.save_predictions:
        with open(os.path.join(save_dir, "predictions.json"), "w") as f:
            json.dump(test_results["details"], f, indent=2)
        print(f"\n💾 详细预测已保存到: {os.path.join(save_dir, 'predictions.json')}")
    
    print("=" * 60)
    print(f"结果保存到: {save_dir}")
    print("=" * 60)
    
    return test_results["accuracy"]


if __name__ == "__main__":
    main()
