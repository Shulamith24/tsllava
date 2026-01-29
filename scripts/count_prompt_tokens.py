#!/usr/bin/env python3
"""
计算实验 A 中使用的 prompt token 数量

用途：
1. 分析不同数据集的 prompt token 数量
2. 为实验 B 提供合适的 num_prefix_tokens 参数
3. 验证 token 数量对性能的影响

使用方法：
    python scripts/count_prompt_tokens.py --dataset Adiac
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import argparse
from transformers import AutoTokenizer
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import UCRClassificationDataset


def parse_args():
    parser = argparse.ArgumentParser(description="计算实验 A 的 prompt token 数量")
    parser.add_argument("--dataset", type=str, default="Adiac", help="UCR数据集名称")
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B", help="LLM模型ID")
    parser.add_argument("--num_samples", type=int, default=10, help="采样样本数量")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print(f"计算数据集 {args.dataset} 的 Prompt Token 数量")
    print("=" * 60)
    
    # 加载 tokenizer
    print(f"\n加载 tokenizer: {args.llm_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    print(f"\n加载数据集: {args.dataset}")
    try:
        dataset = UCRClassificationDataset(
            split="train",
            EOS_TOKEN=tokenizer.eos_token,
            dataset_name=args.dataset,
            raw_data_path="./data",
        )
        print(f"✓ 数据集加载成功")
        print(f"  - 数据集大小: {len(dataset)}")
        print(f"  - 类别数: {dataset.get_num_classes()}")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return 1
    
    # 采样并计算 token 数量
    print(f"\n采样 {args.num_samples} 个样本...")
    num_samples = min(args.num_samples, len(dataset))
    
    pre_prompt_tokens = []
    post_prompt_tokens = []
    ts_text_tokens = []
    total_tokens = []
    
    for i in range(num_samples):
        sample = dataset[i]
        
        # Tokenize pre_prompt
        pre_tok = tokenizer(sample["pre_prompt"], return_tensors="pt")
        pre_len = pre_tok.attention_mask.sum().item()
        pre_prompt_tokens.append(pre_len)
        
        # Tokenize post_prompt
        post_tok = tokenizer(sample["post_prompt"], return_tensors="pt")
        post_len = post_tok.attention_mask.sum().item()
        post_prompt_tokens.append(post_len)
        
        # Tokenize time_series_text
        ts_text_len = 0
        for ts_text in sample["time_series_text"]:
            ts_tok = tokenizer(ts_text, return_tensors="pt")
            ts_text_len += ts_tok.attention_mask.sum().item()
        ts_text_tokens.append(ts_text_len)
        
        # Total prompt tokens (不包括 TS embeddings)
        total = pre_len + post_len + ts_text_len
        total_tokens.append(total)
    
    # 统计
    print("\n" + "=" * 60)
    print("Token 统计结果")
    print("=" * 60)
    
    avg_pre = sum(pre_prompt_tokens) / len(pre_prompt_tokens)
    avg_post = sum(post_prompt_tokens) / len(post_prompt_tokens)
    avg_ts_text = sum(ts_text_tokens) / len(ts_text_tokens)
    avg_total = sum(total_tokens) / len(total_tokens)
    
    print(f"\nPre-Prompt:")
    print(f"  - 平均长度: {avg_pre:.1f} tokens")
    print(f"  - 范围: {min(pre_prompt_tokens)} - {max(pre_prompt_tokens)}")
    
    print(f"\nPost-Prompt:")
    print(f"  - 平均长度: {avg_post:.1f} tokens")
    print(f"  - 范围: {min(post_prompt_tokens)} - {max(post_prompt_tokens)}")
    
    print(f"\nTime Series Text:")
    print(f"  - 平均长度: {avg_ts_text:.1f} tokens")
    print(f"  - 范围: {min(ts_text_tokens)} - {max(ts_text_tokens)}")
    
    print(f"\n总 Prompt Tokens (不含 TS embeddings):")
    print(f"  - 平均长度: {avg_total:.1f} tokens")
    print(f"  - 范围: {min(total_tokens)} - {max(total_tokens)}")
    
    # 建议
    print("\n" + "=" * 60)
    print("建议的 num_prefix_tokens 值")
    print("=" * 60)
    
    # 四舍五入到最近的 8 的倍数
    suggested = int(round(avg_total / 8) * 8)
    print(f"\n推荐值: {suggested} (最接近的 8 的倍数)")
    print(f"精确值: {int(avg_total)}")
    
    # 提供运行命令
    print("\n" + "=" * 60)
    print("运行实验 B（使用相同数量的随机初始化 tokens）")
    print("=" * 60)
    print(f"""
python scripts/train_ucr_classification_experiment_b.py \\
    --dataset {args.dataset} \\
    --num_prefix_tokens {suggested} \\
    --epochs 30 \\
    --batch_size 4 \\
    --use_lora
""")
    
    # 显示示例文本
    print("\n" + "=" * 60)
    print("示例 Prompt 文本")
    print("=" * 60)
    sample = dataset[0]
    print(f"\nPre-Prompt ({pre_prompt_tokens[0]} tokens):")
    print(f"  {sample['pre_prompt'][:200]}...")
    print(f"\nPost-Prompt ({post_prompt_tokens[0]} tokens):")
    print(f"  {sample['post_prompt']}")
    print(f"\nTime Series Text ({ts_text_tokens[0]} tokens):")
    for i, ts_text in enumerate(sample['time_series_text']):
        print(f"  [{i}] {ts_text[:100]}...")
    
    return 0


if __name__ == "__main__":
    exit(main())
