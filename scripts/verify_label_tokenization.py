"""
验证Excel风格标签（A, B, ..., Z, AA, AB, ...）的tokenization

检查这些标签是否被tokenizer编码为单个token，
这对于logits分类推理的正确性至关重要。
"""

import argparse
from transformers import AutoTokenizer


def index_to_excel_label(index: int) -> str:
    """将整数索引转换为类似Excel列名的字母标签。"""
    if index < 0:
        raise ValueError(f"Index must be non-negative, got {index}")
    
    if index < 26:
        return chr(ord('A') + index)
    else:
        adjusted = index - 26
        prefix_idx = adjusted // 26
        suffix_idx = adjusted % 26
        return chr(ord('A') + prefix_idx) + chr(ord('A') + suffix_idx)


def verify_labels_tokenization(tokenizer, max_labels: int = 100):
    """
    验证标签的tokenization。
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_labels: 要验证的最大标签数量
    
    Returns:
        tuple: (单token标签列表, 多token标签列表)
    """
    single_token_labels = []
    multi_token_labels = []
    
    print(f"\n{'='*60}")
    print(f"验证前 {max_labels} 个Excel风格标签的tokenization")
    print(f"Tokenizer: {tokenizer.name_or_path}")
    print(f"{'='*60}\n")
    
    for i in range(max_labels):
        label = index_to_excel_label(i)
        
        # 使用 add_special_tokens=False 避免添加 BOS/EOS
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        num_tokens = len(token_ids)
        
        if num_tokens == 1:
            single_token_labels.append((i, label, token_ids[0]))
        else:
            multi_token_labels.append((i, label, token_ids))
    
    # 打印结果
    print("✅ 单token标签:")
    print("-" * 40)
    for idx, label, token_id in single_token_labels:
        decoded = tokenizer.decode([token_id])
        print(f"  {idx:3d} -> {label:4s} -> token_id={token_id:5d} -> decoded='{decoded}'")
    
    if multi_token_labels:
        print(f"\n❌ 多token标签 (共 {len(multi_token_labels)} 个):")
        print("-" * 40)
        for idx, label, token_ids in multi_token_labels:
            tokens = [tokenizer.decode([tid]) for tid in token_ids]
            print(f"  {idx:3d} -> {label:4s} -> token_ids={token_ids} -> tokens={tokens}")
    
    # 统计
    print(f"\n{'='*60}")
    print(f"统计结果:")
    print(f"  - 单token标签数: {len(single_token_labels)}")
    print(f"  - 多token标签数: {len(multi_token_labels)}")
    print(f"  - 单token比例: {100*len(single_token_labels)/max_labels:.1f}%")
    print(f"{'='*60}")
    
    if multi_token_labels:
        print("\n⚠️  警告: 存在被编码为多token的标签!")
        print("   这可能会影响logits分类的准确性。")
        print("   建议:")
        print("   1. 只使用单token标签进行分类")
        print("   2. 或者将这些标签作为特殊token添加到tokenizer中")
    else:
        print("\n🎉 所有标签都是单token，可以安全使用!")
    
    return single_token_labels, multi_token_labels


def main():
    parser = argparse.ArgumentParser(description="验证Excel风格标签的tokenization")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace模型ID"
    )
    parser.add_argument(
        "--max_labels", 
        type=int, 
        default=100,
        help="要验证的最大标签数量 (默认100，覆盖A-Z和AA-BV)"
    )
    args = parser.parse_args()
    
    print(f"加载tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    
    single, multi = verify_labels_tokenization(tokenizer, args.max_labels)
    
    # 额外测试：验证一些特定的边界情况
    print("\n" + "="*60)
    print("边界情况测试:")
    print("="*60)
    
    test_labels = ["A", "Z", "AA", "AZ", "BA", "ZZ"]
    for label in test_labels:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        print(f"  '{label}' -> {token_ids} (num_tokens={len(token_ids)})")


if __name__ == "__main__":
    main()
