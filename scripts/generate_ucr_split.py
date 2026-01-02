# UCR数据集随机划分脚本
# 生成98个训练数据集和30个测试数据集的划分列表

import os
import random

# 设置随机种子确保可重复
random.seed(42)

# UCR数据集目录
ucr_dir = r"C:\Users\QYH\Downloads\tsllava\data\UCRArchive_2018"

# 获取所有数据集名称（排除元数据目录）
all_datasets = []
for name in os.listdir(ucr_dir):
    full_path = os.path.join(ucr_dir, name)
    if os.path.isdir(full_path):
        # 排除元数据目录
        if name != "Missing_value_and_variable_length_datasets_adjusted":
            all_datasets.append(name)

print(f"共发现 {len(all_datasets)} 个数据集")

# 随机打乱
random.shuffle(all_datasets)

# 划分: 98个训练, 30个测试
train_datasets = sorted(all_datasets[:98])
test_datasets = sorted(all_datasets[98:])

print(f"训练集: {len(train_datasets)} 个")
print(f"测试集: {len(test_datasets)} 个")

# 保存训练集列表
train_file = r"C:\Users\QYH\Downloads\tsllava\data\ucr_train_98_datasets.txt"
with open(train_file, 'w') as f:
    f.write("# UCR训练数据集列表 (98个)\n")
    f.write("# 用于TSLANet预训练\n")
    f.write("# 随机种子: 42\n")
    f.write("#\n")
    for name in train_datasets:
        f.write(name + "\n")
print(f"训练集列表已保存到: {train_file}")

# 保存测试集列表
test_file = r"C:\Users\QYH\Downloads\tsllava\data\ucr_test_30_datasets.txt"
with open(test_file, 'w') as f:
    f.write("# UCR测试数据集列表 (30个)\n")
    f.write("# 用于跨域能力测试\n")
    f.write("# 随机种子: 42\n")
    f.write("#\n")
    for name in test_datasets:
        f.write(name + "\n")
print(f"测试集列表已保存到: {test_file}")

print("\n=== 训练集 ===")
for i, name in enumerate(train_datasets, 1):
    print(f"{i:3d}. {name}")

print("\n=== 测试集 ===")
for i, name in enumerate(test_datasets, 1):
    print(f"{i:3d}. {name}")
