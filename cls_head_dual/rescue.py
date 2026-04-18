"""
BN 层拯救脚本 (针对双分支编码器)。

用途：如果预训练的双分支编码器的 BatchNorm 统计数据被污染（出现 NaN），
可以用这个脚本通过干净数据重新校准 BN 层的 running_mean 和 running_var。

原理：
    1. 检查卷积核是否被污染 → 如果是则无法拯救
    2. 清空 BN 层统计数据
    3. 在 train 模式下喂入干净数据重新计算 BN 统计数据
    4. 保存校正后的模型
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pandas as pd
import numpy as np

# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.CrossAttentionVec import DualBranchMCGEncoder
from cls_head_dual.finetune_model import DualBranchFineTuneDataset


def rescue_corrupted_bn():
    """拯救被污染的 BN 层"""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CORRUPTED_PATH = "../checkpoints/dual_branch_mcg_encoder_ep100.pth"
    RESCUED_PATH = "../checkpoints/dual_branch_mcg_encoder_ep100_RESCUED.pth"

    print(f"🚑 开始诊断双分支编码器模型: {CORRUPTED_PATH}")
    state_dict = torch.load(CORRUPTED_PATH, map_location=DEVICE)

    # 1. 深度体检：检查第一个卷积核
    # 双分支模型中，时空分支的第一个 Conv3d 权重
    first_conv_key = None
    for key in state_dict.keys():
        if 'weight' in key and ('conv' in key.lower() or 'st_branch.0' in key):
            first_conv_key = key
            break

    if first_conv_key:
        conv_weight = state_dict[first_conv_key]
        if torch.isnan(conv_weight).any():
            print(f"❌ 致命坏消息：卷积核 '{first_conv_key}' 包含 NaN。模型已彻底损坏，必须重新预训练。")
            return
        else:
            print(f"✅ 好消息：核心卷积核 '{first_conv_key}' 非常健康！仅 BN 层可能被污染。")
    else:
        print("⚠️ 未能定位卷积核，继续尝试修复...")

    # 2. 加载模型
    model = DualBranchMCGEncoder(in_channels=36, feature_dim=256).to(DEVICE)
    model.load_state_dict(state_dict)

    # 3. 清空所有被污染的 BN 层统计数据
    print("🧹 正在清空被污染的 BatchNorm 统计数据...")
    bn_count = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm3d)):
            module.reset_running_stats()
            module.momentum = 0.5  # 较大的动量加快均值收敛
            bn_count += 1
    print(f"   已重置 {bn_count} 个 BatchNorm 层")

    # 4. 准备干净的数据源
    df = pd.read_csv(r"E:\Pythonpro\MCG_quexue_xinshuai\label.csv")
    DATA_DIR = r"E:\Pythonpro\MCG_quexue_xinshuai\all_data_folder"
    clean_dataset = DualBranchFineTuneDataset(df, DATA_DIR, task="Ischemia")

    # 只抽取前 2000 个心拍来重新计算均值
    subset_indices = list(range(min(2000, len(clean_dataset))))
    rescue_loader = DataLoader(Subset(clean_dataset, subset_indices),
                               batch_size=32, shuffle=True)

    # 5. 执行"透析"：在 train 模式下前向传播
    print(f"🩸 正在使用干净数据进行 BN 重新校准...")
    model.train()

    with torch.no_grad():
        for x, _ in tqdm(rescue_loader, desc="重新校准 BN"):
            x = x.to(DEVICE)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            _ = model(x)

    # 6. 保存拯救成功的新模型
    torch.save(model.state_dict(), RESCUED_PATH)
    print("=" * 50)
    print(f"🎉 双分支编码器模型拯救成功！")
    print(f"👉 请在 train_finetune.py 中修改 ENCODER_WEIGHT_PATH 为:")
    print(f"   {RESCUED_PATH}")


if __name__ == "__main__":
    rescue_corrupted_bn()
