import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pandas as pd
import numpy as np

# 导入你的网络和数据集
from models.MCG2Vec import MCGEncoder
from cls_head.finetune_model import MCGFineTuneDataset


def rescue_corrupted_bn():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CORRUPTED_PATH = "../checkpoints/mcg2vec_encoder_ep100.pth"
    RESCUED_PATH = "../checkpoints/mcg2vec_encoder_ep100_RESCUED.pth"

    print(f"🚑 开始诊断模型: {CORRUPTED_PATH}")
    state_dict = torch.load(CORRUPTED_PATH, map_location=DEVICE)

    # 1. 深度体检：检查核心卷积核是否被污染
    conv_weight = state_dict['conv_blocks.0.weight']
    if torch.isnan(conv_weight).any():
        print("❌ 致命坏消息：卷积核权重也包含 NaN。拦截失败，模型已彻底损坏，您必须重新进行 100 轮预训练。")
        return
    else:
        print("✅ 好消息：核心卷积核非常健康！仅仅是 BN 层被污染，可以被拯救！")

    # 2. 加载模型
    model = MCGEncoder(in_channels=36, feature_dim=256).to(DEVICE)
    model.load_state_dict(state_dict)

    # 3. 清空所有被污染的 BN 层统计数据
    print("🧹 正在清空被污染的 BatchNorm 统计数据...")
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.reset_running_stats()
            # 设为一个较大的动量，加快均值重新收敛的速度
            module.momentum = 0.5

            # 4. 准备干净的数据源 (复用你的微调 Dataset)
    df = pd.read_csv("E:\Pythonpro\MCG_quexue_xinshuai\label.csv")
    DATA_DIR = r"E:\Pythonpro\MCG_quexue_xinshuai\all_data_folder"
    # 只取一部分数据来校准均值即可，不需要跑全量
    clean_dataset = MCGFineTuneDataset(df, DATA_DIR, task="Ischemia")
    # 我们只抽取前 2000 个心拍来重新计算均值，速度极快
    subset_indices = list(range(min(2000, len(clean_dataset))))
    rescue_loader = DataLoader(Subset(clean_dataset, subset_indices), batch_size=128, shuffle=True)

    # 5. 执行“透析”：在 train 模式下重新前向传播，让模型重新看到干净数据并计算新的 BN 均值
    print(f"🩸 正在使用干净数据进行 BN 重新校准 (透析)...")
    model.train()  # 必须在 train 模式下 BN 才会更新

    with torch.no_grad():  # 不需要反向传播，不需要算梯度！
        for x, _ in tqdm(rescue_loader, desc="Recalibrating BN"):
            x = x.to(DEVICE)
            # 加强防线，确保传入的数据绝对干净
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            _ = model(x)

    # 6. 保存拯救成功的新模型
    torch.save(model.state_dict(), RESCUED_PATH)
    print("=" * 50)
    print(f"🎉 模型拯救成功！")
    print(f"👉 请在 train_finetune.py 中使用新的权重路径: \n{RESCUED_PATH}")


if __name__ == "__main__":
    rescue_corrupted_bn()