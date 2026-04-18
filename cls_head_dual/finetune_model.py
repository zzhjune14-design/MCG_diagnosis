import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# 导入双分支编码器
from models.CrossAttentionVec import DualBranchMCGEncoder


class DualBranchFineTuneDataset(Dataset):
    """
    面向双分支编码器的微调数据集。
    数据格式：每个 .npy 文件对应一个患者，形状为 [num_beats, 36, 700]。
    标签来自 CSV 文件。
    """

    def __init__(self, df, data_dir, task="Ischemia"):
        """
        df: 包含 subject 和标签的 DataFrame
        data_dir: 存放 .npy 信号文件的文件夹路径
        task: "Ischemia" (缺血) 或 "xinshuai" (心衰)
        """
        self.sample_index_map = []
        self.labels = []
        self.task = task

        print(f"🚀 正在为 {task} 任务建立患者级索引 (双分支架构)...")

        for _, row in df.iterrows():
            subject_id = int(row['subject'])
            label_val = row[task]

            # 跳过空值(NaN)患者
            if pd.isna(label_val):
                continue

            # 标签转换: 心衰中的 2.0 (非心衰) 转为 0，1.0 (心衰) 保持 1
            if task == "xinshuai":
                binary_label = 0.0 if label_val == 2.0 else 1.0
            else:
                binary_label = float(label_val)  # Ischemia 原本就是 0 或 1

            # 文件名规则：{subject_id}.npy
            fname = f"{subject_id}.npy"
            fpath = os.path.join(data_dir, fname)

            if not os.path.exists(fpath):
                continue

            # 使用 mmap_mode 读取并拆解该病人的所有心拍
            data_mmap = np.load(fpath, mmap_mode='r')
            num_beats = data_mmap.shape[0]
            for i in range(num_beats):
                self.sample_index_map.append((fpath, i))
                self.labels.append(binary_label)

        # 打印类别分布
        pos_count = sum(1 for l in self.labels if l == 1.0)
        neg_count = len(self.labels) - pos_count
        print(f"✅ 数据集构建完毕: 总样本 {len(self.labels)}, "
              f"正类 {pos_count}, 负类 {neg_count}, "
              f"正类比例 {pos_count / max(len(self.labels), 1):.2%}")

    def __len__(self):
        return len(self.sample_index_map)

    def __getitem__(self, idx):
        fpath, beat_idx = self.sample_index_map[idx]
        data_mmap = np.load(fpath, mmap_mode='r')

        x = np.array(data_mmap[beat_idx])  # 形状: [36, 700]
        x_tensor = torch.from_numpy(x).float()
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)

        return x_tensor, label_tensor


class DualBranchFineTuneModel(nn.Module):
    """
    基于双分支编码器 (DualBranchMCGEncoder) 的微调分类模型。

    架构：
        DualBranchMCGEncoder (时空+频空+交叉注意力) -> 分类头

    特性：
        - 支持从纯 encoder 预训练权重加载（SimCLR 预训练产物）
        - 支持从完整微调模型接力加载（例如先训缺血再训心衰）
        - 冻结/解冻 encoder 方便两阶段训练
    """

    def __init__(self, encoder_weight_path=None, in_channels=36, feature_dim=256):
        super(DualBranchFineTuneModel, self).__init__()

        # 骨干网络：双分支编码器
        self.encoder = DualBranchMCGEncoder(in_channels=in_channels, feature_dim=feature_dim)

        # 分类头：BN 稳压 + Dropout 防过拟合 + 两层 MLP
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 1)  # 输出 Logits (BCEWithLogitsLoss)
        )

        # ==========================================
        # 🚀 智能权重加载：兼容无监督预训练与有监督接力微调
        # ==========================================
        if encoder_weight_path:
            state_dict = torch.load(encoder_weight_path, map_location='cpu')

            # 判断是否为完整微调模型 (包含 encoder. 和 classifier. 前缀)
            if any(k.startswith('encoder.') for k in state_dict.keys()):
                print(f"🔄 检测到接力微调权重，正在剥离并提取 Encoder 特征...")
                # 剥离前缀：只保留以 encoder. 开头的键，并去掉 "encoder." 字符串
                encoder_state = {k.replace('encoder.', ''): v
                                 for k, v in state_dict.items() if k.startswith('encoder.')}
                self.encoder.load_state_dict(encoder_state)
            else:
                print(f"🔄 检测到无监督预训练权重 (纯 Encoder state_dict)，直接加载...")
                self.encoder.load_state_dict(state_dict)

            print(f"✅ 成功加载双分支特征提取器权重！")

    def forward(self, x):
        """
        x: [B, 36, 700] — 36通道心磁信号
        返回: [B] — logits (使用 BCEWithLogitsLoss)
        """
        features = self.encoder(x)       # [B, 256]
        logits = self.classifier(features)  # [B, 1]
        return logits.squeeze(-1)

    def freeze_encoder(self):
        """冻结编码器参数 (用于 Warmup 阶段)"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("🔒 Encoder 已冻结 (仅训练分类头)")

    def unfreeze_encoder(self):
        """解冻编码器参数 (用于联合微调阶段)"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("🔓 Encoder 已解冻 (全量微调)")
