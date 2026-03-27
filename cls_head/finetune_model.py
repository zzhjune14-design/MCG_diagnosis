import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# 导入你第二步定义的 MCGEncoder
from models.MCG2Vec import MCGEncoder


class MCGFineTuneDataset(Dataset):
    def __init__(self, df, data_dir, task="Ischemia"):
        """
        df: 包含 subject 和标签的 DataFrame
        data_dir: 存放 .npy 信号文件的文件夹路径
        task: "Ischemia" (缺血) 或 "xinshuai" (心衰)
        """
        self.sample_index_map = []
        self.labels = []
        self.task = task

        print(f"🚀 正在为 {task} 任务建立患者级索引...")

        for _, row in df.iterrows():
            subject_id = int(row['subject'])
            label_val = row[task]

            # 如果是心衰任务，跳过空值(NaN)患者
            if pd.isna(label_val):
                continue

            # 标签转换: 心衰中的 2.0 (非心衰) 转为 0，1.0 (心衰) 保持 1
            if task == "xinshuai":
                binary_label = 0.0 if label_val == 2.0 else 1.0
            else:
                binary_label = float(label_val)  # Ischemia 原本就是 0 或 1

            # 假设你的文件名为 subject_1.npy, subject_2.npy 等
            # 你需要根据实际的文件命名规则修改这里
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

    def __len__(self):
        return len(self.sample_index_map)

    def __getitem__(self, idx):
        fpath, beat_idx = self.sample_index_map[idx]
        data_mmap = np.load(fpath, mmap_mode='r')

        x = np.array(data_mmap[beat_idx])
        x_tensor = torch.from_numpy(x).float()
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)

        return x_tensor, label_tensor


class MCGFineTuneModel(nn.Module):
    def __init__(self, encoder_weight_path=None, in_channels=36, feature_dim=256):
        super(MCGFineTuneModel, self).__init__()

        self.encoder = MCGEncoder(in_channels=in_channels, feature_dim=feature_dim)

        # 针对心衰任务：加入了 BN 稳压，并调大了 Dropout (0.4) 防止小样本过拟合
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 1)  # 输出 Logits
        )

        # ==========================================
        # 🚀 智能权重加载：兼容无监督预训练与有监督接力微调
        # ==========================================
        if encoder_weight_path:
            state_dict = torch.load(encoder_weight_path, map_location='cpu')

            # 判断是否为完整微调模型 (包含 encoder. 和 classifier. 前缀)
            if any(k.startswith('encoder.') for k in state_dict.keys()):
                print(f"🔄 检测到接力微调权重，正在剥离并提取 Encoder 特征...")
                # 剥离前缀：只保留以 encoder. 开头的键，并去掉 "encoder." 这个字符串
                encoder_state = {k.replace('encoder.', ''): v
                                 for k, v in state_dict.items() if k.startswith('encoder.')}
                self.encoder.load_state_dict(encoder_state)
            else:
                print(f"🔄 检测到无监督预训练权重，直接加载 Encoder...")
                self.encoder.load_state_dict(state_dict)

            print(f"✅ 成功加载特征提取器权重！")

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits.squeeze(-1)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True