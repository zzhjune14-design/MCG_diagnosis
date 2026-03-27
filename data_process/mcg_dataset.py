import os
# 必须加在 import torch 之前！
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
# ... 其他引用
from torch.utils.data import Dataset
import numpy as np
import random
from tqdm import tqdm


# ==========================================
# 1. 信号增强器 (保持不变)
# ==========================================
class MCGAugmentations:
    """
    SimCLR 核心增强模块：
    1. 时间掩码 (Temporal Masking)
    2. 通道丢失 (Channel Dropout)
    3. 高斯噪声 (Gaussian Noise)
    """

    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, x):
        # x shape: [36, 700]
        aug_x = x.copy()

        if random.random() > self.prob:
            return torch.from_numpy(aug_x).float()

        # A. 时间掩码 (随机抹掉一段信号)
        if random.random() < 0.5:
            n_channels, seq_len = aug_x.shape
            mask_len = int(seq_len * random.uniform(0.1, 0.5))  # 遮挡 10%~50%
            start = random.randint(0, seq_len - mask_len)

            # 简单起见，对所有通道遮挡同一段时间(模拟瞬间干扰)
            # 或者你可以写循环对每个通道独立遮挡
            aug_x[:, start:start + mask_len] = 0

            # B. 通道丢失 (模拟传感器故障)
        if random.random() < 0.5:
            n_channels = aug_x.shape[0]
            # 随机丢弃 10% 的通道
            num_drop = max(1, int(n_channels * 0.1))
            drop_indices = random.sample(range(n_channels), num_drop)
            for ch in drop_indices:
                # 用高斯噪声填充，模拟坏道
                aug_x[ch, :] = np.random.normal(0, 0.1, size=aug_x.shape[1])

        # C. 高斯噪声 (模拟环境底噪)
        if random.random() < 0.5:
            noise_level = random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_level, size=aug_x.shape)
            aug_x += noise

        return torch.from_numpy(aug_x).float()


# ==========================================
# 2. 全量数据集 (升级版)
# ==========================================
class MCGContrastiveDataset(Dataset):
    def __init__(self, data_dirs):
        self.sample_index_map = []  # 仅存放索引：[(文件路径, 心拍所在行数), ...]
        self.augmentor = MCGAugmentations(prob=1.0)

        print(f"🚀 正在建立数据索引 (极低内存占用)...")
        for d in data_dirs:
            if not os.path.exists(d): continue

            files = [f for f in os.listdir(d) if f.endswith('.npy')]
            for fname in tqdm(files, desc=f"Indexing {os.path.basename(d)}"):
                path = os.path.join(d, fname)
                try:
                    # 关键修改：mmap_mode='r' 只读取元数据(shape)，不把几百MB数据载入内存
                    data_mmap = np.load(path, mmap_mode='r')
                    num_beats = data_mmap.shape[0]
                    for i in range(num_beats):
                        self.sample_index_map.append((path, i))
                except Exception as e:
                    print(f"跳过坏文件 {fname}: {e}")

        print(f"✅ 索引建立完毕！总心拍数: {len(self.sample_index_map)}")

    def __len__(self):
        return len(self.sample_index_map)

    def __getitem__(self, idx):
        # 1. 查表找到对应的文件路径和局部索引
        file_path, beat_idx = self.sample_index_map[idx]

        # 2. 局部极速读取：仅把当前需要的 [36, 700] 读进内存
        data_mmap = np.load(file_path, mmap_mode='r')
        x = np.array(data_mmap[beat_idx])

        # 3. 增强
        x_i = self.augmentor(x)
        x_j = self.augmentor(x)

        return x_i, x_j


# ==========================================
# 3. 验证代码
# ==========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 修改为你的路径
    HEALTHY_DIR = r'E:\Pythonpro\MCG_quexue_xinshuai\heartbeat_segments\healthy'
    SICK_DIR = r'E:\Pythonpro\MCG_quexue_xinshuai\heartbeat_segments\sick'

    # 初始化
    dataset = MCGContrastiveDataset([HEALTHY_DIR, SICK_DIR])

    # 打印总样本数
    print(f"Dataset length: {len(dataset)}")

    # 取一个样本看看
    x1, x2 = dataset[100]  # 取第100个心拍

    # 画图验证增强效果
    plt.figure(figsize=(10, 4))
    plt.plot(x1[0].numpy(), label='Augmentation 1')
    plt.plot(x2[0].numpy(), label='Augmentation 2', alpha=0.7)
    plt.title("SimCLR Data Augmentation Check (Channel 0)")
    plt.legend()
    plt.show()