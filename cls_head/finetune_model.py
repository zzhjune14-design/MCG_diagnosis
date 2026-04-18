import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# 导入你第二步定义的 MCGEncoder
from models.MCG2Vec import MCGEncoder


class MCGFineTuneDataset(Dataset):
    def __init__(self, df, data_dir, task="Ischemia", return_subject_id=False):
        """
        df: 包含 subject 和标签的 DataFrame
        data_dir: 存放 .npy 信号文件的文件夹路径
        task: "Ischemia" (缺血) 或 "xinshuai" (心衰)
        """
        self.sample_index_map = []
        self.labels = []
        self.task = task
        self.return_subject_id = return_subject_id
        self.subject_ids = []

        print(f"Building patient-level index for task={task}...")

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
                self.subject_ids.append(subject_id)

    def __len__(self):
        return len(self.sample_index_map)

    def __getitem__(self, idx):
        fpath, beat_idx = self.sample_index_map[idx]
        data_mmap = np.load(fpath, mmap_mode='r')

        x = np.array(data_mmap[beat_idx])
        x_tensor = torch.from_numpy(x).float()
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.return_subject_id:
            subject_tensor = torch.tensor(self.subject_ids[idx], dtype=torch.long)
            return x_tensor, label_tensor, subject_tensor

        return x_tensor, label_tensor


class MCGFineTuneModel(nn.Module):
    def __init__(self, encoder_weight_path=None, in_channels=36, feature_dim=256):
        super(MCGFineTuneModel, self).__init__()

        self.encoder = MCGEncoder(in_channels=in_channels, feature_dim=feature_dim)

        # ==================================
        # 创新架构: Gated Residual Network (门控残差网络) 
        # 它能有效地在小样本上做高级非线性特征交叉与选择
        # ==================================
        hidden_dim = 128
        
        # 降维与升维
        self.fc_in = nn.Linear(feature_dim, hidden_dim)
        self.fc_act = nn.ELU()
        
        # 两个特征交叉路径
        self.fc_gate1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_gate2 = nn.Linear(hidden_dim, hidden_dim)

        self.bn_res = nn.BatchNorm1d(hidden_dim)
        # ==========================================
        # 🚀 创新架构：基于 Transformer Encoder 的时序注意力机制
        # 绕过原 encoder 的 Global Avg Pool，处理时间维度 44 的序列 [B, 256, 44]
        # ==========================================
        
        # 1. 引入 1 层 Transformer Encoder 挖掘沿着时间轴各特征通道的相关性
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8, # 多头注意力同时捕捉不同维度的心电特征
            dim_feedforward=512,
            dropout=0.3,
            activation='gelu',
            batch_first=True # 处理形状为 [B, T, feature_dim] 的数据
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 2. 从注意力输出后的一维池化 + 健壮的分类头
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        if encoder_weight_path:
            state_dict = torch.load(encoder_weight_path, map_location='cpu')
            if any(k.startswith('encoder.') for k in state_dict.keys()):
                print("Detected checkpoint with encoder prefix, stripping prefix before load...")
                encoder_state = {k.replace('encoder.', ''): v
                                 for k, v in state_dict.items() if k.startswith('encoder.')}
                self.encoder.load_state_dict(encoder_state)
            else:
                self.encoder.load_state_dict(state_dict)

    def forward(self, x):
        # 1. 截取 Backbone 输出：[B, 36, 700] -> [B, 256, 44]
        # 跳过了 self.encoder.global_pool
        features = self.encoder.conv_blocks(x)
        
        # 2. 形状转换以适配 Transformer: [B, 256, 44] -> [B, 44, 256] 
        features = features.permute(0, 2, 1)
        
        # 3. 走自注意力网络挖掘时间维度前后的因果关联 -> [B, 44, 256]
        attn_out = self.transformer(features)
        
        # 4. 全局平均池化 (聚合时间步，将含有注意力的时间步聚合成一个特征向量)
        # [B, 44, 256] -> [B, 256, 44] -> [B, 256, 1] -> [B, 256]
        pooled_out = attn_out.permute(0, 2, 1).mean(dim=2)
        
        # 5. 分类输出
        logits = self.classifier(pooled_out)
        return logits.squeeze(-1)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


# ==========================================
# 🚀 新增模块：机器学习波形特征提取器
# ==========================================
class WaveformFeatureExtractor(nn.Module):
    """
    通过 PyTorch 张量操作直接提取波形的传统机器学习时域/统计特征
    输入 shape: [B, C, L] (Batch, 36通道, 700长度)
    """
    def __init__(self):
        super(WaveformFeatureExtractor, self).__init__()

    def forward(self, x):
        # 1. 均值、标准差、极值
        mean_feat = torch.mean(x, dim=2)          # [B, 36]
        std_feat = torch.std(x, dim=2)            # [B, 36]
        max_feat = torch.max(x, dim=2).values     # [B, 36]
        min_feat = torch.min(x, dim=2).values     # [B, 36]
        
        # 2. 峰峰值 (Peak-to-Peak) 和 均方根 (RMS)
        ptp_feat = max_feat - min_feat            # [B, 36]
        rms_feat = torch.sqrt(torch.mean(x**2, dim=2) + 1e-8) # [B, 36]
        
        # 3. 差分特征 (反映波形的剧烈跳变和抖动，类似复杂度)
        diff_x = torch.diff(x, dim=2)
        diff_abs_sum = torch.sum(torch.abs(diff_x), dim=2)    # 反映波形总变化距离
        diff_std = torch.std(diff_x, dim=2)                   # 变化率的标准差

        # 4. 能量 (Energy)
        energy_feat = torch.sum(x**2, dim=2)
        
        # 拼接 9 种特征，最后得到的是 [B, 36 * 9]
        features = torch.cat([
            mean_feat, std_feat, max_feat, min_feat,
            ptp_feat, rms_feat, diff_abs_sum, diff_std, energy_feat
        ], dim=1)
        
        return features


class SpectralFeatureExtractor(nn.Module):
    def __init__(self):
        super(SpectralFeatureExtractor, self).__init__()

    def forward(self, x):
        spectrum = torch.fft.rfft(x, dim=2)
        magnitude = torch.abs(spectrum)
        power = magnitude ** 2

        mean_mag = magnitude.mean(dim=2)
        std_mag = magnitude.std(dim=2)
        max_mag = magnitude.max(dim=2).values

        num_bins = magnitude.shape[2]
        low_end = max(1, num_bins // 8)
        mid_end = max(low_end + 1, num_bins // 3)

        low_band = power[:, :, :low_end].sum(dim=2)
        mid_band = power[:, :, low_end:mid_end].sum(dim=2)
        high_band = power[:, :, mid_end:].sum(dim=2)

        freq_axis = torch.linspace(0.0, 1.0, steps=num_bins, device=x.device, dtype=x.dtype).view(1, 1, -1)
        power_sum = power.sum(dim=2) + 1e-8
        spectral_centroid = (power * freq_axis).sum(dim=2) / power_sum
        spectral_bandwidth = torch.sqrt(
            ((freq_axis - spectral_centroid.unsqueeze(-1)) ** 2 * power).sum(dim=2) / power_sum + 1e-8
        )

        return torch.cat(
            [
                mean_mag,
                std_mag,
                max_mag,
                low_band,
                mid_band,
                high_band,
                spectral_centroid,
                spectral_bandwidth,
            ],
            dim=1,
        )


# ==========================================
# 方案一：纯传统特征 + 深度学全连接网络(MLP)
# ==========================================
class MLPWaveformModel(nn.Module):
    """
    抛弃了 CNN 和预训练阶段，完全只使用 9种传统波形特征 送入 MLP中进行的实验。
    """
    def __init__(self, in_channels=36, num_handcrafted_features=9):
        super(MLPWaveformModel, self).__init__()
        self.extractor = WaveformFeatureExtractor()
        
        total_feats = in_channels * num_handcrafted_features # 36 * 9 = 324维
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(total_feats),
            nn.Linear(total_feats, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # 提取统计学特征
        feats = self.extractor(x)
        # 用分类头进行预测
        logits = self.classifier(feats)
        return logits.squeeze(-1)
        
    def freeze_encoder(self):
        # 为了兼容训练代码的逻辑占位
        pass

    def unfreeze_encoder(self):
        # 为了兼容训练代码的逻辑占位
        pass


class MLPSpectralModel(nn.Module):
    def __init__(self, in_channels=36, num_spectral_features=8):
        super(MLPSpectralModel, self).__init__()
        self.extractor = SpectralFeatureExtractor()

        total_feats = in_channels * num_spectral_features
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(total_feats),
            nn.Linear(total_feats, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        feats = self.extractor(x)
        logits = self.classifier(feats)
        return logits.squeeze(-1)

    def freeze_encoder(self):
        pass

    def unfreeze_encoder(self):
        pass


# =========================================================
# 【你的创新点】方案二升级版：基于临床先验门控特征融合的深度网络
# (Cross-Modal Gating Feature Fusion Network)
# =========================================================
class HybridFineTuneModel(nn.Module):
    """
    创新融合架构：
    不再是粗暴拼接，而是用人工提取的“物理/临床指标(324维)”作为先验，
    去生成对黑盒深度特征(256维)的门控自适应权重进行抑制/激发。
    实现：白盒临床知识 指导 黑盒深度网络 抽取针对缺血的细粒度病灶。
    """
    def __init__(self, encoder_weight_path=None, in_channels=36, num_handcrafted_features=9, feature_dim=256):
        super(HybridFineTuneModel, self).__init__()
        
        # 1. 深度学习骨干 (完全可以直接加载现成的预训练权重，无需重练！)
        self.encoder = MCGEncoder(in_channels=in_channels, feature_dim=feature_dim)
        if encoder_weight_path:
            state_dict = torch.load(encoder_weight_path, map_location='cpu')
            if any(k.startswith('encoder.') for k in state_dict.keys()):
                encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
                self.encoder.load_state_dict(encoder_state)
            else:
                self.encoder.load_state_dict(state_dict)

        # 2. 传统波形特征提取器
        self.handcrafted_extractor = WaveformFeatureExtractor()
        
        # 维数设定
        self.handcrafted_dim = in_channels * num_handcrafted_features # 36 * 9 = 324
        self.deep_dim = feature_dim # 256
        
        # ======================================================
        # 🚀 你的专属创新：跨模态特征门控生成器 (Cross-Modal Gating)
        # ======================================================
        # 将手工特征对齐/压缩到与深度特征相同维度
        self.hand_bottleneck = nn.Sequential(
            nn.Linear(self.handcrafted_dim, self.deep_dim),
            nn.BatchNorm1d(self.deep_dim),
            nn.ReLU()
        )
        
        # 门控注意力信号生成 (输出 0~1 的权重向量)
        self.attention_gate = nn.Sequential(
            nn.Linear(self.deep_dim, self.deep_dim),
            nn.Sigmoid()
        )
        
        # 融合网络最后负责收尾判别的强力全连接分类器
        combined_dim = self.deep_dim * 2 # 门控激发的深度特征256 + 原始指标的特征256 = 512
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(combined_dim),
            nn.Dropout(0.5),
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # 步骤 1: 获取原味预训练的黑盒深度特征 -> 维度 [B, 256]
        deep_features = self.encoder(x)
        
        # 步骤 2: 提取临床统计白盒特征 -> 维度 [B, 324]
        hand_features = self.handcrafted_extractor(x)
        
        # 步骤 3: 提取对齐后的先验信息 -> 维度 [B, 256]
        hand_aligned = self.hand_bottleneck(hand_features)
        
        # 步骤 4(⭐创新核): 发动交叉门控，生成通道特征激活权重 -> 维度 [B, 256]
        # (即明确告诉深度网络哪几个通道目前在物理层面正在发生剧烈抖动)
        gating_weight = self.attention_gate(hand_aligned)
        
        # 步骤 5: 门控阻滞与激发，并做残差连接以保护预训练本身的主心骨信息
        gated_deep_features = deep_features * gating_weight + deep_features
        
        # 步骤 6: 信息双轨并肩，共同走向分类器
        fused_features = torch.cat([gated_deep_features, hand_aligned], dim=1)
        logits = self.classifier(fused_features)
        
        return logits.squeeze(-1)

    def freeze_encoder(self):
        # 微调阶段1: 冻结这批庞大的参数，只练下面我们手搭的那个新门控机制
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        # 微调阶段2: 全部解冻联调
        for param in self.encoder.parameters():
            param.requires_grad = True
