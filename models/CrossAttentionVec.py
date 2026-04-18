import torch
import torch.nn as nn


class BiDirectionalCrossAttention(nn.Module):
    """
    双向交叉注意力模块 (Symmetric Co-Attention)
    时域与频域相互作为 Query 提取对方的信息
    """

    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        # 时域查频域
        self.attn_st_to_sf = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # 频域查时域
        self.attn_sf_to_st = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm_st = nn.LayerNorm(embed_dim)
        self.norm_sf = nn.LayerNorm(embed_dim)

    def forward(self, st_seq, sf_seq):
        # 1. 时域作为 Query, 频域作为 Key/Value
        st_attended, _ = self.attn_st_to_sf(query=st_seq, key=sf_seq, value=sf_seq)
        st_out = self.norm_st(st_seq + st_attended)

        # 2. 频域作为 Query, 时域作为 Key/Value
        sf_attended, _ = self.attn_sf_to_st(query=sf_seq, key=st_seq, value=st_seq)
        sf_out = self.norm_sf(sf_seq + sf_attended)

        # 3. 拼接双向增强后的特征 [B, SeqLen, embed_dim * 2]
        fused = torch.cat([st_out, sf_out], dim=-1)
        return fused


class DualBranchMCGEncoder(nn.Module):
    def __init__(self, in_channels=36, feature_dim=256):
        super(DualBranchMCGEncoder, self).__init__()

        # STFT 参数设置
        self.n_fft = 64
        self.hop_length = 16
        # 输出频点数: n_fft // 2 + 1 = 33
        self.freq_bins = self.n_fft // 2 + 1

        self.register_buffer('window', torch.hann_window(self.n_fft))

        # ==========================================
        # 分支 1: 时空分支 (Spatio-Temporal Branch)
        # ==========================================
        # 输入: [B, 1, 700, 6, 6]
        self.st_branch = nn.Sequential(
            # 时间轴下采样 4 倍: 700 -> 175
            nn.Conv3d(1, 32, kernel_size=(7, 3, 3), stride=(4, 1, 1), padding=(3, 1, 1)),
            nn.BatchNorm3d(32),
            nn.GELU(),

            # 时间轴下采样 4 倍: 175 -> 44, 空间下采样 6x6 -> 3x3
            nn.Conv3d(32, 64, kernel_size=(5, 3, 3), stride=(4, 2, 2), padding=(2, 1, 1)),
            nn.BatchNorm3d(64),
            nn.GELU(),

            # 时间轴下采样 2 倍: 44 -> 22, 通道升至 128
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.GELU(),

            # 抹平空间维度，保留时序: [B, 128, 22, H, W] -> [B, 128, 22, 1, 1]
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )

        # ==========================================
        # 分支 2: 频空分支 (Spatio-Frequency Branch)
        # ==========================================
        # 输入: [B, Freq(33), Time(45), 6, 6] (把频率当作通道数，省显存！)
        self.sf_branch = nn.Sequential(
            # 时间轴下采样 2 倍: 45 -> 23
            nn.Conv3d(self.freq_bins, 64, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.BatchNorm3d(64),
            nn.GELU(),

            # 通道升至 128，时间进一步微调对齐到 22
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.GELU(),

            # 抹平空间维度，强制对齐时间维度到 22: [B, 128, 22, 1, 1]
            nn.AdaptiveAvgPool3d((22, 1, 1))
        )

        # ==========================================
        # 融合与输出 (Fusion & Projection)
        # ==========================================
        # 注意: 经过交叉注意力拼接后，通道数变为 128 * 2 = 256
        self.bi_cross_attn = BiDirectionalCrossAttention(embed_dim=128, num_heads=4)

        self.out_proj = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.GELU()
        )

    def forward(self, x):
        """ x shape: [B, 36, 700] """
        B, C, T = x.shape

        # -------------------------------------------
        # 1. 准备时空分支数据 [B, 1, 700, 6, 6]
        # -------------------------------------------
        # permute(0,2,1) 把通道放到最后，方便 view 成 6x6
        x_st = x.permute(0, 2, 1).view(B, T, 6, 6).unsqueeze(1)

        # -------------------------------------------
        # 2. 准备频空分支数据 (STFT) [B, 33, Time, 6, 6]
        # -------------------------------------------
        # 将 Batch 和 Channel 合并，做批量的 STFT
        x_flat = x.view(B * C, T)

        # STFT 输出: [B*36, Freq(33), Time(45)]
        stft_out = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
            center=True
        ).abs()

        _, F_bins, T_bins = stft_out.shape

        # 形状还原与转换: [B*36, 33, 45] -> [B, 36, 33, 45] -> [B, 6, 6, 33, 45]
        x_sf = stft_out.view(B, 6, 6, F_bins, T_bins)
        # 乾坤大挪移：把 Freq 放到通道维度，Time 放到深度维度 -> [B, Freq, Time, H, W]
        x_sf = x_sf.permute(0, 3, 4, 1, 2)

        # -------------------------------------------
        # 3. 提取特征
        # -------------------------------------------
        # st_feat: [B, 128, 22, 1, 1] -> 转换为序列 [B, 22, 128]
        feat_st = self.st_branch(x_st).squeeze(-1).squeeze(-1).permute(0, 2, 1)

        # sf_feat: [B, 128, 22, 1, 1] -> 转换为序列 [B, 22, 128]
        feat_sf = self.sf_branch(x_sf).squeeze(-1).squeeze(-1).permute(0, 2, 1)

        # -------------------------------------------
        # 4. 双向交叉注意力融合
        # -------------------------------------------
        # fused_seq: [B, 22, 256]
        fused_seq = self.bi_cross_attn(feat_st, feat_sf)

        # -------------------------------------------
        # 5. 全局时序池化与映射
        # -------------------------------------------
        # 沿着时间序列求平均 -> [B, 256]
        pooled_feat = fused_seq.mean(dim=1)

        # 投影到目标特征维度
        out = self.out_proj(pooled_feat)
        return out

# ==========================================
# 补充：专为双分支架构设计的 SimCLR 包装壳
# ==========================================
class DualBranchSimCLR(nn.Module):
    """
    封装 双分支 Encoder 和 Projection Head。
    """
    def __init__(self, in_channels=36, feature_dim=256, proj_hidden_dim=512, out_dim=128):
        super(DualBranchSimCLR, self).__init__()

        # 核心骨干：使用你新写的双分支编码器
        self.encoder = DualBranchMCGEncoder(in_channels=in_channels, feature_dim=feature_dim)

        # 投影头：两层 MLP，用于计算对比损失
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.SiLU(),
            nn.Linear(proj_hidden_dim, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


# --- 验证测试 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(32, 36, 700).to(device)

    model = DualBranchMCGEncoder(feature_dim=256).to(device)
    out = model(dummy_input)

    print(f"✅ 模型测试通过！输出维度: {out.shape}")