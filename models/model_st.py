import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    空间注意力模块：让模型自动关注 6x6 网格中更重要的传感器位置
    """

    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        # 压缩时间轴和通道轴，只看空间
        # 输入假设是 (B, C, T, H, W)，我们在 H, W 上做注意力
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, T, H, W) -> 我们先要把 T 和 C 融合或者是对 T 做平均
        # 为了简单高效，我们在 "通道+时间" 维度上做 MaxPool 和 AvgPool
        # 变换为 (B, C*T, H, W) 太大，我们先对 T 维度做平均

        # x_mean_time: (B, C, H, W)
        x_mean_time = torch.mean(x, dim=2)

        # 在通道维度做 AvgPool 和 MaxPool -> (B, 1, H, W)
        avg_out = torch.mean(x_mean_time, dim=1, keepdim=True)
        max_out, _ = torch.max(x_mean_time, dim=1, keepdim=True)

        # 拼接 -> (B, 2, H, W)
        scale = torch.cat([avg_out, max_out], dim=1)

        # 卷积 -> (B, 1, H, W)
        scale = self.conv1(scale)

        # Sigmoid -> (B, 1, 1, H, W) 注意力图，广播到 T 和 C
        return self.sigmoid(scale).unsqueeze(2)


class ST_Block(nn.Module):
    """
    (2+1)D 时空卷积块
    先做 2D 空间卷积 (1, 3, 3)，再做 1D 时间卷积 (k, 1, 1)
    """

    def __init__(self, in_channels, out_channels, stride=(1, 1, 1)):
        super(ST_Block, self).__init__()

        # 1. 空间卷积: Kernel=(1, 3, 3), 也就是只看 6x6 平面，不跨越时间
        # Padding=(0, 1, 1) 保持 6x6 大小不变
        self.spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3),
                                      stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.bn_s = nn.BatchNorm3d(out_channels)

        # 2. 时间卷积: Kernel=(kernel_t, 1, 1), 只在时间轴上滑动
        # stride 控制时间维度的下采样 (比如 stride=2 时间长度减半)
        # stride 输入格式: (time_stride, 1, 1)
        self.temporal_conv = nn.Conv3d(out_channels, out_channels, kernel_size=(5, 1, 1),
                                       stride=stride, padding=(2, 0, 0), bias=False)
        self.bn_t = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # 残差连接 (Downsample)
        self.downsample = None
        if stride[0] != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x

        # 先空间
        out = self.spatial_conv(x)
        out = self.bn_s(out)
        out = self.relu(out)

        # 后时间
        out = self.temporal_conv(out)
        out = self.bn_t(out)

        # 残差相加
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class STNet_MCG(nn.Module):
    def __init__(self, dropout=0.3, fc_hidden=256):
        super(STNet_MCG, self).__init__()

        # 输入是 (B, 6, 6, T)，我们需要把它转成 Conv3d 需要的 (B, C, T, H, W)
        # 初始特征映射: 1通道 -> 16通道
        self.pre_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(1, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        # --- 核心主干：时空特征提取 ---
        # 保持 6x6 不变，只压缩时间 T
        self.layer1 = ST_Block(16, 32, stride=(2, 1, 1))  # T=1000 -> 500
        self.layer2 = ST_Block(32, 64, stride=(2, 1, 1))  # T=500 -> 250
        self.layer3 = ST_Block(64, 128, stride=(2, 1, 1))  # T=250 -> 125
        self.layer4 = ST_Block(128, 256, stride=(2, 1, 1))  # T=125 -> 62

        # --- 创新点：空间注意力 ---
        # 在高维特征上，让模型回头看一眼 6x6 哪里最重要
        self.spatial_att = SpatialAttention()

        # 全局池化: 把 (T, 6, 6) 全部 AvgPool 掉 -> (1, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # --- 多任务头 ---
        self.head_ischemia = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)
        )

        self.head_xinshuai = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)
        )

        self._init_weights()

    def forward(self, x):
        # x shape: (B, 6, 6, T)
        # Conv3d 需要: (B, C, T, H, W)

        # 1. 维度调整
        # (B, 6, 6, T) -> (B, T, 6, 6) -> (B, 1, T, 6, 6)
        x = x.permute(0, 3, 1, 2).unsqueeze(1)

        # 2. 前向特征提取
        x = self.pre_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 256, 62, 6, 6)

        # 3. 施加空间注意力 (让模型关注心脏核心区域)
        att_map = self.spatial_att(x)
        x = x * att_map

        # 4. 池化与分类
        x = self.global_pool(x)  # (B, 256, 1, 1, 1)

        # 两个头分别预测
        logit_isch = self.head_ischemia(x).squeeze(-1)
        logit_xin = self.head_xinshuai(x).squeeze(-1)

        return logit_isch, logit_xin

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)