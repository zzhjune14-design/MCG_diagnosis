import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    空间注意力模块：自适应关注 6x6 网格中的关键传感器
    """

    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入 x: (B, C, T, H, W)
        # 我们先对时间 T 维度做平均，简化为 (B, C, H, W)
        x_mean_time = torch.mean(x, dim=2)

        # 在通道维度 C 上做 AvgPool 和 MaxPool -> (B, 1, H, W)
        avg_out = torch.mean(x_mean_time, dim=1, keepdim=True)
        max_out, _ = torch.max(x_mean_time, dim=1, keepdim=True)

        # 拼接 -> (B, 2, H, W)
        scale = torch.cat([avg_out, max_out], dim=1)

        # 卷积提取空间特征 -> (B, 1, H, W)
        scale = self.conv1(scale)

        # 生成注意力图 (0~1) -> (B, 1, 1, H, W) 方便广播乘法
        return self.sigmoid(scale).unsqueeze(2)


class ST_Block(nn.Module):
    """
    (2+1)D 时空卷积块：解耦空间卷积和时间卷积
    """

    def __init__(self, in_channels, out_channels, stride=(1, 1, 1)):
        super(ST_Block, self).__init__()

        # 1. 空间卷积: (1, 3, 3), 也就是只看 6x6 平面
        self.spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3),
                                      stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.bn_s = nn.BatchNorm3d(out_channels)

        # 2. 时间卷积: (k, 1, 1), 只在时间轴上滑动
        # stride 控制时间维度的下采样
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


class STNet_LSTM_MCG(nn.Module):
    """
    CRNN 架构：
    1. ST-CNN 提取局部空间形态特征
    2. BiLSTM 提取全局时序演变特征
    """

    def __init__(self, dropout=0.5, fc_hidden=256, lstm_hidden=128):
        super(STNet_LSTM_MCG, self).__init__()

        # --- 1. CNN 前端：特征提取 ---
        # Input: (B, 1, T, 6, 6)
        self.pre_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(1, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        # 下采样层：为了适配 LSTM，我们稍微减少下采样次数，保留一定的时间长度
        # 假设原始 T=1000
        self.layer1 = ST_Block(16, 32, stride=(2, 1, 1))  # T -> 500
        self.layer2 = ST_Block(32, 64, stride=(2, 1, 1))  # T -> 250
        self.layer3 = ST_Block(64, 128, stride=(2, 1, 1))  # T -> 125
        # (这里去掉了 layer4，因为 125 的序列长度对于 LSTM 来说正好，太短反而没意义)

        # 空间注意力
        self.spatial_att = SpatialAttention()

        # 空间池化：把 6x6 压扁，但 **保留时间 T**
        # 目标输出形状: (B, C, T, 1, 1)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # --- 2. LSTM 中端：时序建模 ---
        # input_size = 128 (CNN layer3 的输出通道数)
        # bidirectional = True -> 输出维度 = lstm_hidden * 2
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=lstm_hidden,
                            num_layers=2,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)

        # --- 3. 全连接 后端：多任务分类 ---
        # LSTM 的输出维度是 2 * lstm_hidden
        self.feature_dim = lstm_hidden * 2

        self.head_ischemia = nn.Sequential(
            nn.Linear(self.feature_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)
        )

        self.head_xinshuai = nn.Sequential(
            nn.Linear(self.feature_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)
        )

        self._init_weights()

    def forward(self, x):
        # Input x: (B, 6, 6, T)
        # 调整为 Conv3d 需要的 (B, 1, T, 6, 6)
        x = x.permute(0, 3, 1, 2).unsqueeze(1)

        # 1. CNN 特征提取
        x = self.pre_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # 此时 shape: (B, 128, 125, 6, 6)

        # 2. 空间注意力
        att_map = self.spatial_att(x)
        x = x * att_map

        # 3. 空间压缩 (保留时间)
        x = self.spatial_pool(x)  # -> (B, 128, 125, 1, 1)

        # 4. 维度调整喂给 LSTM
        # LSTM 需要: (Batch, Seq_Len, Features)
        x = x.squeeze(-1).squeeze(-1)  # -> (B, 128, 125)
        x = x.permute(0, 2, 1)  # -> (B, 125, 128)

        # 5. LSTM 前向传播
        # self.lstm 输出: (output, (h_n, c_n))
        lstm_out, _ = self.lstm(x)  # -> (B, 125, 256)

        # 6. 时序聚合
        # 策略：对所有时间步取平均 (Global Temporal Pooling)
        feat = torch.mean(lstm_out, dim=1)  # -> (B, 256)

        # 7. 多任务分类
        logit_isch = self.head_ischemia(feat).squeeze(-1)
        logit_xin = self.head_xinshuai(feat).squeeze(-1)

        return logit_isch, logit_xin

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                # LSTM 默认初始化通常足够，这里可以不做额外操作
                # 或者手动初始化 input-gate biases
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)