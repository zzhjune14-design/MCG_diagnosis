# models/CNN1D.py
import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple
import math

# -----------------------
# Adapter: (6,6,t) -> (36,t)
# -----------------------
class AmcgTo36(nn.Module):
    def __init__(self, mode: str = "reshape", flatten_order: str = "row", eps: float = 1e-5):
        super().__init__()
        assert mode in ("reshape", "bn", "mix1x1")
        assert flatten_order in ("row", "col")
        self.mode = mode
        self.flatten_order = flatten_order
        if mode == "bn":
            self.bn = nn.BatchNorm1d(36, eps=eps)
        elif mode == "mix1x1":
            self.mix = nn.Conv1d(36, 36, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor):
        # handle (6,6,t) -> (1,6,6,t)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # if already (B,36,t) -> pass-through
        if x.dim() == 3 and x.size(1) == 36:
            return x.float()

        # expect (B,6,6,t)
        if x.dim() != 4 or x.size(1) != 6 or x.size(2) != 6:
            raise ValueError(f"AmcgTo36 expected (B,6,6,t) or (B,36,t) or (6,6,t). Got {tuple(x.shape)}")

        B, _, _, T = x.shape
        if self.flatten_order == "row":
            x36 = x.reshape(B, 36, T)
        else:
            x36 = x.transpose(1,2).reshape(B, 36, T)

        x36 = x36.float()

        if self.mode == "reshape":
            return x36
        if self.mode == "bn":
            return self.bn(x36)
        if self.mode == "mix1x1":
            return self.mix(x36)


# -----------------------
# CNN that accepts raw amcg and outputs two logits
# -----------------------
class CNN1D_from_amcg(nn.Module):
    def __init__(self,
                 adapter_mode: str = "reshape",
                 in_channels: int = 36,
                 channels: Sequence[int] = (64, 128, 256),
                 kernels: Sequence[int] = (7, 5, 3),
                 pools: Sequence[int] = (2, 2, 2),
                 dropout: float = 0.3,
                 fc_hidden: int = 256):
        super().__init__()
        self.adapter = AmcgTo36(mode=adapter_mode, flatten_order="row")

        if isinstance(kernels, int):
            kernels = [kernels] * len(channels)
        if isinstance(pools, int):
            pools = [pools] * len(channels)

        layers = []
        cur_ch = in_channels
        for out_ch, k, p in zip(channels, kernels, pools):
            pad = k // 2
            layers.append(nn.Conv1d(cur_ch, out_ch, kernel_size=k, padding=pad))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if p and p > 1:
                layers.append(nn.MaxPool1d(p))
            layers.append(nn.Dropout(dropout))
            cur_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # two separate classification heads (each outputs one logit)
        self.head_ischemia = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cur_ch, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)
        )
        self.head_xinshuai = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cur_ch, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)
        )

        # 初始化权重
        self._init_weights()

    def forward(self, x: torch.Tensor):
        x36 = self.adapter(x)            # (B,36,t)
        feat = self.features(x36)        # (B,C,L')
        pooled = self.global_pool(feat).squeeze(-1)  # (B, C)
        logit_isch = self.head_ischemia(pooled).squeeze(-1)   # (B,)
        logit_xin = self.head_xinshuai(pooled).squeeze(-1)    # (B,)
        return logit_isch, logit_xin

    def _init_weights(self):
        """
        初始化模型权重：
        - Conv1d/Linear: Kaiming normal（fan_out, ReLU）
        - BatchNorm: weight=1 bias=0
        - Linear bias=0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))  # 或使用 kaiming_normal_
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
