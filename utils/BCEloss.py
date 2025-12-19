import torch
import torch.nn as nn


class BCEWithLogitsLossWithSmoothing(nn.Module):
    def __init__(self, label_smoothing=0.0, pos_weight=None, reduction='mean'):
        """
        兼容旧版 PyTorch 的带 Label Smoothing 的 BCEWithLogitsLoss
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        # 初始化标准的 BCEWithLogitsLoss (不带 label_smoothing 参数)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, input, target):
        # 如果开启了平滑，手动修改 target
        if self.label_smoothing > 0:
            # 公式: new_target = target * (1 - ε) + 0.5 * ε
            # 这会将 0 变成 0.5ε，将 1 变成 1 - 0.5ε
            with torch.no_grad():
                target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        return self.bce(input, target)