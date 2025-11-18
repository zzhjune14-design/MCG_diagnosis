import torch
import torch.nn as nn
from typing import Optional

# -----------------------
# Adapter: (6,6,t) -> (36,t)
# -----------------------
class AmcgTo36(nn.Module):
    """
    Convert amcg arrays into 36 channels.
    Input accepted shapes:
      - (B, 6, 6, t)
      - (6, 6, t)  -> converted to (1,6,6,t)
      - (B, 36, t) -> returned as-is (no-op)
    modes:
      - "reshape": simply flatten the 6x6 into 36 (row-major)
      - "bn": reshape then BatchNorm1d(36) (per-channel norm)
      - "mix1x1": reshape then 1x1 Conv1d(36->36) to learn channel mixing
    """
    def __init__(self, mode: str = "reshape", flatten_order: str = "row", eps: float = 1e-5):
        super().__init__()
        assert mode in ("reshape", "bn", "mix1x1")
        assert flatten_order in ("row", "col")
        self.mode = mode
        self.flatten_order = flatten_order
        if mode == "bn":
            self.bn = nn.BatchNorm1d(36, eps=eps)
        elif mode == "mix1x1":
            # learnable linear mixing across the 36 channels (applied per time step)
            # implemented as Conv1d with kernel_size=1
            self.mix = nn.Conv1d(36, 36, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor):
        """
        x: tensor in one of accepted shapes.
        returns: tensor (B,36,t)
        """
        # handle (6,6,t) -> (1,6,6,t)
        if x.dim() == 3:
            # assume (6,6,t)
            x = x.unsqueeze(0)

        # if already (B,36,t) -> pass-through
        if x.dim() == 3 and x.size(1) == 36:
            return x.float()

        # expect (B,6,6,t)
        if x.dim() != 4 or x.size(1) != 6 or x.size(2) != 6:
            raise ValueError(f"AmcgTo36 expected (B,6,6,t) or (B,36,t) or (6,6,t). Got {tuple(x.shape)}")

        B, _, _, T = x.shape
        # flatten 6x6 -> 36
        # choose order: row-major (default) or column-major
        if self.flatten_order == "row":
            # x: (B,6,6,T) -> permute to (B, 6, 6, T) already -> reshape
            x36 = x.reshape(B, 36, T)  # row-major flatten preserves order (0,0),(0,1)...(5,5)
        else:
            # column-major: transpose first then reshape
            x36 = x.transpose(1,2).reshape(B, 36, T)

        x36 = x36.float()

        if self.mode == "reshape":
            return x36

        if self.mode == "bn":
            # BatchNorm1d expects (B, C, L)
            return self.bn(x36)

        if self.mode == "mix1x1":
            # Conv1d operates on (B, C, L)
            return self.mix(x36)


# -----------------------
# Example CNN that accepts raw amcg
# -----------------------
class CNN1D_from_amcg(nn.Module):
    def __init__(self,
                 num_classes: int,
                 adapter_mode: str = "reshape",
                 in_channels: int = 36,
                 channels=(64,128,256),
                 kernels=(7,5,3),
                 pools=(2,2,2),
                 dropout=0.3,
                 fc_hidden=256):
        super().__init__()
        # adapter: (6,6,t) -> (36,t)
        self.adapter = AmcgTo36(mode=adapter_mode, flatten_order="row")

        # build conv stack (operates on (B,36,t))
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cur_ch, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Accepts:
          - (B,6,6,t) tensor
          - (6,6,t) tensor (single sample)
          - (B,36,t) preprocessed tensor (will be passed through adapter as no-op)
        Returns:
          logits (B, num_classes)
        """
        # adapter handles shape checking and conversion
        x36 = self.adapter(x)       # (B,36,t)
        # pass through conv layers
        feat = self.features(x36)   # (B, C_last, L')
        pooled = self.global_pool(feat).squeeze(-1)  # (B, C_last)
        logits = self.classifier(pooled)             # (B, num_classes)
        return logits.squeeze(-1)



# -----------------------
# quick demo
# -----------------------
if __name__ == "__main__":
    B = 4
    t = 1000
    # create dummy raw input in shape (B,6,6,t)
    raw = torch.randn(B, 6, 6, t)

    for mode in ("reshape", "bn", "mix1x1"):
        model = CNN1D_from_amcg(num_classes=5, adapter_mode=mode)
        model.eval()
        out = model(raw)  # internally converts to (B,36,t)
        print(f"mode={mode}: out.shape={out.shape}, params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
