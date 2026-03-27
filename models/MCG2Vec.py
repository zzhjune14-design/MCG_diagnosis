import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. MCG2Vec 卷积编码器 (基于一维卷积)
# ==========================================
class MCGEncoder(nn.Module):
    """
    针对 36通道、700长度心磁信号设计的 1D-CNN 编码器。
    采用类似 ResNet 的基本模块进行下采样和特征提取。
    """

    def __init__(self, in_channels=36, feature_dim=256):
        super(MCGEncoder, self).__init__()

        # 特征提取模块
        self.conv_blocks = nn.Sequential(
            # Block 1: [B, 36, 700] -> [B, 64, 350]
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # Block 2: [B, 64, 350] -> [B, 128, 175]
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # Block 3: [B, 128, 175] -> [B, 256, 88]
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            # Block 4: [B, 256, 88] -> [B, 256, 44]
            nn.Conv1d(256, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )

        # 全局平均池化，抹平时间维度：[B, 256, 44] -> [B, 256, 1] -> [B, 256]
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)  # 展平为向量
        return x


# ==========================================
# 2. SimCLR 模型封装 (Encoder + Projection Head)
# ==========================================
class MCG2VecSimCLR(nn.Module):
    """
    封装 Encoder 和 Projection Head。
    注意：对比学习的 Loss 是在 Projection Head 的输出上计算的，
    但在下游任务微调时，我们会丢弃 Projection Head，只用 Encoder。
    """

    def __init__(self, in_channels=36, feature_dim=256, proj_hidden_dim=512, out_dim=128):
        super(MCG2VecSimCLR, self).__init__()

        self.encoder = MCGEncoder(in_channels=in_channels, feature_dim=feature_dim)

        # Projection Head: 两层 MLP，论文微调时提到了 Swish 激活 [cite: 405]，这里我们在投影层保持非线性
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.SiLU(),  # SiLU 等价于 Swish
            nn.Linear(proj_hidden_dim, out_dim)
        )

    def forward(self, x):
        # 1. 获取基础表征 (用于下游任务)
        h = self.encoder(x)
        # 2. 获取投影表征 (仅用于计算对比损失)
        z = self.projector(h)
        return h, z


# ==========================================
# 3. NT-Xent 对比损失函数 (Bug 修复版)
# ==========================================
class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss
    论文中设定 temperature = 0.1
    """

    def __init__(self, batch_size, temperature=0.1):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2, eps=1e-6)

    def forward(self, z_i, z_j):
        """
        z_i: 来自视图 1 的投影特征 [Batch, Dim]
        z_j: 来自视图 2 的投影特征 [Batch, Dim]
        """
        # 沿着 Batch 维度拼接两个视图，总共 2N 个样本
        z = torch.cat((z_i, z_j), dim=0)  # [2N, Dim]

        # 计算 2N x 2N 的余弦相似度矩阵
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # 构造 Labels：对于正样本对，i 与 i+N 互为正样本
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # 正样本对应的 logits 放在矩阵第一列
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2 * self.batch_size, 1)

        # 提取负样本：掩码掉对角线（自己和自己）以及正样本的位置
        mask = self._get_correlated_mask(self.batch_size).to(z.device)
        negative_samples = sim[mask].reshape(2 * self.batch_size, -1)

        # Logits: [正样本相似度, 负样本相似度1, 负样本相似度2, ...]
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        # Label 永远是 0，因为我们把正样本放在了第 0 列
        labels = torch.zeros(2 * self.batch_size, dtype=torch.long).to(z.device)

        loss = self.criterion(logits, labels)
        return loss / (2 * self.batch_size)

    def _get_correlated_mask(self, batch_size):
        """修复点：使用 torch.diag 替代报错的 torch.eye(..., k=...)"""
        # 主对角线 (自己和自己)
        diag = torch.eye(2 * batch_size)
        # 上偏移对角线 (视图1 -> 视图2 的正样本)
        l1 = torch.diag(torch.ones(batch_size), diagonal=batch_size)
        # 下偏移对角线 (视图2 -> 视图1 的正样本)
        l2 = torch.diag(torch.ones(batch_size), diagonal=-batch_size)

        # 组合并取反，得到负样本的 mask
        mask = (diag + l1 + l2).bool()
        return ~mask


# ==========================================
# 4. 测试验证
# ==========================================
if __name__ == "__main__":
    # 模拟参数
    BATCH_SIZE = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟数据输入 (2个增强视图)
    x_i = torch.randn(BATCH_SIZE, 36, 700).to(device)
    x_j = torch.randn(BATCH_SIZE, 36, 700).to(device)

    # 初始化网络与 Loss
    model = MCG2VecSimCLR(in_channels=36, feature_dim=256, out_dim=128).to(device)
    criterion = NTXentLoss(batch_size=BATCH_SIZE, temperature=0.1)  # 论文中设为 0.1

    # 前向传播
    h_i, z_i = model(x_i)
    h_j, z_j = model(x_j)

    # 计算 Loss
    loss = criterion(z_i, z_j)

    print(f"✅ 模型测试通过！")
    print(f"   - 基础表征 (h) 维度: {h_i.shape} (微调时使用)")
    print(f"   - 投影表征 (z) 维度: {z_i.shape} (计算对比损失使用)")
    print(f"   - 当前 Batch NT-Xent Loss: {loss.item():.4f}")