import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import csv
from datetime import datetime

from data_process.mcg_dataset import MCGContrastiveDataset
from models.MCG2Vec import MCG2VecSimCLR, NTXentLoss
from models.CrossAttentionVec import DualBranchSimCLR


# 假设你已经导入了 Step 1 的 MCGContrastiveDataset
# 以及 Step 2 的 MCG2VecSimCLR, NTXentLoss
# from dataset import MCGContrastiveDataset
# from models import MCG2VecSimCLR, NTXentLoss

def train_simclr():
    # ==========================================
    # 0. 超参数与环境配置 (严格对齐文献)
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    ACCUM_STEPS = 4
    EPOCHS = 100
    INIT_LR = 1e-3
    MIN_LR = INIT_LR * 0.01
    WEIGHT_DECAY = 1e-4
    TEMPERATURE = 0.1

    SAVE_DIR = "./checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 【新增】断点恢复路径。如果为空字符串 ""，则从头训练；如果填入路径，则继续训练
    RESUME_CHECKPOINT = "./checkpoints/dual_branch_mcg_encoder_checkpoint_latest.pth"  # 示例: "./checkpoints2/dual_branch_mcg_encoder_checkpoint_latest.pth"
    START_EPOCH = 0  # 记录从哪个 epoch 开始

    # ==========================================
    # 0.5 【新增】保存超参数日志与初始化 CSV
    # ==========================================
    # 1. 保存配置 JSON
    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "batch_size": BATCH_SIZE,
        "accum_steps": ACCUM_STEPS,
        "epochs": EPOCHS,
        "init_lr": INIT_LR,
        "min_lr": MIN_LR,
        "weight_decay": WEIGHT_DECAY,
        "temperature": TEMPERATURE,
        "resume_checkpoint": RESUME_CHECKPOINT
    }
    with open(os.path.join(SAVE_DIR, "train_params.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # 2. 初始化 Loss 记录的 CSV 文件
    csv_log_path = os.path.join(SAVE_DIR, "training_loss.csv")
    # 如果是从头训练，则新建 CSV 并写入表头；如果是续训，则保留原有文件
    if not (RESUME_CHECKPOINT and os.path.isfile(RESUME_CHECKPOINT)):
        with open(csv_log_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Average_Loss", "Learning_Rate"])

    # ==========================================
    # 1. 数据加载
    # ==========================================
    # 请替换为你的实际数据路径
    HEALTHY_DIR = r'E:\Pythonpro\MCG_quexue_xinshuai\heartbeat_segments\healthy'
    SICK_DIR = r'E:\Pythonpro\MCG_quexue_xinshuai\heartbeat_segments\sick'

    dataset = MCGContrastiveDataset([HEALTHY_DIR, SICK_DIR])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0, pin_memory=True, drop_last=True)
    # ==========================================
    # 2. 模型、损失函数与优化器配置
    # ==========================================
    model = DualBranchSimCLR(in_channels=36, feature_dim=256, out_dim=128).to(DEVICE)
    criterion = NTXentLoss(batch_size=BATCH_SIZE, temperature=TEMPERATURE)

    # 优化器: AdamW
    optimizer = optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)

    # 学习率调度: 余弦退火 (Cosine Decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)

    # AMP 自动混合精度 (加速训练，节省显存)
    scaler = torch.amp.GradScaler('cuda')

    # ==========================================
    # 2.5 【新增】断点续训加载逻辑
    # ==========================================
    if RESUME_CHECKPOINT and os.path.isfile(RESUME_CHECKPOINT):
        print(f"🔄 正在从断点恢复训练: {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)

        # 恢复所有状态
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        START_EPOCH = checkpoint['epoch']

        print(f"✅ 成功加载断点！将从 Epoch {START_EPOCH + 1} 继续训练。")
    else:
        if RESUME_CHECKPOINT:
            print(f"⚠️ 找不到断点文件 {RESUME_CHECKPOINT}，将从头开始训练。")

    # ==========================================
    # 3. 核心训练循环
    # ==========================================
    print(f"🚀 开始预训练 (设备: {DEVICE})...")

    for epoch in range(START_EPOCH, EPOCHS):
        model.train()
        total_loss = 0.0

        # 进度条
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]")

        for step, (x_i, x_j) in enumerate(pbar):
            x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)

            # 修复点 1：强制清洗异常数据，将原数据中可能隐藏的 NaN 或 Inf 替换为 0
            x_i = torch.nan_to_num(x_i, nan=0.0, posinf=0.0, neginf=0.0)
            x_j = torch.nan_to_num(x_j, nan=0.0, posinf=0.0, neginf=0.0)

            with torch.amp.autocast('cuda'):
                _, z_i = model(x_i)
                _, z_j = model(x_j)
                loss = criterion(z_i, z_j) / ACCUM_STEPS

            # 异常拦截：如果这个 Batch 依然算出了 NaN（极小概率），直接跳过不更新，防止污染
            if torch.isnan(loss):
                continue

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
                # 修复点 2：在步进前解开 scaler，加入梯度裁剪（阈值设为 1.0 或 2.0 均可）
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            current_loss = loss.item() * ACCUM_STEPS
            total_loss += current_loss
            pbar.set_postfix({'Loss': f"{current_loss:.4f}", 'LR': f"{scheduler.get_last_lr()[0]:.2e}"})
        # 更新学习率
        scheduler.step()

        # 计算 Epoch 平均 Loss
        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"📊 Epoch [{epoch + 1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")

        # ==========================================
        # 【新增】将 Loss 写入 CSV (追加模式)
        # ==========================================
        with open(csv_log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, round(avg_loss, 6), current_lr])

        # ==========================================
        # 4. 模型保存 (区分: 续训断点 vs 纯 Encoder)
        # ==========================================
        # 4.1 每隔 5 个 Epoch 或者最后一天，保存给下游微调用的 Encoder
        if (epoch + 1) % 2 == 0 or (epoch + 1) == EPOCHS:
            encoder_path = os.path.join(SAVE_DIR, f"dual_branch_mcg_encoder_ep{epoch + 1}.pth")
            torch.save(model.encoder.state_dict(), encoder_path)
            print(f"💾 保存下游微调专用 Encoder 权重至: {encoder_path}")

        # 4.2 【新增】每一个 Epoch 都覆盖保存最新的“完整断点”，防止突然断电
        checkpoint_path = os.path.join(SAVE_DIR, "dual_branch_mcg_encoder_checkpoint_latest.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)


if __name__ == "__main__":
    train_simclr()