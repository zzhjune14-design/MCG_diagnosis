"""
双分支编码器 (DualBranchMCGEncoder) 分类头微调训练脚本。

架构：
    预训练编码器 (时空+频空+交叉注意力融合) ──→ 分类头 ──→ 二分类

训练策略 (两阶段)：
    阶段一 (Warmup)   : 冻结 Encoder，仅预热分类头 (5 Epoch)
    阶段二 (Joint)    : 解冻 Encoder，全量联合微调 (30 Epoch)

验证方式：
    5 折分层交叉验证 (按患者级别划分，同一患者的所有心拍不会跨折)

评估指标：
    Loss / Acc / Sensitivity / Specificity / F1 / AUC
"""

import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm

# 将项目根目录添加到 sys.path，确保能正确导入 models 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from finetune_model import DualBranchFineTuneDataset, DualBranchFineTuneModel


def train_finetune():
    """双分支编码器分类头微调主流程"""

    # ==========================================
    # 🔧 超参数配置 (在这里统一修改)
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64                  # 双分支模型较大，batch 不宜太大
    EPOCHS_WARMUP = 5                # 阶段一：冻结 Encoder，预热分类头
    EPOCHS_JOINT = 30                # 阶段二：解冻 Encoder，联合微调
    LR_WARMUP = 1e-3                 # 预热阶段学习率
    LR_JOINT = 5e-4                  # 联合微调学习率 (比预热小，防止破坏预训练特征)
    N_FOLDS = 5                      # 交叉验证折数

    # ==========================================
    # 📂 路径配置
    # ==========================================
    # 任务选择: "Ischemia" (心肌缺血) 或 "xinshuai" (心衰)
    TASK = "xinshuai"

    # 预训练编码器权重 (你训练好的 100 轮双分支 backbone)
    ENCODER_WEIGHT_PATH = r"D:\New_python_project\MCG_diagnosis\checkpoints\dual_branch_mcg_encoder_ep100_RESCUED.pth"

    # 接力微调权重 (可选：设为 None 表示从预训练权重开始；
    #     设为某个已训好的 .pth 表示从该模型接力训练另一个任务)
    RELAY_WEIGHT_PATH = r"D:\New_python_project\MCG_diagnosis\finetune_checkpoints_dual\best_model_Ischemia_fold3.pth"

    # 数据路径
    LABEL_CSV = r"E:\Pythonpro\MCG_quexue_xinshuai\label.csv"
    DATA_DIR = r"E:\Pythonpro\MCG_quexue_xinshuai\all_data_folder"

    # 输出目录
    SAVE_DIR = "./finetune_checkpoints_dual"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ==========================================
    # 📊 初始化全局 CSV 日志
    # ==========================================
    log_file = os.path.join(SAVE_DIR, f"{TASK}_training_log.csv")
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Fold", "Phase", "Epoch", "Train_Loss", "Val_Loss",
                         "Val_Acc", "Val_Sens", "Val_Spec", "Val_F1", "Val_AUC"])

    # ==========================================
    # 1. 读入患者标签并构建数据集
    # ==========================================
    print(f"📋 正在读取标签文件: {LABEL_CSV}")
    df = pd.read_csv(LABEL_CSV)
    if TASK == "xinshuai":
        df = df.dropna(subset=['xinshuai'])

    subjects = df['subject'].values
    # 构建分层标签
    if TASK == "xinshuai":
        y_patient = (df[TASK].values == 1.0).astype(int)
    else:
        y_patient = (df[TASK].values == 1.0).astype(int)

    # 构建全局 Dataset
    print(f"📦 正在构建 {TASK} 任务的全局数据集...")
    full_dataset = DualBranchFineTuneDataset(df, DATA_DIR, task=TASK)

    # 计算 BCE Loss 权重应对样本不平衡
    pos_count = sum(full_dataset.labels)
    neg_count = len(full_dataset.labels) - pos_count
    pos_weight = torch.tensor([neg_count / max(1, pos_count)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"⚖️ 正负样本比: {neg_count:.0f}:{pos_count:.0f}, pos_weight={pos_weight.item():.2f}")

    # 确定使用的权重路径
    weight_path = RELAY_WEIGHT_PATH if RELAY_WEIGHT_PATH else ENCODER_WEIGHT_PATH

    # ==========================================
    # 2. 5 折分层交叉验证
    # ==========================================
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    all_fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(subjects, y_patient)):
        print(f"\n{'=' * 60}")
        print(f"🔄 正在训练第 {fold + 1}/{N_FOLDS} 折")
        print(f"{'=' * 60}")

        train_subjects = set(subjects[train_idx])
        val_subjects = set(subjects[val_idx])

        # 按患者级别筛选样本索引
        # 注意：文件名格式为 {subject_id}.npy
        train_indices = [i for i, (path, _) in enumerate(full_dataset.sample_index_map)
                         if int(os.path.basename(path).split('.')[0]) in train_subjects]
        val_indices = [i for i, (path, _) in enumerate(full_dataset.sample_index_map)
                       if int(os.path.basename(path).split('.')[0]) in val_subjects]

        print(f"   训练集: {len(train_indices)} 个心拍, 验证集: {len(val_indices)} 个心拍")

        train_loader = DataLoader(Subset(full_dataset, train_indices),
                                  batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(Subset(full_dataset, val_indices),
                                batch_size=BATCH_SIZE, shuffle=False)

        # ==========================================
        # 3. 初始化模型 (每折从预训练权重重新开始)
        # ==========================================
        model = DualBranchFineTuneModel(
            encoder_weight_path=weight_path,
            in_channels=36,
            feature_dim=256
        ).to(DEVICE)

        # 记录该 Fold 的最佳验证 AUC
        best_auc = 0.0

        # ------------------------------------------
        # 阶段一：Warmup (冻结 Encoder, 仅训练分类头)
        # ------------------------------------------
        model.freeze_encoder()
        optimizer = optim.Adam(model.classifier.parameters(), lr=LR_WARMUP)
        print(f"\n--- 阶段一：冻结编码器，预热分类头 ({EPOCHS_WARMUP} Epochs, LR={LR_WARMUP}) ---")
        best_auc = run_epochs(
            model, train_loader, val_loader, optimizer, criterion,
            EPOCHS_WARMUP, DEVICE,
            fold=fold + 1, phase="Warmup", log_file=log_file,
            save_dir=SAVE_DIR, task=TASK, best_auc=best_auc
        )

        # ------------------------------------------
        # 阶段二：全量微调 (解冻 Encoder + 分类头联合训练)
        # ------------------------------------------
        model.unfreeze_encoder()
        # 使用差异化学习率：Encoder 用更小的 LR 保护预训练特征
        optimizer = optim.Adam([
            {'params': model.encoder.parameters(), 'lr': LR_JOINT * 0.1},  # Encoder: 更小的学习率
            {'params': model.classifier.parameters(), 'lr': LR_JOINT}       # 分类头: 正常学习率
        ])
        print(f"\n--- 阶段二：解冻编码器，联合微调 ({EPOCHS_JOINT} Epochs, LR_enc={LR_JOINT * 0.1}, LR_cls={LR_JOINT}) ---")
        best_auc = run_epochs(
            model, train_loader, val_loader, optimizer, criterion,
            EPOCHS_JOINT, DEVICE,
            fold=fold + 1, phase="Joint", log_file=log_file,
            save_dir=SAVE_DIR, task=TASK, best_auc=best_auc
        )

        all_fold_results.append(best_auc)
        print(f"\n🏁 第 {fold + 1} 折最佳 AUC: {best_auc:.4f}")

    # ==========================================
    # 汇总所有折的结果
    # ==========================================
    print(f"\n{'=' * 60}")
    print(f"📊 {N_FOLDS} 折交叉验证结果汇总")
    print(f"{'=' * 60}")
    for i, auc_val in enumerate(all_fold_results):
        print(f"   Fold {i + 1}: AUC = {auc_val:.4f}")
    mean_auc = np.mean(all_fold_results)
    std_auc = np.std(all_fold_results)
    print(f"   ────────────────────────")
    print(f"   平均 AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"{'=' * 60}")


def run_epochs(model, train_loader, val_loader, optimizer, criterion,
               epochs, device, fold, phase, log_file, save_dir, task, best_auc):
    """
    运行指定数量的训练轮次，并在验证集上评估。

    参数：
        model: 模型
        train_loader: 训练集 DataLoader
        val_loader: 验证集 DataLoader
        optimizer: 优化器
        criterion: 损失函数
        epochs: 训练轮数
        device: 设备
        fold: 当前折数
        phase: "Warmup" 或 "Joint"
        log_file: CSV 日志文件路径
        save_dir: 模型保存目录
        task: 任务名称
        best_auc: 历史最佳 AUC

    返回：
        best_auc: 更新后的最佳 AUC
    """
    for epoch in range(epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        valid_batches = 0

        for x, y in tqdm(train_loader, desc=f"[Fold{fold}] {phase} Ep{epoch + 1}/{epochs} [Train]"):
            x, y = x.to(device), y.to(device)

            # 🚑 防线 1：清洗 NaN, Inf
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            # 🚑 防线 2：拦截异常 Loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()

            # 🚑 防线 3：梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()
            train_loss += loss.item()
            valid_batches += 1

        train_loss = train_loss / max(1, valid_batches)

        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"[Fold{fold}] {phase} Ep{epoch + 1}/{epochs} [Val]"):
                x, y = x.to(device), y.to(device)
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= max(len(val_loader), 1)

        # 🚑 防线 4：清洗预测值
        all_preds = np.nan_to_num(np.array(all_preds), nan=0.0, posinf=0.0, neginf=0.0)
        all_labels = np.array(all_labels)

        # ==========================================
        # 计算 5 大核心指标
        # ==========================================
        try:
            val_auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            val_auc = 0.5  # 只有一个类别时给默认值

        preds_binary = (all_preds > 0.5).astype(int)
        val_acc = accuracy_score(all_labels, preds_binary)
        val_f1 = f1_score(all_labels, preds_binary, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(all_labels, preds_binary, labels=[0, 1]).ravel()
        val_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        val_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(
            f"📊 {phase} Ep {epoch + 1}: "
            f"TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | Sens: {val_sens:.4f} | Spec: {val_spec:.4f} | "
            f"F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

        # 记录到 CSV
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([fold, phase, epoch + 1,
                             round(train_loss, 4), round(val_loss, 4),
                             round(val_acc, 4), round(val_sens, 4),
                             round(val_spec, 4), round(val_f1, 4), round(val_auc, 4)])

        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join(save_dir, f"best_model_{task}_fold{fold}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🌟 发现最佳 AUC ({best_auc:.4f})，已保存至: {save_path}")

    return best_auc


if __name__ == "__main__":
    train_finetune()
