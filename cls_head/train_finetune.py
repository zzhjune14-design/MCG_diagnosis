import os
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

# 假设这两个类在 finetune_model.py 中定义
from finetune_model import MCGFineTuneDataset, MCGFineTuneModel


def train_finetune():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    EPOCHS_WARMUP = 5
    EPOCHS_JOINT = 30
    TASK = "xinshuai"  # 任务: "Ischemia" 或 "xinshuai"
    SAVE_DIR = "./finetune_checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 初始化全局 CSV 日志 (新增了灵敏度 Sens、特异性 Spec、F1)
    log_file = os.path.join(SAVE_DIR, f"{TASK}_training_log.csv")
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Fold", "Phase", "Epoch", "Train_Loss", "Val_Loss",
                         "Val_Acc", "Val_Sens", "Val_Spec", "Val_F1", "Val_AUC"])

    # 1. 读入患者标签
    df = pd.read_csv("E:\Pythonpro\MCG_quexue_xinshuai\label.csv")
    if TASK == "xinshuai":
        df = df.dropna(subset=['xinshuai'])

    subjects = df['subject'].values
    y_patient = (df[TASK].values == 1.0).astype(int) if TASK == "Ischemia" else (df[TASK].values == 1.0).astype(int)

    # 全局 Dataset (懒加载)
    DATA_DIR = r"E:\Pythonpro\MCG_quexue_xinshuai\all_data_folder"  # 替换为实际存放.npy的目录
    full_dataset = MCGFineTuneDataset(df, DATA_DIR, task=TASK)

    # 计算 BCE Loss 权重应对潜在不平衡
    pos_count = sum(full_dataset.labels)
    neg_count = len(full_dataset.labels) - pos_count
    pos_weight = torch.tensor([neg_count / max(1, pos_count)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 2. 5 折分层交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(subjects, y_patient)):
        print(f"\n{'=' * 20} 正在训练第 {fold + 1} 折 {'=' * 20}")

        train_subjects = set(subjects[train_idx])
        val_subjects = set(subjects[val_idx])

        # 筛选对应的样本索引 (假设完整文件名为 subject_X.npy)
        train_indices = [i for i, (path, _) in enumerate(full_dataset.sample_index_map)
                         if int(os.path.basename(path).split('_')[-1].split('.')[0]) in train_subjects]
        val_indices = [i for i, (path, _) in enumerate(full_dataset.sample_index_map)
                       if int(os.path.basename(path).split('_')[-1].split('.')[0]) in val_subjects]

        train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=True)
        val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False)

        # 3. 初始化模型
        # model = MCGFineTuneModel(encoder_weight_path="../checkpoints/mcg2vec_encoder_ep100_RESCUED.pth").to(DEVICE)
        model = MCGFineTuneModel(encoder_weight_path=r"D:\New_python_project\MCG_diagnosis\cls_head\finetune_checkpoints\best_model_Ischemia_fold5.pth").to(DEVICE)

        # 用来记录该 Fold 的最佳验证 AUC
        best_auc = 0.0

        # 阶段一：Warmup (冻结 Encoder)
        model.freeze_encoder()
        optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
        print("--- 阶段一：冻结预训练编码器，预热分类头 (5 Epochs) ---")
        best_auc = run_epochs(model, train_loader, val_loader, optimizer, criterion, EPOCHS_WARMUP, DEVICE,
                              fold=fold + 1, phase="Warmup", log_file=log_file, save_dir=SAVE_DIR, task=TASK,
                              best_auc=best_auc)

        # 阶段二：全量微调 (解冻 Encoder)
        model.unfreeze_encoder()
        optimizer = optim.Adam(model.parameters(), lr=3e-3)
        print("--- 阶段二：解冻编码器，联合微调全网参数 (30 Epochs) ---")
        best_auc = run_epochs(model, train_loader, val_loader, optimizer, criterion, EPOCHS_JOINT, DEVICE,
                              fold=fold + 1, phase="Joint", log_file=log_file, save_dir=SAVE_DIR, task=TASK,
                              best_auc=best_auc)


def run_epochs(model, train_loader, val_loader, optimizer, criterion, epochs, device, fold, phase, log_file, save_dir,
               task, best_auc):
    for epoch in range(epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        valid_batches = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            # 🚑 防线 1：全面清洗 NaN, 正无穷 (posinf), 负无穷 (neginf)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            # 🚑 防线 2：拦截异常 Loss，防止脏梯度污染模型权重
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()

            # 🚑 防线 3：加入梯度裁剪，强制将过大的梯度缩放回安全范围
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
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                x, y = x.to(device), y.to(device)
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)

        # 🚑 防线 4：最后兜底，确保传入 sklearn 的预测值绝对干净
        all_preds = np.nan_to_num(np.array(all_preds), nan=0.0, posinf=0.0, neginf=0.0)
        all_labels = np.array(all_labels)

        # ==========================================
        # 计算 5 大核心指标
        # ==========================================
        val_auc = roc_auc_score(all_labels, all_preds)
        preds_binary = (all_preds > 0.5).astype(int)
        val_acc = accuracy_score(all_labels, preds_binary)
        val_f1 = f1_score(all_labels, preds_binary, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(all_labels, preds_binary, labels=[0, 1]).ravel()
        val_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        val_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(
            f"📊 {phase} Ep {epoch + 1}: Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Sens: {val_sens:.4f} | Spec: {val_spec:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([fold, phase, epoch + 1,
                             round(train_loss, 4), round(val_loss, 4),
                             round(val_acc, 4), round(val_sens, 4),
                             round(val_spec, 4), round(val_f1, 4), round(val_auc, 4)])

        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join(save_dir, f"best_model_{task}_fold{fold}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🌟 发现最佳验证 AUC ({best_auc:.4f})，已保存最佳权重至: {save_path}")

    return best_auc


if __name__ == "__main__":
    train_finetune()