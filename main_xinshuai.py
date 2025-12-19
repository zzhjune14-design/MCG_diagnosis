import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc

# --- 导入您项目中的工具 ---
# 请确保这些路径与您的项目结构一致
from data_process.data_utils import set_seed, gather_pickle_files, _get_binary_labels_from_raws_or_map
from models.model_st import STNet_MCG  # 使用您的 STNet
from data_process.build_data import build_dataloaders
from utils.BCEloss import BCEWithLogitsLossWithSmoothing
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.save_result import append_metrics_csv, plot_metrics_curves, save_roc_plot, save_confusion_matrix, \
    save_run_metadata


# -----------------------------------------------------------------------------
# 1. 定义单任务训练函数 (只训练 Xinshuai)
# -----------------------------------------------------------------------------
def train_epoch_single(model, loader, optimizer, loss_fn, device, label_map, field_target="xinshuai"):
    model.train()
    running_loss = 0.0
    n_samples = 0

    # 只记录这一个任务的指标
    accum = {'probs': [], 'preds': [], 'trues': []}

    for Xb, subjects, raws in tqdm(loader, desc="Train Xinshuai", leave=False):
        Xb = Xb.to(device)

        # 1. 获取标签 (只获取心衰的)
        # 注意：对于非缺血病人，这里会自动返回 -1
        y_np = _get_binary_labels_from_raws_or_map(raws, subjects, field_target, label_map)

        # Mask: 过滤掉 -1 的无效样本 (即过滤掉非缺血病人)
        mask = (y_np != -1)

        # 如果这一批全是 -1 (比如全是健康人)，直接跳过计算，防止报错
        if mask.sum() == 0:
            continue

        # 将 Numpy 转 Tensor，并处理 Mask (无效填 0)
        y_tensor = torch.tensor(np.where(mask, y_np, 0), dtype=torch.float32, device=device).view(-1, 1)

        optimizer.zero_grad()

        # 2. 前向传播
        out = model(Xb)

        # 兼容性处理
        if isinstance(out, dict):
            logit = out.get(field_target)
        else:
            # 假设模型返回 (logit_isch, logit_xin)，我们只取第二个
            _, logit = out

        if logit.dim() == 1: logit = logit.view(-1, 1)

        # 3. 计算 Loss (Masked Loss)
        # 先算所有样本的 Loss
        loss_per_sample = loss_fn(logit, y_tensor).view(-1)

        # 只对 Mask 为 True 的样本求和，并除以有效样本数
        loss = (loss_per_sample * torch.tensor(mask, dtype=torch.float32, device=device)).sum() / max(mask.sum(), 1)

        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 统计 Loss (只统计有效样本的 Loss)
        effective_batch_size = mask.sum().item()
        running_loss += loss.item() * effective_batch_size
        n_samples += effective_batch_size

        # 4. 收集数据用于计算指标
        probs = torch.sigmoid(logit).detach().cpu().numpy().reshape(-1)
        preds = (probs > 0.5).astype(int).tolist()  # 默认阈值 0.5 (后续在 inference 时可调整)

        # 只收集有效样本 (Masked)
        accum['probs'].extend(probs[mask].tolist())
        accum['preds'].extend(np.array(preds)[mask].tolist())
        accum['trues'].extend(y_np[mask].tolist())

    avg_loss = running_loss / n_samples if n_samples > 0 else 0.0

    # 计算指标
    metrics = _metrics_from_acc(accum)
    return avg_loss, metrics, accum


# -----------------------------------------------------------------------------
# 2. 定义单任务验证函数
# -----------------------------------------------------------------------------
@torch.no_grad()
def eval_epoch_single(model, loader, loss_fn, device, label_map, field_target="xinshuai"):
    model.eval()
    running_loss = 0.0
    n_samples = 0
    accum = {'probs': [], 'preds': [], 'trues': []}

    for Xb, subjects, raws in tqdm(loader, desc="Eval Xinshuai", leave=False):
        Xb = Xb.to(device)
        y_np = _get_binary_labels_from_raws_or_map(raws, subjects, field_target, label_map)

        mask = (y_np != -1)
        if mask.sum() == 0: continue

        y_tensor = torch.tensor(np.where(mask, y_np, 0), dtype=torch.float32, device=device).view(-1, 1)

        out = model(Xb)
        if isinstance(out, dict):
            logit = out.get(field_target)
        else:
            _, logit = out

        if logit.dim() == 1: logit = logit.view(-1, 1)

        loss_per_sample = loss_fn(logit, y_tensor).view(-1)
        loss = (loss_per_sample * torch.tensor(mask, dtype=torch.float32, device=device)).sum() / max(mask.sum(), 1)

        effective_batch_size = mask.sum().item()
        running_loss += loss.item() * effective_batch_size
        n_samples += effective_batch_size

        probs = torch.sigmoid(logit).cpu().numpy().reshape(-1)
        preds = (probs > 0.5).astype(int).tolist()

        accum['probs'].extend(probs[mask].tolist())
        accum['preds'].extend(np.array(preds)[mask].tolist())
        accum['trues'].extend(y_np[mask].tolist())

    avg_loss = running_loss / n_samples if n_samples > 0 else 0.0
    metrics = _metrics_from_acc(accum)
    return avg_loss, metrics, accum


# --- 辅助函数：计算指标 ---
def _metrics_from_acc(a):
    trues = np.array(a['trues'])
    preds = np.array(a['preds'])
    probs = np.array(a['probs'])

    if len(trues) == 0:
        return {'acc': 0, 'f1': 0, 'sens': 0, 'spec': 0, 'auc': 0}

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, zero_division=0)

    tp = int(((preds == 1) & (trues == 1)).sum())
    tn = int(((preds == 0) & (trues == 0)).sum())
    fp = int(((preds == 1) & (trues == 0)).sum())
    fn = int(((preds == 0) & (trues == 1)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        if len(np.unique(trues)) > 1:
            auc_val = roc_auc_score(trues, probs)
        else:
            auc_val = 0.5
    except Exception:
        auc_val = 0.0

    return {'acc': acc, 'f1': f1, 'sens': sens, 'spec': spec, 'auc': auc_val}


# -----------------------------------------------------------------------------
# 3. 主流程函数 (Main Xinshuai)
# -----------------------------------------------------------------------------
def main_xinshuai(pickle_folder, label_csv, out_dir="./output",
                  model_name="STNet_Xinshuai_Only",  # 独立目录
                  batch_size=32, epochs=70, lr=1e-3,
                  seed=42, use_amp=True, num_workers=4,
                  resume_from=None, save_every_n_epochs=1, patience=15):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 目标字段
    field_xin = "xinshuai"

    # 数据准备：关键修改 stratify_by="xinshuai"
    # 这会保证每折里 (心衰/非心衰) 的比例一致，而不是 (缺血/非缺血)
    print(f"Building dataloaders (Stratify by {field_xin})...")
    dl_res = build_dataloaders(
        pickle_folder=pickle_folder,
        label_csv=label_csv,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        stratify_by=field_xin
    )

    if isinstance(dl_res, tuple) and len(dl_res) == 2:
        dataloaders_per_fold, label_map = dl_res
    else:
        dataloaders_per_fold = [(dl_res[0], dl_res[1])]
        label_map = dl_res[2]

    # 开始交叉验证
    cv_summary = []

    for fold_idx, (train_loader, val_loader) in enumerate(dataloaders_per_fold):
        print(f"\n{'=' * 40}")
        print(f"Starting Fold {fold_idx}")
        print(f"{'=' * 40}")

        # 目录准备
        fold_dir = os.path.join(out_dir, model_name, f"fold_{fold_idx}")
        ckpt_dir = os.path.join(fold_dir, "checkpoints")
        plots_dir = os.path.join(fold_dir, "plots")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # 模型初始化 (Dropout 可以稍微大一点，心衰容易过拟合)
        model = STNet_MCG(dropout=0.6, fc_hidden=256).to(device)

        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        # Loss: 使用 0.15 的标签平滑 (之前的经验值)
        # 如果样本极度不平衡，可以在这里加 pos_weight
        loss_fn = BCEWithLogitsLossWithSmoothing(label_smoothing=0.15)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # 指标记录
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        train_f1s, val_f1s = [], []
        train_aucs, val_aucs = [], []

        best_val_auc = -1.0
        patience_counter = 0
        start_epoch = 1

        # 训练循环
        for epoch in range(start_epoch, epochs + 1):
            # --- Train ---
            t_loss, t_met, _ = train_epoch_single(model, train_loader, optimizer, loss_fn, device, label_map, field_xin)

            # --- Eval ---
            v_loss, v_met, v_accum = eval_epoch_single(model, val_loader, loss_fn, device, label_map, field_xin)

            scheduler.step()

            # --- Logging ---
            print(f"Fold {fold_idx} Epoch {epoch}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"  [Loss] Train: {t_loss:.4f}  Val: {v_loss:.4f}")
            print(f"  [Train] Acc: {t_met['acc']:.4f}  F1: {t_met['f1']:.4f}  AUC: {t_met['auc']:.4f}")
            print(
                f"  [Val  ] Acc: {v_met['acc']:.4f}  F1: {v_met['f1']:.4f}  AUC: {v_met['auc']:.4f}  Sens: {v_met['sens']:.4f}")
            print("-" * 60)

            # --- Append ---
            train_losses.append(t_loss);
            val_losses.append(v_loss)
            train_accs.append(t_met['acc']);
            val_accs.append(v_met['acc'])
            train_f1s.append(t_met['f1']);
            val_f1s.append(v_met['f1'])
            train_aucs.append(t_met['auc']);
            val_aucs.append(v_met['auc'])

            # --- Checkpoint & Early Stopping ---
            current_metric = v_met['auc']

            # Save Last
            save_checkpoint(os.path.join(ckpt_dir, "last.pth"), model, optimizer, epoch, best_val_auc)

            if current_metric > best_val_auc:
                print(f"  🔥 New Best AUC: {best_val_auc:.4f} -> {current_metric:.4f}")
                best_val_auc = current_metric
                patience_counter = 0

                save_checkpoint(os.path.join(ckpt_dir, "best.pth"), model, optimizer, epoch, best_val_auc)

                # 画 ROC
                if len(np.unique(v_accum['trues'])) > 1:
                    fpr, tpr, _ = roc_curve(v_accum['trues'], v_accum['probs'])
                    save_roc_plot(plots_dir, f"best_roc_fold{fold_idx}", fpr, tpr, v_met['auc'], epoch)
                    save_confusion_matrix(plots_dir, f"best_cm_fold{fold_idx}", v_accum['trues'], v_accum['preds'],
                                          epoch)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"🛑 Early stopping at epoch {epoch}")
                    break

        # Fold 结束
        plot_metrics_curves(plots_dir, f"Xinshuai_Fold{fold_idx}",
                            train_losses, val_losses, train_accs, val_accs,
                            train_f1s, val_f1s, train_aucs, val_aucs)

        # Summary
        cv_summary.append({
            "fold": fold_idx,
            "best_val_auc": best_val_auc,
            "final_acc": v_met['acc'],
            "final_f1": v_met['f1']
        })

    # 全部结束
    summary_path = os.path.join(out_dir, model_name, "cv_summary.json")
    with open(summary_path, "w") as f:
        json.dump(cv_summary, f, indent=2)
    print(f"\nDone! Results saved to {summary_path}")


# -----------------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    config_path = r"config/config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    else:
        cfg = {
            "pickle_folder": r"E:\Pythonpro\MCG_quexue_xinshuai\data_pickle",
            "label_csv": r"E:\Pythonpro\MCG_quexue_xinshuai\label.csv",
            "out_dir": "./output"
        }

    main_xinshuai(
        pickle_folder=cfg['pickle_folder'],
        label_csv=cfg['label_csv'],
        out_dir=cfg['out_dir'],
        model_name="STNet_Xinshuai_Only",
        batch_size=32,
        epochs=70,
        patience=15
    )