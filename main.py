# main.py
import os
import json
import time
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc

from data_process.data_utils import set_seed, build_dataloaders, gather_pickle_files
from models.CNN1D import CNN1D_from_amcg

# 尝试导入结果保存与 checkpoint 工具（兼容多种文件命名）
try:
    # 你原来可能用的是 utils.save_result
    from utils.save_result import (
        append_metrics_csv, plot_metrics_curves, save_roc_plot,
        save_confusion_matrix, save_run_metadata
    )
except Exception:
    # 备选名（assistant 也可能提供过不同名字）
    try:
        from utils.save_result import (
            append_metrics_csv, plot_metrics_curves, save_roc_plot,
            save_confusion_matrix, save_run_metadata
        )
    except Exception as e:
        raise ImportError("请确保 utils/save_result.py 中包含 append_metrics_csv, plot_metrics_curves, "
                          "save_roc_plot, save_confusion_matrix, save_run_metadata 等函数。错误：" + str(e))

# checkpoint 工具导入（兼容 utils.checkpoint 或 utils.checkpoint_utils）
try:
    from utils.checkpoint import save_checkpoint, load_checkpoint
except Exception:
    try:
        from utils.checkpoint import save_checkpoint, load_checkpoint
    except Exception as e:
        raise ImportError("请确保存在 utils/checkpoint.py 或 utils/checkpoint_utils.py，且包含 save_checkpoint/load_checkpoint。错误：" + str(e))


# -------------------------
# 辅助：从 batch raws 或 label_map 获取二分类标签（0/1/-1）
# -------------------------
def _normalize_binary_label_value(v) -> int:
    """
    将可能的输入值标准化为 0/1/-1：
      - 有效标签：1 / 0
      - 缺失标签：None / NaN -> -1
    """
    if v is None:
        return -1
    try:
        if isinstance(v, float) and np.isnan(v):
            return -1
    except Exception:
        pass
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "yes", "y", "true", "t", "有", "positive", "pos"):
            return 1
        if s in ("2", "0", "no", "n", "false", "f", "无", "negative", "neg"):
            return 0
        try:
            iv = int(s)
            return 1 if iv == 1 else 0
        except Exception:
            return -1
    if isinstance(v, (int, float, bool, np.integer, np.floating)):
        try:
            iv = int(v)
            return 1 if iv == 1 else 0
        except Exception:
            return -1
    return -1


def _get_binary_labels_from_raws_or_map(raws, subjects, field_name: str, label_map: dict):
    """
    返回 numpy array shape (B,) of ints (0/1/-1)
    -1 表示该样本无标签
    """
    B = len(subjects)
    labels = []

    def _try_get_from_raw(r, key):
        if not isinstance(r, dict):
            return None
        if key in r:
            return r.get(key)
        lk = key.lower()
        uk = key.upper()
        for k in (key, lk, uk):
            if k in r:
                return r.get(k)
        return None

    if isinstance(raws, (list, tuple)) and len(raws) > 0 and isinstance(raws[0], dict):
        example_val = _try_get_from_raw(raws[0], field_name)
        if example_val is not None:
            for r in raws:
                v = _try_get_from_raw(r, field_name)
                labels.append(_normalize_binary_label_value(v))
            return np.array(labels, dtype=int)

    for s in subjects:
        lm_entry = label_map.get(s, None)
        if lm_entry is None:
            labels.append(-1)
            continue
        if isinstance(lm_entry, dict):
            raw_v = lm_entry.get(field_name, lm_entry.get(field_name.lower(), -1))
        else:
            raw_v = lm_entry
        labels.append(_normalize_binary_label_value(raw_v))
    return np.array(labels, dtype=int)


# -------------------------
# train / eval 函数支持 mask
# -------------------------
def train_epoch(model, loader, optimizer, loss_fns: dict,
                device, label_map, field_isch="Ischemia", field_xin="xinshuai"):
    model.train()
    running_loss = 0.0
    n_samples = 0

    accum = {
        field_isch: {'probs': [], 'preds': [], 'trues': []},
        field_xin:  {'probs': [], 'preds': [], 'trues': []}
    }

    for Xb, subjects, raws in tqdm(loader, desc="train", leave=False):
        Xb = Xb.to(device)
        y_isch_np = _get_binary_labels_from_raws_or_map(raws, subjects, field_isch, label_map)
        y_xin_np  = _get_binary_labels_from_raws_or_map(raws, subjects, field_xin, label_map)

        # mask: -1 表示缺失
        mask_isch = (y_isch_np != -1)
        mask_xin  = (y_xin_np  != -1)

        y_isch = torch.tensor(np.where(mask_isch, y_isch_np, 0), dtype=torch.float32, device=device).view(-1,1)
        y_xin  = torch.tensor(np.where(mask_xin, y_xin_np, 0), dtype=torch.float32, device=device).view(-1,1)

        optimizer.zero_grad()
        out = model(Xb)
        if isinstance(out, dict):
            logit_isch = out.get(field_isch)
            logit_xin  = out.get(field_xin)
        else:
            logit_isch, logit_xin = out

        if logit_isch.dim() == 1:
            logit_isch = logit_isch.view(-1,1)
        if logit_xin.dim() == 1:
            logit_xin = logit_xin.view(-1,1)

        # per-sample loss
        loss_isch_per_sample = loss_fns[field_isch](logit_isch, y_isch).view(-1)
        loss_xin_per_sample  = loss_fns[field_xin](logit_xin, y_xin).view(-1)

        # 只统计 mask 内的样本
        loss_isch = (loss_isch_per_sample * torch.tensor(mask_isch, dtype=torch.float32, device=device)).sum() / max(mask_isch.sum(),1)
        loss_xin  = (loss_xin_per_sample  * torch.tensor(mask_xin,  dtype=torch.float32, device=device)).sum() / max(mask_xin.sum(),1)

        total_loss = loss_isch + loss_xin
        total_loss.backward()
        optimizer.step()

        batch_size = Xb.size(0)
        running_loss += total_loss.item() * batch_size
        n_samples += batch_size

        probs_isch = torch.sigmoid(logit_isch).detach().cpu().numpy().reshape(-1)
        probs_xin  = torch.sigmoid(logit_xin).detach().cpu().numpy().reshape(-1)
        preds_isch = (probs_isch > 0.5).astype(int).tolist()
        preds_xin  = (probs_xin  > 0.5).astype(int).tolist()

        accum[field_isch]['probs'].extend(probs_isch[mask_isch].tolist())
        accum[field_isch]['preds'].extend(np.array(preds_isch)[mask_isch].tolist())
        accum[field_isch]['trues'].extend(y_isch_np[mask_isch].tolist())

        accum[field_xin]['probs'].extend(probs_xin[mask_xin].tolist())
        accum[field_xin]['preds'].extend(np.array(preds_xin)[mask_xin].tolist())
        accum[field_xin]['trues'].extend(y_xin_np[mask_xin].tolist())

    avg_loss = running_loss / n_samples if n_samples > 0 else float('nan')

    def _metrics_from_acc(a):
        trues = np.array(a['trues'])
        preds = np.array(a['preds'])
        probs = np.array(a['probs'])
        acc = accuracy_score(trues, preds) if len(trues)>0 else float('nan')
        f1 = f1_score(trues, preds, zero_division=0) if len(trues)>0 else float('nan')
        tp = int(((preds==1) & (trues==1)).sum())
        tn = int(((preds==0) & (trues==0)).sum())
        fp = int(((preds==1) & (trues==0)).sum())
        fn = int(((preds==0) & (trues==1)).sum())
        sens = tp / (tp + fn) if (tp + fn)>0 else 0.0
        spec = tn / (tn + fp) if (tn + fp)>0 else 0.0
        try:
            auc_val = roc_auc_score(trues, probs) if len(np.unique(trues))>1 else float('nan')
        except Exception:
            auc_val = float('nan')
        return {'acc': acc, 'f1': f1, 'sens': sens, 'spec': spec, 'auc': auc_val}

    metrics = {
        field_isch: _metrics_from_acc(accum[field_isch]),
        field_xin:  _metrics_from_acc(accum[field_xin])
    }
    return avg_loss, metrics, accum


@torch.no_grad()
def eval_epoch(model, loader, loss_fns: dict,
               device, label_map, field_isch="Ischemia", field_xin="xinshuai"):
    # 和 train_epoch 一样逻辑，只是去掉 optimizer/梯度
    model.eval()
    running_loss = 0.0
    n_samples = 0

    accum = {
        field_isch: {'probs': [], 'preds': [], 'trues': []},
        field_xin:  {'probs': [], 'preds': [], 'trues': []}
    }

    for Xb, subjects, raws in tqdm(loader, desc="eval", leave=False):
        Xb = Xb.to(device)
        y_isch_np = _get_binary_labels_from_raws_or_map(raws, subjects, field_isch, label_map)
        y_xin_np  = _get_binary_labels_from_raws_or_map(raws, subjects, field_xin, label_map)

        mask_isch = (y_isch_np != -1)
        mask_xin  = (y_xin_np  != -1)

        y_isch = torch.tensor(np.where(mask_isch, y_isch_np, 0), dtype=torch.float32, device=device).view(-1,1)
        y_xin  = torch.tensor(np.where(mask_xin, y_xin_np, 0), dtype=torch.float32, device=device).view(-1,1)

        out = model(Xb)
        if isinstance(out, dict):
            logit_isch = out.get(field_isch)
            logit_xin  = out.get(field_xin)
        else:
            logit_isch, logit_xin = out

        if logit_isch.dim() == 1:
            logit_isch = logit_isch.view(-1,1)
        if logit_xin.dim() == 1:
            logit_xin = logit_xin.view(-1,1)

        loss_isch_per_sample = loss_fns[field_isch](logit_isch, y_isch).view(-1)
        loss_xin_per_sample  = loss_fns[field_xin](logit_xin, y_xin).view(-1)

        loss_isch = (loss_isch_per_sample * torch.tensor(mask_isch, dtype=torch.float32, device=device)).sum() / max(mask_isch.sum(),1)
        loss_xin  = (loss_xin_per_sample  * torch.tensor(mask_xin,  dtype=torch.float32, device=device)).sum() / max(mask_xin.sum(),1)

        total_loss = loss_isch + loss_xin

        batch_size = Xb.size(0)
        running_loss += total_loss.item() * batch_size
        n_samples += batch_size

        probs_isch = torch.sigmoid(logit_isch).cpu().numpy().reshape(-1)
        probs_xin  = torch.sigmoid(logit_xin).cpu().numpy().reshape(-1)
        preds_isch = (probs_isch > 0.5).astype(int).tolist()
        preds_xin  = (probs_xin > 0.5).astype(int).tolist()

        accum[field_isch]['probs'].extend(probs_isch[mask_isch].tolist())
        accum[field_isch]['preds'].extend(np.array(preds_isch)[mask_isch].tolist())
        accum[field_isch]['trues'].extend(y_isch_np[mask_isch].tolist())

        accum[field_xin]['probs'].extend(probs_xin[mask_xin].tolist())
        accum[field_xin]['preds'].extend(np.array(preds_xin)[mask_xin].tolist())
        accum[field_xin]['trues'].extend(y_xin_np[mask_xin].tolist())

    avg_loss = running_loss / n_samples if n_samples > 0 else float('nan')

    def _metrics_from_acc(a):
        trues = np.array(a['trues'])
        preds = np.array(a['preds'])
        probs = np.array(a['probs'])
        acc = accuracy_score(trues, preds) if len(trues)>0 else float('nan')
        f1 = f1_score(trues, preds, zero_division=0) if len(trues)>0 else float('nan')
        tp = int(((preds==1) & (trues==1)).sum())
        tn = int(((preds==0) & (trues==0)).sum())
        fp = int(((preds==1) & (trues==0)).sum())
        fn = int(((preds==0) & (trues==1)).sum())
        sens = tp / (tp + fn) if (tp + fn)>0 else 0.0
        spec = tn / (tn + fp) if (tn + fp)>0 else 0.0
        try:
            auc_val = roc_auc_score(trues, probs) if len(np.unique(trues))>1 else float('nan')
        except Exception:
            auc_val = float('nan')
        return {'acc': acc, 'f1': f1, 'sens': sens, 'spec': spec, 'auc': auc_val}

    metrics = {
        field_isch: _metrics_from_acc(accum[field_isch]),
        field_xin:  _metrics_from_acc(accum[field_xin])
    }
    return avg_loss, metrics, accum



# -------------------------
# 主函数：支持单折或多折（folds），按 model_name 创建输出路径
# -------------------------
def main(pickle_folder: str, label_csv: str, out_dir: str = "./output",
         adapter_mode: str = "bn", batch_size: int = 8, epochs: int = 20,
         lr: float = 1e-3, num_workers: int = 4,
         seed: int = 42, resume_from: str = None, use_amp: bool = False,
         model_name: str = "CNN1D", field_isch: str = "Ischemia", field_xin: str = "xinshuai",
         save_every_n_epochs: int = 1):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # 请求 dataloaders（注意：你的 build_dataloaders 已改为可能返回 folds）
    dl_res = build_dataloaders(
        pickle_folder=pickle_folder,
        label_csv=label_csv,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
    )

    # 解析返回
    if isinstance(dl_res, tuple) and len(dl_res) == 3:
        dataloaders_per_fold = [(dl_res[0], dl_res[1])]
        label_map = dl_res[2]
    elif isinstance(dl_res, tuple) and len(dl_res) == 2:
        dataloaders_per_fold, label_map = dl_res
    else:
        raise RuntimeError("build_dataloaders 返回类型不符合预期")

    files_all = gather_pickle_files(pickle_folder)
    print(f"Found {len(files_all)} pickle files, labels: {len(label_map)}")
    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # per-fold loop
    cv_summary = []
    for fold_idx, (train_loader, val_loader) in enumerate(dataloaders_per_fold):
        fold_name = f"fold_{fold_idx}" if len(dataloaders_per_fold) > 1 else "fold_0"
        model_output_dir = os.path.join(out_dir, model_name, fold_name)
        plots_dir = os.path.join(model_output_dir, "plots")
        ckpt_dir = os.path.join(model_output_dir, "checkpoints")
        os.makedirs(model_output_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        # 保存 run metadata
        meta = {
            "model_name": model_name,
            "adapter_mode": adapter_mode,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "num_workers": num_workers,
            "seed": seed,
            "fold": fold_idx,
            "timestamp": time_stamp
        }
        save_run_metadata(model_output_dir, meta)

        model = CNN1D_from_amcg(adapter_mode=adapter_mode).to(device)

        print("Model params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fns = {
            field_isch: nn.BCEWithLogitsLoss(),
            field_xin:  nn.BCEWithLogitsLoss()
        }
        scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

        # resume
        start_epoch = 1
        best_val_metric = -1.0
        if resume_from is not None and os.path.exists(resume_from):
            print(f"Loading checkpoint from {resume_from} ...")
            start_epoch, best_val_metric, ckpt_label_map = load_checkpoint(resume_from, model, optimizer, scaler)
            if ckpt_label_map is not None:
                label_map = ckpt_label_map
            print(f" Resuming from epoch {start_epoch}, best_val_metric={best_val_metric}")

        # containers for plotting
        train_losses, val_losses = [], []
        train_accs_isch, val_accs_isch = [], []
        train_f1_isch, val_f1_isch = [], []
        train_auc_isch, val_auc_isch = [], []

        train_accs_xin, val_accs_xin = [], []
        train_f1_xin, val_f1_xin = [], []
        train_auc_xin, val_auc_xin = [], []

        # 训练循环
        for epoch in range(start_epoch, epochs + 1):
            print(f"\nFold {fold_idx} Epoch {epoch}/{epochs}")
            train_loss, train_metrics, train_accum = train_epoch(model, train_loader, optimizer, loss_fns,
                                                                 device, label_map, field_isch=field_isch, field_xin=field_xin)
            val_loss, val_metrics, val_accum = eval_epoch(model, val_loader, loss_fns,
                                                          device, label_map, field_isch=field_isch, field_xin=field_xin)

            # 打印每个任务指标
            print(f"[{field_isch}] train acc={train_metrics[field_isch]['acc']:.4f} f1={train_metrics[field_isch]['f1']:.4f} auc={train_metrics[field_isch]['auc']:.4f}")
            print(f"[{field_isch}]  val  acc={val_metrics[field_isch]['acc']:.4f} f1={val_metrics[field_isch]['f1']:.4f} auc={val_metrics[field_isch]['auc']:.4f}")
            print(f"[{field_xin}]  train acc={train_metrics[field_xin]['acc']:.4f} f1={train_metrics[field_xin]['f1']:.4f} auc={train_metrics[field_xin]['auc']:.4f}")
            print(f"[{field_xin}]  val  acc={val_metrics[field_xin]['acc']:.4f} f1={val_metrics[field_xin]['f1']:.4f} auc={val_metrics[field_xin]['auc']:.4f}")

            # 保存数据到内存（用于绘图）
            train_losses.append(train_loss); val_losses.append(val_loss)
            train_accs_isch.append(train_metrics[field_isch]['acc']); val_accs_isch.append(val_metrics[field_isch]['acc'])
            train_f1_isch.append(train_metrics[field_isch]['f1']); val_f1_isch.append(val_metrics[field_isch]['f1'])
            train_auc_isch.append(train_metrics[field_isch]['auc']); val_auc_isch.append(val_metrics[field_isch]['auc'])

            train_accs_xin.append(train_metrics[field_xin]['acc']); val_accs_xin.append(val_metrics[field_xin]['acc'])
            train_f1_xin.append(train_metrics[field_xin]['f1']); val_f1_xin.append(val_metrics[field_xin]['f1'])
            train_auc_xin.append(train_metrics[field_xin]['auc']); val_auc_xin.append(val_metrics[field_xin]['auc'])

            # 组织 metrics 行并追加到 CSV（动态 header）
            metrics_row = {
                'epoch': epoch,
                'train_loss': train_loss, 'val_loss': val_loss,
                f'{field_isch}_train_acc': train_metrics[field_isch]['acc'],
                f'{field_isch}_val_acc':   val_metrics[field_isch]['acc'],
                f'{field_isch}_train_f1':  train_metrics[field_isch]['f1'],
                f'{field_isch}_val_f1':    val_metrics[field_isch]['f1'],
                f'{field_isch}_train_auc': train_metrics[field_isch]['auc'],
                f'{field_isch}_val_auc':   val_metrics[field_isch]['auc'],
                f'{field_xin}_train_acc': train_metrics[field_xin]['acc'],
                f'{field_xin}_val_acc':   val_metrics[field_xin]['acc'],
                f'{field_xin}_train_f1':  train_metrics[field_xin]['f1'],
                f'{field_xin}_val_f1':    val_metrics[field_xin]['f1'],
                f'{field_xin}_train_auc': train_metrics[field_xin]['auc'],
                f'{field_xin}_val_auc':   val_metrics[field_xin]['auc'],
            }
            append_metrics_csv(model_output_dir, model_name, metrics_row)

            # 保存 last checkpoint
            last_path = os.path.join(ckpt_dir, "last_checkpoint.pth")
            save_checkpoint(last_path, model, optimizer, epoch, best_val_metric, scaler=scaler, label_map=label_map)

            # 判断是否最好（此处用两个任务 val F1 的平均作为主度量；你可按需修改）
            avg_val_f1 = (val_metrics[field_isch]['f1'] + val_metrics[field_xin]['f1']) / 2.0
            if avg_val_f1 > best_val_metric:
                best_val_metric = avg_val_f1
                best_path = os.path.join(ckpt_dir, "best_checkpoint.pth")
                save_checkpoint(best_path, model, optimizer, epoch, best_val_metric, scaler=scaler, label_map=label_map)
                print("  saved best_checkpoint.pth")

            # 每 save_every_n_epochs 保存 ROC 与混淆矩阵，避免每个 epoch 都写图开销太大（可配置）
            if epoch % save_every_n_epochs == 0:
                # Ischemia ROC & CM
                if len(np.unique(val_accum[field_isch]['trues'])) > 1:
                    fpr, tpr, _ = roc_curve(val_accum[field_isch]['trues'], val_accum[field_isch]['probs'])
                    roc_auc_val = auc(fpr, tpr)
                    save_roc_plot(plots_dir, model_name + "_" + field_isch, fpr, tpr, roc_auc_val, epoch)
                save_confusion_matrix(plots_dir, model_name + "_" + field_isch, val_accum[field_isch]['trues'], val_accum[field_isch]['preds'], epoch)

                # Xinshuai ROC & CM
                if len(np.unique(val_accum[field_xin]['trues'])) > 1:
                    fpr2, tpr2, _ = roc_curve(val_accum[field_xin]['trues'], val_accum[field_xin]['probs'])
                    roc_auc_val2 = auc(fpr2, tpr2)
                    save_roc_plot(plots_dir, model_name + "_" + field_xin, fpr2, tpr2, roc_auc_val2, epoch)
                save_confusion_matrix(plots_dir, model_name + "_" + field_xin, val_accum[field_xin]['trues'], val_accum[field_xin]['preds'], epoch)

        # end epoch loop

        # 保存 final checkpoint
        final_path = os.path.join(ckpt_dir, "final_checkpoint.pth")
        save_checkpoint(final_path, model, optimizer, epochs, best_val_metric, scaler=scaler, label_map=label_map)

        # 绘制并保存 summary 曲线（整个训练过程）
        plot_metrics_curves(plots_dir, model_name + f"_fold{fold_idx}",
                            train_losses, val_losses,
                            train_accs_isch, val_accs_isch,
                            train_f1_isch, val_f1_isch,
                            train_auc_isch, val_auc_isch)
        # 另外为第二个任务也绘图（文件名里带任务名）
        plot_metrics_curves(plots_dir, model_name + f"_{field_xin}_fold{fold_idx}",
                            train_losses, val_losses,
                            train_accs_xin, val_accs_xin,
                            train_f1_xin, val_f1_xin,
                            train_auc_xin, val_auc_xin)

        # 记录该折最佳指标到 cv_summary
        cv_summary.append({
            "fold": fold_idx,
            "best_avg_val_f1": best_val_metric,
            f"{field_isch}_best_val_f1": val_metrics[field_isch]['f1'],
            f"{field_xin}_best_val_f1": val_metrics[field_xin]['f1']
        })

    # end fold loop

    # 保存 cv_summary 到主目录（如果是单折也会保存）
    summary_path = os.path.join(out_dir, model_name, "cv_summary.json")
    with open(summary_path, "w") as f:
        json.dump(cv_summary, f, indent=2)

    print("Training finished for all folds. CV summary saved to", summary_path)


if __name__ == "__main__":
    PICKLE_FOLDER = r"E:\Pythonpro\MCG_quexue_xinshuai\data_pickle"
    LABEL_CSV = r"E:\Pythonpro\MCG_quexue_xinshuai\label.csv"
    main(PICKLE_FOLDER, LABEL_CSV, out_dir="./output", adapter_mode="bn",
         batch_size=8, epochs=30, lr=1e-3, num_workers=4, seed=42,
         resume_from=None, use_amp=False, model_name="CNN1D",
         field_isch="Ischemia", field_xin="xinshuai", save_every_n_epochs=1)
