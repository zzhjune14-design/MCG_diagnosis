# main.py
import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc

from data_process.data_utils import set_seed, gather_pickle_files, _get_binary_labels_from_raws_or_map, _normalize_binary_label_value
from models.model_st import STNet_MCG
from models.CNN1D import CNN1D_from_amcg
from data_process.build_data import build_dataloaders
from utils.BCEloss import BCEWithLogitsLossWithSmoothing
from utils.error_analysis import save_validation_results
from models.model_st_lstm import STNet_LSTM_MCG

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

THRESHOLD_ISCH = 0.5
THRESHOLD_XIN = 0.35



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

            # -----------------------------------------------------------------
            # 【推荐写法】：切片法 (无论 loss_fn 是 mean 还是 none 都适用，且更快)
            # -----------------------------------------------------------------

            # 1. 缺血 Loss
            # 找出有效的索引 (bool tensor)
            valid_idx_isch = (mask_isch > 0)

            if valid_idx_isch.sum() > 0:
                # 只把有效的数据喂给 loss 函数
                # 这样 loss 函数根本看不到无效数据，绝对安全
                loss_isch = loss_fns[field_isch](
                    logit_isch[valid_idx_isch],
                    y_isch[valid_idx_isch]
                )
            else:
                loss_isch = torch.tensor(0.0, device=device, requires_grad=True)

            # 2. 心衰 Loss (之前崩的地方)
            valid_idx_xin = (mask_xin > 0)

            if valid_idx_xin.sum() > 0:
                # 只把缺血病人的数据喂给心衰 loss 函数
                loss_xin = loss_fns[field_xin](
                    logit_xin[valid_idx_xin],
                    y_xin[valid_idx_xin]
                )
            else:
                # 如果这一批全是健康人，跳过
                loss_xin = torch.tensor(0.0, device=device, requires_grad=True)

        # -----------------------------------------------------------------

        total_loss = loss_isch + loss_xin
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_size = Xb.size(0)
        running_loss += total_loss.item() * batch_size
        n_samples += batch_size

        probs_isch = torch.sigmoid(logit_isch).detach().cpu().numpy().reshape(-1)
        probs_xin  = torch.sigmoid(logit_xin).detach().cpu().numpy().reshape(-1)
        preds_isch = (probs_isch > THRESHOLD_ISCH).astype(int).tolist()
        preds_xin  = (probs_xin  > THRESHOLD_XIN).astype(int).tolist()

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
        preds_isch = (probs_isch > THRESHOLD_ISCH).astype(int).tolist()
        preds_xin  = (probs_xin > THRESHOLD_XIN).astype(int).tolist()

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
         model_name: str = "MODEL_ST", field_isch: str = "Ischemia", field_xin: str = "xinshuai",
         save_every_n_epochs: int = 1, patience: int = 100):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # 数据加载模块
    dl_res = build_dataloaders(
        pickle_folder=pickle_folder,
        label_csv=label_csv,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        stratify_by="xinshuai"
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

        # model = CNN1D_from_amcg(adapter_mode=adapter_mode).to(device)

        # model = STNet_MCG(dropout=0.5, fc_hidden=256).to(device)

        model = STNet_LSTM_MCG(dropout=0.3, fc_hidden=256, lstm_hidden=128).to(device)

        print("Model params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        # 2. 【新增】定义调度器 (使用余弦退火，效果通常最好)
        # T_max=epochs: 让学习率在整个训练过程中从 1e-3 慢慢降到 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        loss_fns = {
            field_isch: BCEWithLogitsLossWithSmoothing(label_smoothing=0.1),
            field_xin:  BCEWithLogitsLossWithSmoothing(label_smoothing=0.15)
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

        patience_counter = 0

        # 训练循环
        for epoch in range(start_epoch, epochs + 1):
            print(f"\nFold {fold_idx} Epoch {epoch}/{epochs}")
            train_loss, train_metrics, train_accum = train_epoch(model, train_loader, optimizer, loss_fns,
                                                                 device, label_map, field_isch=field_isch, field_xin=field_xin)
            val_loss, val_metrics, val_accum = eval_epoch(model, val_loader, loss_fns,
                                                          device, label_map, field_isch=field_isch, field_xin=field_xin)

            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current LR: {current_lr:.6f}")
            # -----------------------------------------------------------
            # 打印每个任务的详细指标 (Acc / Sens / Spec / AUC)
            # -----------------------------------------------------------

            # 1. 优先打印 Loss
            print(f"  [Loss]      train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")

            # 2. 缺血任务 (Ischemia)
            print(
                f"  [{field_isch}] Train: acc={train_metrics[field_isch]['acc']:.4f}  sens={train_metrics[field_isch]['sens']:.4f}  spec={train_metrics[field_isch]['spec']:.4f}  auc={train_metrics[field_isch]['auc']:.4f}")
            print(
                f"  [{field_isch}] Val  : acc={val_metrics[field_isch]['acc']:.4f}  sens={val_metrics[field_isch]['sens']:.4f}  spec={val_metrics[field_isch]['spec']:.4f}  auc={val_metrics[field_isch]['auc']:.4f}")

            # 3. 心衰任务 (Xinshuai)
            print(
                f"  [{field_xin}]  Train: acc={train_metrics[field_xin]['acc']:.4f}  sens={train_metrics[field_xin]['sens']:.4f}  spec={train_metrics[field_xin]['spec']:.4f}  auc={train_metrics[field_xin]['auc']:.4f}")
            print(
                f"  [{field_xin}]  Val  : acc={val_metrics[field_xin]['acc']:.4f}  sens={val_metrics[field_xin]['sens']:.4f}  spec={val_metrics[field_xin]['spec']:.4f}  auc={val_metrics[field_xin]['auc']:.4f}")

            print("-" * 120)  # 分割线画长一点
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

            # -------------------------------------------------------
            # 1. 无条件保存 Last Checkpoint (用于断点续训)
            # -------------------------------------------------------
            last_path = os.path.join(ckpt_dir, "last_checkpoint.pth")
            save_checkpoint(last_path, model, optimizer, epoch, best_val_metric, scaler=scaler, label_map=label_map)

            # -------------------------------------------------------
            # 2. 择优保存 Best Checkpoint & 绘制最佳 ROC/混淆矩阵
            # -------------------------------------------------------
            # 计算当前分数的平均 F1
            avg_val_f1 = (val_metrics[field_isch]['f1'] + val_metrics[field_xin]['f1']) / 2.0

            if avg_val_f1 > best_val_metric:
                print(f"  SOTA! Avg F1 improved: {best_val_metric:.4f} -> {avg_val_f1:.4f}")
                best_val_metric = avg_val_f1

                patience_counter = 0

                # A. 保存最佳模型
                best_path = os.path.join(ckpt_dir, "best_checkpoint.pth")
                save_checkpoint(best_path, model, optimizer, epoch, best_val_metric, scaler=scaler, label_map=label_map)

                # B. 只有在创新高时，才更新 ROC 和 混淆矩阵 图片
                #    (这样 plots 文件夹里保留的就是效果最好那一轮的图)

                # --- 缺血任务图 ---
                if len(np.unique(val_accum[field_isch]['trues'])) > 1:
                    fpr, tpr, _ = roc_curve(val_accum[field_isch]['trues'], val_accum[field_isch]['probs'])
                    roc_auc_val = auc(fpr, tpr)
                    save_roc_plot(plots_dir, "best_" + model_name + "_" + field_isch, fpr, tpr, roc_auc_val,
                                  epoch=epoch)

                save_confusion_matrix(plots_dir, "best_" + model_name + "_" + field_isch,
                                      val_accum[field_isch]['trues'], val_accum[field_isch]['preds'], epoch=epoch)

                # --- 心衰任务图 ---
                if len(np.unique(val_accum[field_xin]['trues'])) > 1:
                    fpr, tpr, _ = roc_curve(val_accum[field_xin]['trues'], val_accum[field_xin]['probs'])
                    roc_auc_val = auc(fpr, tpr)
                    save_roc_plot(plots_dir, "best_" + model_name + "_" + field_xin, fpr, tpr, roc_auc_val, epoch=epoch)

                save_confusion_matrix(plots_dir, "best_" + model_name + "_" + field_xin,
                                          val_accum[field_xin]['trues'], val_accum[field_xin]['preds'], epoch=epoch)
            else:
                    patience_counter += 1
                    print(f"  No improvement for {patience_counter} epochs.")
                    if patience_counter >= patience:
                        print(f"  Early stopping after {epoch} epochs.")
                        break

            # =========================================================
            # END EPOCH LOOP (这里缩进退回一级，循环结束)
            # =========================================================

            # -------------------------------------------------------
            # 3. 训练结束收尾：保存 Final 模型
            # -------------------------------------------------------
        final_path = os.path.join(ckpt_dir, "final_checkpoint.pth")
        save_checkpoint(final_path, model, optimizer, epochs, best_val_metric, scaler=scaler, label_map=label_map)

        # -------------------------------------------------------
        # 4. 🌟 就在这里！绘制整个训练过程的 Loss / Acc / F1 / AUC 曲线
        #    (因为只有循环结束了，列表里的数据才是完整的)
        # -------------------------------------------------------
        print(f"Generating summary plots for fold {fold_idx}...")

        # 绘制缺血任务曲线
        plot_metrics_curves(plots_dir, model_name + f"_fold{fold_idx}",
                            train_losses, val_losses,
                            train_accs_isch, val_accs_isch,
                            train_f1_isch, val_f1_isch,
                            train_auc_isch, val_auc_isch)

        # 绘制心衰任务曲线
        plot_metrics_curves(plots_dir, model_name + f"_{field_xin}_fold{fold_idx}",
                            train_losses, val_losses,
                            train_accs_xin, val_accs_xin,
                            train_f1_xin, val_f1_xin,
                            train_auc_xin, val_auc_xin)

        # -------------------------------------------------------
        # 5. 记录这一折的详细成绩 (修改版：保存所有指标)
        # -------------------------------------------------------
        fold_result = {
            "fold": fold_idx,
            "best_avg_val_f1": best_val_metric,  # 这是历史最佳平均 F1
        }

        # 循环遍历两个任务 (Ischemia, xinshuai) 和五个指标
        # 自动生成 key，例如: "Ischemia_final_acc", "xinshuai_final_sens"
        for task_name in [field_isch, field_xin]:
            # val_metrics 结构: {task: {'acc': 0.8, 'sens': 0.7 ...}}
            for metric in ['acc', 'sens', 'spec', 'auc', 'f1']:
                # 获取最后一轮(Final Epoch)的指标值
                value = val_metrics[task_name][metric]

                # 存入字典，key 格式例如: Ischemia_final_auc
                fold_result[f"{task_name}_final_{metric}"] = value

        cv_summary.append(fold_result)

        # -------------------------------------------------------
        # 6. 【新增】生成错题本 (Error Analysis Report)
        # -------------------------------------------------------
        print("Generating error analysis report using BEST model...")

        # 1. 重新加载这一折效果最好的模型
        best_ckpt_path = os.path.join(ckpt_dir, "best_checkpoint.pth")
        load_checkpoint(best_ckpt_path, model, optimizer=None)  # 不需要加载优化器

        # 2. 运行分析
        error_csv_path = os.path.join(model_output_dir, f"error_analysis_fold{fold_idx}.csv")

        save_validation_results(
            model, val_loader, device, label_map,
            field_isch, field_xin,
            error_csv_path
        )

    # =========================================================
    # END FOLD LOOP
    # =========================================================

    # 保存 cv_summary 到主目录（如果是单折也会保存）
    summary_path = os.path.join(out_dir, model_name, "cv_summary.json")
    with open(summary_path, "w") as f:
        json.dump(cv_summary, f, indent=2)

    print("Training finished for all folds. CV summary saved to", summary_path)


if __name__ == "__main__":
    # 1. 读取 JSON 配置文件
    config_path = r"config/config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # 2. 调用主函数
    main(
        cfg['pickle_folder'],
        cfg['label_csv'],
        out_dir=cfg['out_dir'],
        adapter_mode=cfg['adapter_mode'],
        batch_size=cfg['batch_size'],
        epochs=cfg['epochs'],
        lr=cfg['lr'],
        num_workers=cfg['num_workers'],
        seed=cfg['seed'],
        resume_from=cfg['resume_from'],
        use_amp=cfg['use_amp'],
        model_name=cfg['model_name'],
        field_isch=cfg['field_isch'],
        field_xin=cfg['field_xin'],
        save_every_n_epochs=cfg['save_every_n_epochs']
    )
