# utils/save_result.py
import os
import json
import csv
from typing import Dict, List, Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# =========================
# 通用工具
# =========================
def ensure_dir(path: str):
    """确保目录存在（递归创建）"""
    if path is None:
        return
    os.makedirs(path, exist_ok=True)


# =========================
# 保存/追加 metrics 到 CSV（动态 header）
# =========================
def append_metrics_csv(output_dir: str, model_name: str, metrics_row: Dict):
    """
    将一行指标追加到 CSV 文件。
    - output_dir: 目标目录
    - model_name: 用于命名文件 e.g. {model_name}_metrics.csv
    - metrics_row: 字典，键为列名，值为数字或可被 str() 的对象
    动作是：若文件不存在，写 header；否则追加。
    """
    ensure_dir(output_dir)
    filepath = os.path.join(output_dir, f"{model_name}_metrics.csv")
    file_exists = os.path.exists(filepath)

    # 尽量把 numpy types 转为原生 python types，避免 csv 写入错误
    cleaned = {}
    for k, v in metrics_row.items():
        if isinstance(v, (np.integer,)):
            cleaned[k] = int(v)
        elif isinstance(v, (np.floating,)):
            cleaned[k] = float(v)
        else:
            cleaned[k] = v

    # 写入 CSV（动态列顺序以 metrics_row.keys() 为准）
    with open(filepath, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cleaned.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(cleaned)


# =========================
# 绘制并保存训练曲线（loss/acc/f1/auc）
# =========================
def plot_metrics_curves(output_dir: str, model_name: str,
                        train_losses: Sequence[float], val_losses: Sequence[float],
                        train_accs: Sequence[float], val_accs: Sequence[float],
                        train_f1s: Sequence[float], val_f1s: Sequence[float],
                        train_aucs: Sequence[float], val_aucs: Sequence[float],
                        dpi: int = 120):
    """
    保存四张图：loss/accuracy/f1/auc。
    文件名格式：{model_name}_loss_curve.png / _accuracy_curve.png / _f1_curve.png / _auc_curve.png
    """
    ensure_dir(output_dir)
    epochs = list(range(1, len(train_losses) + 1))

    def _safe_plot(x, y_list, labels, title, ylabel, out_name):
        plt.figure(figsize=(8, 6), dpi=dpi)
        for y, lbl in zip(y_list, labels):
            plt.plot(epochs, y, label=lbl, linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{model_name} - {title}")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, out_name)
        plt.savefig(path)
        plt.close()

    # Loss
    _safe_plot(epochs, [train_losses, val_losses], ["Train Loss", "Val Loss"], "Loss", "Loss", f"{model_name}_loss_curve.png")
    # Accuracy
    _safe_plot(epochs, [train_accs, val_accs], ["Train Acc", "Val Acc"], "Accuracy", "Accuracy", f"{model_name}_accuracy_curve.png")
    # F1
    _safe_plot(epochs, [train_f1s, val_f1s], ["Train F1", "Val F1"], "F1 Score", "F1", f"{model_name}_f1_curve.png")
    # AUC
    _safe_plot(epochs, [train_aucs, val_aucs], ["Train AUC", "Val AUC"], "AUC", "AUC", f"{model_name}_auc_curve.png")


# =========================
# 绘制并保存 ROC 曲线
# =========================
def save_roc_plot(output_dir: str, model_name: str, fpr: Sequence[float], tpr: Sequence[float],
                  roc_auc_val: float, epoch: Optional[int] = None, dpi: int = 120):
    """
    保存 ROC 曲线图片。
    文件名: {model_name}_roc_epoch_{epoch}.png 或 {model_name}_roc.png（若 epoch 为 None）
    """
    ensure_dir(output_dir)
    plt.figure(figsize=(7, 6), dpi=dpi)
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC={roc_auc_val:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    title = f"{model_name} - ROC"
    if epoch is not None:
        title += f" (Epoch {epoch})"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="lower right")
    fname = f"{model_name}_roc_epoch_{epoch}.png" if epoch is not None else f"{model_name}_roc.png"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()


# =========================
# 绘制并保存混淆矩阵
# =========================
def save_confusion_matrix(output_dir: str, model_name: str, trues: Sequence[int], preds: Sequence[int],
                          epoch: Optional[int] = None, labels: Optional[List[str]] = None, dpi: int = 120):
    """
    保存混淆矩阵热力图。
    - labels: 可指定类别名列表，例如 ['neg','pos']，否则用 [0,1]
    - 文件名: {model_name}_confusion_epoch_{epoch}.png
    """
    ensure_dir(output_dir)
    if labels is None:
        labels = [0, 1]

    cm = confusion_matrix(trues, preds, labels=[0, 1])
    fig = plt.figure(figsize=(4, 4), dpi=dpi)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{model_name} - Confusion Matrix" + (f" (Epoch {epoch})" if epoch is not None else ""))
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    # 在每个格子里写数字
    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(int(cm[i, j]), 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fname = f"{model_name}_confusion_epoch_{epoch}.png" if epoch is not None else f"{model_name}_confusion.png"
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()


# =========================
# 保存运行元信息（metadata）
# =========================
def save_run_metadata(output_dir: str, metadata: Dict):
    """
    将训练运行的若干参数等写入 JSON 文件，便于追踪。
    """
    ensure_dir(output_dir)
    path = os.path.join(output_dir, "run_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# =========================
# 兼容旧函数名（如果其它代码使用旧名）
# =========================
# 将你的旧函数名映射到新实现，方便替换到现有 main.py
def plot_metrics(*args, **kwargs):
    return plot_metrics_curves(*args, **kwargs)


def plot_roc_curve(fpr, tpr, roc_auc, output_dir, model_name, epoch):
    return save_roc_plot(output_dir, model_name, fpr, tpr, roc_auc, epoch)
