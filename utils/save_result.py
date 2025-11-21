import matplotlib.pyplot as plt
import os
import csv

# ================================================
#  通用工具：确保目录存在
# ================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ================================================
#  绘制训练过程曲线
# ================================================
def plot_metrics(
        train_losses, val_losses,
        train_accs, val_accs,
        train_f1s, val_f1s,
        train_aucs, val_aucs,
        output_dir, model_name
):
    ensure_dir(output_dir)
    epochs = range(1, len(train_losses) + 1)

    # ---- 公用：画图风格 ----
    def _plot(y1, y2, label1, label2, title, ylabel, filename):
        plt.figure(figsize=(8, 6), dpi=120)
        plt.plot(epochs, y1, label=label1, linewidth=2)
        plt.plot(epochs, y2, label=label2, linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.title(f"{model_name} - {title}")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    # Loss
    _plot(train_losses, val_losses,
          "Train Loss", "Validation Loss",
          "Training and Validation Loss",
          "Loss",
          f"{model_name}_loss_curve.png")

    # Accuracy
    _plot(train_accs, val_accs,
          "Train Accuracy", "Validation Accuracy",
          "Training and Validation Accuracy",
          "Accuracy",
          f"{model_name}_accuracy_curve.png")

    # F1 Score
    _plot(train_f1s, val_f1s,
          "Train F1", "Validation F1",
          "Training and Validation F1 Score",
          "F1 Score",
          f"{model_name}_f1_curve.png")

    # AUC
    _plot(train_aucs, val_aucs,
          "Train AUC", "Validation AUC",
          "Training and Validation AUC",
          "AUC",
          f"{model_name}_auc_curve.png")


# ================================================
#  绘制 ROC 曲线
# ================================================
def plot_roc_curve(fpr, tpr, roc_auc, output_dir, model_name, epoch):
    ensure_dir(output_dir)

    plt.figure(figsize=(7, 6), dpi=120)
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve (Epoch {epoch})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                             f"{model_name}_roc_epoch_{epoch}.png"))
    plt.close()


# ================================================
#  保存指标到 CSV
# ================================================
def save_metrics_to_csv(metrics, output_dir, model_name):
    ensure_dir(output_dir)

    filepath = os.path.join(output_dir, f"{model_name}_metrics.csv")
    fieldnames = [
        "epoch",
        "train_loss", "val_loss",
        "train_acc", "val_acc",
        "train_f1", "val_f1",
        "train_auc", "val_auc"
    ]

    file_exists = os.path.exists(filepath)

    # 安全转换为 Python float
    cleaned_metrics = {k: float(v) if isinstance(v, (float, int)) else v
                       for k, v in metrics.items()}

    with open(filepath, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(cleaned_metrics)
