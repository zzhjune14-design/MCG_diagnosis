import matplotlib.pyplot as plt
import os
import csv

# ---- 保存训练过程的指标和绘制曲线 ----
def plot_metrics(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s, train_aucs, val_aucs, output_dir,
                 model_name):
    epochs = range(1, len(train_losses) + 1)

    # Loss curve
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_name}_loss_curve.png"))
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - Training and Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_name}_accuracy_curve.png"))
    plt.close()

    # F1 Score curve
    plt.figure()
    plt.plot(epochs, train_f1s, label="Train F1")
    plt.plot(epochs, val_f1s, label="Validation F1")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title(f"{model_name} - Training and Validation F1 Score")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_name}_f1_curve.png"))
    plt.close()

    # AUC curve
    plt.figure()
    plt.plot(epochs, train_aucs, label="Train AUC")
    plt.plot(epochs, val_aucs, label="Validation AUC")
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.title(f"{model_name} - Training and Validation AUC")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_name}_auc_curve.png"))
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, output_dir, model_name, epoch):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve - Epoch {epoch}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curve_epoch_{epoch}.png"))
    plt.close()


def save_metrics_to_csv(metrics, output_dir, model_name):
    filepath = os.path.join(output_dir, f"{model_name}_metrics.csv")
    fieldnames = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_f1', 'val_f1', 'train_auc',
                  'val_auc']

    # Write header only if the file is new
    file_exists = os.path.exists(filepath)

    with open(filepath, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Write header only once

        # Ensure metrics is a dictionary
        if isinstance(metrics, dict):
            writer.writerow(metrics)
        else:
            print("Error: Expected metrics to be a dictionary, but received:", type(metrics))

