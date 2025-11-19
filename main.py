import os
import torch
import torch.nn as nn
from pathlib import Path
from data_process.data_utils import set_seed, build_dataloaders, gather_pickle_files
from models.CNN1D import CNN1D_from_amcg
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv



# ---- 断点继续训练 ----
def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, best_val_f1: float, scaler: torch.cuda.amp.GradScaler = None,
                    label_map: dict = None, extra: dict = None):
    """
    Save a training checkpoint dict that includes model, optimizer, epoch, best metric, scaler, label_map, etc.
    """
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'best_val_f1': best_val_f1,
    }
    if scaler is not None:
        ckpt['scaler_state_dict'] = scaler.state_dict()
    if label_map is not None:
        ckpt['label_map'] = label_map
    if extra:
        ckpt.update(extra)
    # save RNG state for more exact reproducibility (optional)
    ckpt['torch_rng_state'] = torch.get_rng_state()
    if torch.cuda.is_available():
        ckpt['cuda_rng_state'] = torch.cuda.get_rng_state_all()
    torch.save(ckpt, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None,
                    scaler: torch.cuda.amp.GradScaler = None, map_location=None):
    """
    Load checkpoint into model (and optimizer/scaler if provided).
    Returns: start_epoch (next epoch to run), best_val_f1, label_map (if present)
    """
    if map_location is None:
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(path, map_location=map_location)

    # Load model weights
    model.load_state_dict(ckpt['model_state_dict'])

    # Load optimizer if available in ckpt and optimizer provided
    if optimizer is not None and ckpt.get('optimizer_state_dict') is not None:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception as e:
            # sometimes optimizer keys mismatch if optimizer changed; still warn and continue
            print(f"Warning: failed to load optimizer state_dict cleanly: {e}")

    # Load scaler if present
    if scaler is not None and ckpt.get('scaler_state_dict') is not None:
        try:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        except Exception as e:
            print(f"Warning: failed to load scaler state_dict: {e}")

    # restore RNG states if you want exact reproducibility
    if ckpt.get('torch_rng_state') is not None:
        try:
            torch.set_rng_state(ckpt['torch_rng_state'])
            if torch.cuda.is_available() and ckpt.get('cuda_rng_state') is not None:
                torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
        except Exception as e:
            print(f"Warning: failed to restore RNG state: {e}")

    start_epoch = ckpt.get('epoch', 0) + 1  # resume at next epoch
    best_val_f1 = ckpt.get('best_val_f1', -1.0)
    label_map = ckpt.get('label_map', None)
    return start_epoch, best_val_f1, label_map


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


# ---- 训练和评估函数 ----
def train_epoch(model, loader, optimizer, loss_fn, device, label_map):
    model.train()
    running_loss = 0.0
    preds = []
    trues = []
    all_logits = []  # Store logits for AUC calculation
    for Xb, subjects, raws in tqdm(loader, desc="train", leave=False):
        Xb = Xb.to(device)
        y = torch.tensor([label_map[s] for s in subjects], dtype=torch.float32, device=device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * Xb.size(0)

        preds.extend((torch.sigmoid(logits).detach().cpu().numpy() > 0.5).astype(int).tolist())
        trues.extend(y.detach().cpu().numpy().astype(int).tolist())
        all_logits.extend(torch.sigmoid(logits).detach().cpu().numpy())  # Store probabilities

    # 计算二分类指标
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds)

    # 计算灵敏度、特异性和AUC
    tp = sum((p == 1) and (t == 1) for p, t in zip(preds, trues))
    tn = sum((p == 0) and (t == 0) for p, t in zip(preds, trues))
    fp = sum((p == 1) and (t == 0) for p, t in zip(preds, trues))
    fn = sum((p == 0) and (t == 1) for p, t in zip(preds, trues))

    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    auc = roc_auc_score(trues, all_logits)  # AUC using the predicted probabilities

    return running_loss / len(loader.dataset), acc, f1, sensitivity, specificity, auc, all_logits, trues


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device, label_map):
    model.eval()
    running_loss = 0.0
    preds = []
    trues = []
    all_logits = []  # Store logits for AUC calculation
    for Xb, subjects, raws in tqdm(loader, desc="eval", leave=False):
        Xb = Xb.to(device)
        y = torch.tensor([label_map[s] for s in subjects], dtype=torch.float32, device=device)
        logits = model(Xb)
        loss = loss_fn(logits, y)
        running_loss += loss.item() * Xb.size(0)

        preds.extend((torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int).tolist())
        trues.extend(y.cpu().numpy().astype(int).tolist())
        all_logits.extend(torch.sigmoid(logits).cpu().numpy())  # Store probabilities

    # 计算二分类指标
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds)

    # 计算灵敏度、特异性和AUC
    tp = sum((p == 1) and (t == 1) for p, t in zip(preds, trues))
    tn = sum((p == 0) and (t == 0) for p, t in zip(preds, trues))
    fp = sum((p == 1) and (t == 0) for p, t in zip(preds, trues))
    fn = sum((p == 0) and (t == 1) for p, t in zip(preds, trues))

    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    auc = roc_auc_score(trues, all_logits)  # AUC using the predicted probabilities

    return running_loss / len(loader.dataset), acc, f1, sensitivity, specificity, auc, all_logits, trues


# ---- 修改后的 main ----
def main(pickle_folder: str, label_csv: str, out_dir: str = "./output",
         adapter_mode: str = "bn", batch_size: int = 8, epochs: int = 20,
         lr: float = 1e-3, num_workers: int = 4,
         seed: int = 42, resume_from: str = None, use_amp: bool = False, model_name="CNN1D"):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_output_dir = os.path.join(out_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # --- call build_dataloaders and handle both possible return signatures ---
    dl_res = build_dataloaders(
        pickle_folder=pickle_folder,
        label_csv=label_csv,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
    )

    if isinstance(dl_res, tuple) and len(dl_res) == 3:
        train_loader, val_loader, label_map = dl_res
        using_cv = False
    elif isinstance(dl_res, tuple) and len(dl_res) == 2:
        dataloaders_per_fold, label_map = dl_res
        if not isinstance(dataloaders_per_fold, (list, tuple)) or len(dataloaders_per_fold) == 0:
            raise RuntimeError("build_dataloaders returned unexpected folds structure")
        train_loader, val_loader = dataloaders_per_fold[0]
        using_cv = True

    files_all = gather_pickle_files(pickle_folder)
    print(f"Found {len(files_all)} pickle files, labels: {len(label_map)}")

    model = CNN1D_from_amcg(num_classes=1, adapter_mode=adapter_mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    # resume logic
    start_epoch = 1
    best_val_f1 = -1.0
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from} ...")
        start_epoch, best_val_f1, ckpt_label_map = load_checkpoint(resume_from, model, optimizer, scaler)
        if ckpt_label_map is not None:
            label_map = ckpt_label_map
        print(f" Resuming from epoch {start_epoch}, best_val_f1={best_val_f1}")

    # training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_f1s = []
    val_f1s = []
    train_aucs = []
    val_aucs = []

    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc, train_f1, train_sens, train_spec, train_auc, _, _ = train_epoch(model, train_loader,
                                                                                               optimizer, loss_fn,
                                                                                               device, label_map)
        val_loss, val_acc, val_f1, val_sens, val_spec, val_auc, val_logits, val_trues = eval_epoch(model, val_loader,
                                                                                                   loss_fn, device,
                                                                                                   label_map)

        print(f" train loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} auc={train_auc:.4f}")
        print(f"  val  loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} auc={val_auc:.4f}")

        # Save metrics for later plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        # Save metrics to CSV
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'train_auc': train_auc,
            'val_auc': val_auc
        }
        save_metrics_to_csv(metrics, model_output_dir, model_name)

        # Save last checkpoint
        last_path = os.path.join(model_output_dir, "last_checkpoint.pth")
        save_checkpoint(last_path, model, optimizer, epoch, best_val_f1, scaler=scaler, label_map=label_map)

        # Save best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path = os.path.join(model_output_dir, "best_checkpoint.pth")
            save_checkpoint(best_path, model, optimizer, epoch, best_val_f1, scaler=scaler, label_map=label_map)
            print("  saved best_checkpoint.pth")

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(val_trues, val_logits)
        roc_auc = auc(fpr, tpr)
        plot_roc_curve(fpr, tpr, roc_auc, model_output_dir, model_name, epoch)

    # Final save
    final_path = os.path.join(model_output_dir, "final_checkpoint.pth")
    save_checkpoint(final_path, model, optimizer, epoch, best_val_f1, scaler=scaler, label_map=label_map)

    # Plot and save metrics curves
    plot_metrics(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s, train_aucs, val_aucs,
                 model_output_dir, model_name)

    print("Training finished. Best val F1:", best_val_f1)


if __name__ == "__main__":
    PICKLE_FOLDER = r"E:\Pythonpro\MCG_quexue_xinshuai\data_pickle"
    LABEL_CSV = r"E:\Pythonpro\MCG_quexue_xinshuai\label.csv"
    main(PICKLE_FOLDER, LABEL_CSV, out_dir="./output", adapter_mode="bn",
         batch_size=8, epochs=30, lr=1e-3, num_workers=4, seed=42,
         resume_from="./output/CNN1D/last_checkpoint.pth", use_amp=False, model_name="CNN1D")
