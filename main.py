import torch
import torch.nn as nn
from utils.save_result import *
from data_process.data_utils import set_seed, build_dataloaders, gather_pickle_files
from utils.checkpoint import save_checkpoint, load_checkpoint
from models.CNN1D import CNN1D_from_amcg
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc
from tqdm import tqdm

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
