import os
import torch
import torch.nn as nn
from pathlib import Path
from data_process.data_utils import set_seed, build_dataloaders, gather_pickle_files
from models.CNN1D import CNN1D_from_amcg
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ---- 新增：checkpoint helpers ----
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

# ---- 保持你的 train/eval 函数原样或使用已有实现 ----
def train_epoch(model, loader, optimizer, loss_fn, device, label_map):
    model.train()
    running_loss = 0.0
    preds = []
    trues = []
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
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds)
    return running_loss / len(loader.dataset), acc, f1


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device, label_map):
    model.eval()
    running_loss = 0.0
    preds = []
    trues = []
    for Xb, subjects, raws in tqdm(loader, desc="eval", leave=False):
        Xb = Xb.to(device)
        y = torch.tensor([label_map[s] for s in subjects], dtype=torch.float32, device=device)
        logits = model(Xb)
        loss = loss_fn(logits, y)
        running_loss += loss.item() * Xb.size(0)
        preds.extend((torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int).tolist())
        trues.extend(y.cpu().numpy().astype(int).tolist())
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds)
    return running_loss / len(loader.dataset), acc, f1

# ---- 修改后的 main，增加 resume_from 参数，并使用上面的 save/load ----
def main(pickle_folder: str, label_csv: str, out_dir: str = "./ckpt",
         adapter_mode: str = "bn", batch_size: int = 8, epochs: int = 20,
         lr: float = 1e-3, num_workers: int = 4,
         seed: int = 42, resume_from: str = None, use_amp: bool = False):

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # --- call build_dataloaders and handle both possible return signatures ---
    dl_res = build_dataloaders(
        pickle_folder=pickle_folder,
        label_csv=label_csv,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
    )

    # dl_res may be either:
    #  - (train_loader, val_loader, label_map)  <-- old behavior
    #  - (dataloaders_per_fold, label_map)       <-- new CV behavior (dataloaders_per_fold is list of (train_loader,val_loader))
    if isinstance(dl_res, tuple) and len(dl_res) == 3:
        train_loader, val_loader, label_map = dl_res
        using_cv = False
    elif isinstance(dl_res, tuple) and len(dl_res) == 2:
        dataloaders_per_fold, label_map = dl_res
        if not isinstance(dataloaders_per_fold, (list, tuple)) or len(dataloaders_per_fold) == 0:
            raise RuntimeError("build_dataloaders returned unexpected folds structure")
        # default: pick fold 0 to be backward-compatible
        train_loader, val_loader = dataloaders_per_fold[0]
        using_cv = True
        print(f"[info] build_dataloaders returned {len(dataloaders_per_fold)} folds — using fold 0 by default.")
        print("If you want to train all folds, modify main.py to loop over dataloaders_per_fold.")
    else:
        raise RuntimeError("Unexpected return from build_dataloaders() - expected (train_loader,val_loader,label_map) or (dataloaders_per_fold,label_map)")

    files_all = gather_pickle_files(pickle_folder)
    print(f"Found {len(files_all)} pickle files, labels: {len(label_map)}")

    model = CNN1D_from_amcg(num_classes=1, adapter_mode=adapter_mode).to(device)
    print("Model params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    # optional AMP scaler
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    # resume logic
    start_epoch = 1
    best_val_f1 = -1.0
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from} ...")
        start_epoch, best_val_f1, ckpt_label_map = load_checkpoint(resume_from, model, optimizer, scaler)
        if ckpt_label_map is not None:
            # if label_map was saved in ckpt, prefer it (only if consistent with your current dataloaders)
            label_map = ckpt_label_map
        print(f" Resuming from epoch {start_epoch}, best_val_f1={best_val_f1}")

    # training loop
    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, loss_fn, device, label_map)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, loss_fn, device, label_map)
        print(f" train loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f}")
        print(f"  val  loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")

        # save last checkpoint every epoch
        last_path = os.path.join(out_dir, "last_checkpoint.pth")
        save_checkpoint(last_path, model, optimizer, epoch, best_val_f1, scaler=scaler, label_map=label_map)

        # if improved, save best checkpoint (both to "best_model.pth" and a timestamped copy if you like)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path = os.path.join(out_dir, "best_checkpoint.pth")
            save_checkpoint(best_path, model, optimizer, epoch, best_val_f1, scaler=scaler, label_map=label_map)
            print("  saved best_checkpoint.pth")

    # final save (optional)
    final_path = os.path.join(out_dir, "final_checkpoint.pth")
    save_checkpoint(final_path, model, optimizer, epoch, best_val_f1, scaler=scaler, label_map=label_map)
    print("Training finished. Best val F1:", best_val_f1)



if __name__ == "__main__":
    PICKLE_FOLDER = r"E:\Pythonpro\MCG_quexue_xinshuai\data_pickle"
    LABEL_CSV = r"E:\Pythonpro\MCG_quexue_xinshuai\label.csv"
    # 在脚本/命令行里
    main(PICKLE_FOLDER, LABEL_CSV, out_dir="./ckpt", adapter_mode="bn",
         batch_size=8, epochs=30, lr=1e-3, num_workers=4, seed=42,
         resume_from="./ckpt/last_checkpoint.pth", use_amp=False)
