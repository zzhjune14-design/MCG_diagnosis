import torch
import os


# ================================================================
#  保存 checkpoint
# ================================================================
def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_f1: float,
    scaler: torch.cuda.amp.GradScaler = None,
    label_map: dict = None,
    extra: dict = None
):
    """
    保存训练 checkpoint，包含：
        - epoch
        - model weights
        - optimizer state
        - best_val_f1
        - amp scaler
        - label_map
        - RNG 状态（可选）
    """

    # ------- 确保目录存在 -------
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "best_val_f1": best_val_f1,
    }

    # 混合精度 scaler
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()

    # 标签映射字典
    if label_map is not None:
        ckpt["label_map"] = label_map

    # 额外信息
    if extra:
        ckpt.update(extra)

    # 保存 RNG 状态（复现性更高）
    ckpt["torch_rng_state"] = torch.get_rng_state()
    if torch.cuda.is_available():
        ckpt["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    torch.save(ckpt, path)
    print(f"[Checkpoint] Saved to: {path}")


# ================================================================
#  加载 checkpoint
# ================================================================
def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scaler: torch.cuda.amp.GradScaler = None,
    map_location=None
):
    """
    加载 checkpoint 到 model / optimizer / scaler。
    返回：
        start_epoch: 下次训练从第几轮开始
        best_val_f1: 最佳 F1
        label_map  : 类别标签字典
    """

    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(path, map_location=map_location)
    print(f"[Checkpoint] Loaded from: {path}")

    # ------------------ Model ------------------
    model.load_state_dict(ckpt["model_state_dict"])

    # ------------------ Optimizer ------------------
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"[Warning] Optimizer state load mismatch: {e}")

    # ------------------ AMP Scaler ------------------
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception as e:
            print(f"[Warning] AMP scaler load mismatch: {e}")

    # ------------------ RNG ------------------
    if ckpt.get("torch_rng_state") is not None:
        try:
            torch.set_rng_state(ckpt["torch_rng_state"])
            if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        except Exception as e:
            print(f"[Warning] RNG state restore failed: {e}")

    # ------------------ Return values ------------------
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_f1 = ckpt.get("best_val_f1", -1.0)
    label_map = ckpt.get("label_map", None)

    return start_epoch, best_val_f1, label_map
