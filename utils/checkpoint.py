import torch


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