# build_data.py
from typing import List, Tuple, Dict, Optional
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader


from data_process.data_utils import (
    FilesListDataset, gather_pickle_files, load_label_map, collate_fn_indexed
)


def _make_fold_indices(idxs_with_label, subs, ys, n_splits=5, seed=42):
    """Return list of (train_idx, val_idx) where indices are indexes into files_all."""
    unique_vals = set(ys)
    fold_indices = []
    if len(unique_vals) <= 1:
        # fallback: deterministic random split
        rng = np.random.RandomState(seed)
        all_idxs = idxs_with_label.copy()
        rng.shuffle(all_idxs)
        n = len(all_idxs)
        base = n // n_splits
        extras = n % n_splits
        start = 0
        for f in range(n_splits):
            size = base + (1 if f < extras else 0)
            end = start + size
            val_chunk = all_idxs[start:end]
            train_chunk = [i for i in all_idxs if i not in val_chunk]
            fold_indices.append((train_chunk, val_chunk))
            start = end
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_pos, val_pos in skf.split(subs, ys):
            train_idx = [idxs_with_label[i] for i in train_pos]
            val_idx = [idxs_with_label[i] for i in val_pos]
            fold_indices.append((train_idx, val_idx))
    return fold_indices


def build_dataloaders(pickle_folder: str,
                                label_csv: str,
                                batch_size: int = 8,
                                n_splits: int = 5,
                                seed: int = 42,
                                num_workers: int = 4,
                                pin_memory: bool = True,
                                shuffle_train: bool = True,
                                subject_col: str = "subject",
                                label_cols = None,
                                stratify_by: Optional[str] = None,
                                save_split_folder: Optional[str] = None
                                ) -> Tuple[List[Tuple[DataLoader, DataLoader]], Dict[int, dict]]:
    """
    Build dataloaders per fold. Returns (dataloaders_per_fold, label_map).
    - save_split_folder: if provided, will write JSON files per fold listing train/val subject ids.
    """
    label_map = load_label_map(label_csv, subject_col=subject_col, label_cols=label_cols)
    files_all = gather_pickle_files(pickle_folder)

    # build subjects list
    subjects_all = []
    for p in files_all:
        try:
            subjects_all.append(int(p.stem))
        except Exception:
            subjects_all.append(None)

    idxs_with_label = [i for i, s in enumerate(subjects_all) if s in label_map]
    if len(idxs_with_label) == 0:
        raise RuntimeError("No pickle filenames matched labels in CSV")

    subs = [subjects_all[i] for i in idxs_with_label]

    # choose stratify key
    sample_label_keys = list(next(iter(label_map.values())).keys())
    if stratify_by is None:
        stratify_key = sample_label_keys[0]
    else:
        stratify_key = stratify_by
    if stratify_key not in sample_label_keys:
        print(f"[warn] stratify_by='{stratify_key}' not found. Falling back to '{sample_label_keys[0]}'.")
        stratify_key = sample_label_keys[0]

    ys = [label_map[s][stratify_key] for s in subs]

    fold_indices = _make_fold_indices(idxs_with_label, subs, ys, n_splits=n_splits, seed=seed)

    dataloaders_per_fold = []
    os.makedirs(save_split_folder, exist_ok=True) if save_split_folder else None

    for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
        train_files = [files_all[i] for i in train_idx]
        val_files = [files_all[i] for i in val_idx]

        # ================= 修改这里 =================
        # 训练集：augment=True (开启随机噪声、缩放等)
        train_ds = FilesListDataset(train_files, augment=True)

        # 验证集：augment=False (保持原汁原味，用于评估真实性能)
        val_ds = FilesListDataset(val_files, augment=False)
        # ===========================================

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train,
                                  collate_fn=collate_fn_indexed, num_workers=num_workers,
                                  pin_memory=pin_memory)

        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn_indexed, num_workers=num_workers,
                                pin_memory=pin_memory)

        dataloaders_per_fold.append((train_loader, val_loader))

        # optionally save split info (subject ids) for reproducibility
        if save_split_folder:
            train_subs = [int(p.stem) for p in train_files]
            val_subs = [int(p.stem) for p in val_files]
            with open(os.path.join(save_split_folder, f"fold_{fold_idx:02d}_split.json"), "w", encoding="utf-8") as fh:
                json.dump({"train": train_subs, "val": val_subs, "stratify_key": stratify_key}, fh, indent=2)

    return dataloaders_per_fold, label_map
