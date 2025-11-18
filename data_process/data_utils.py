# data_utils.py
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

# 定义公开接口
__all__ = [
    "FilesListDataset",
    "collate_fn_indexed",
    "load_label_map",
    "set_seed",
    "gather_pickle_files",
    "build_dataloaders",
]


class FilesListDataset(Dataset):
    """加载指定列表的pickle文件，返回(data, subject)"""

    def __init__(self, files_list: List[Path]):
        self.files = files_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        with open(p, "rb") as f:
            data = pickle.load(f)
        try:
            subject = int(p.stem)
        except Exception:
            subject = idx + 1
        return data, subject


def collate_fn_indexed(batch):
    """
    Dataloader的数据处理函数
    batch: list of (raw_dict, subject)
    returns:
      X_tensor: (B,6,6,t_max) padded
      subjects: list[int]
      raws: list[dict]
    """
    raws, subjects = zip(*batch)
    processed = []
    max_t = 0
    for d in raws:
        amcg = np.asarray(d["amcg"])
        # normalize amcg shape to (6,6,t)
        if amcg.ndim == 3:
            if amcg.shape[0] == 6 and amcg.shape[1] == 6:
                arr = amcg
            elif amcg.shape[-1] == 6 and amcg.shape[-2] == 6:
                arr = np.transpose(amcg, (1, 2, 0))
            else:
                idx6 = [i for i, v in enumerate(amcg.shape) if v == 6]
                if len(idx6) >= 2:
                    other = [i for i in (0, 1, 2) if i not in idx6][0]
                    perm = (*idx6, other)
                    arr = np.transpose(amcg, perm)
                else:
                    raise ValueError(f"amcg shape {amcg.shape} not compatible")
        else:
            raise ValueError("amcg must be 3D")
        processed.append(arr.astype(np.float32))
        if arr.shape[2] > max_t:
            max_t = arr.shape[2]

    B = len(processed)
    X = np.zeros((B, 6, 6, max_t), dtype=np.float32)
    for i, arr in enumerate(processed):
        t = arr.shape[2]
        X[i, :, :, :t] = arr
    X_tensor = torch.tensor(X)  # (B,6,6,max_t)
    return X_tensor, list(subjects), list(raws)


def load_label_map(csv_path: str, subject_col: str = "subject", label_col: str = "Ischemia") -> Dict[int, int]:
    """标签映射加载函数 -> label (ints)."""
    df = pd.read_csv(csv_path)
    mapping = {int(r[subject_col]): int(r[label_col]) for _, r in df.iterrows()}
    return mapping


def set_seed(seed: int = 42):
    """固定随机种子函数"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gather_pickle_files(pickle_folder: str, exts: Optional[set] = None) -> List[Path]:
    """Pickle 文件路径收集函数"""
    folder = Path(pickle_folder)
    if exts is None:
        exts = {".pkl", ".pickle", ".dat"}
    files_all = sorted([p for p in folder.iterdir() if p.suffix in exts])
    return files_all


def build_dataloaders(pickle_folder: str,
                      label_csv: str,
                      batch_size: int = 8,
                      n_splits: int = 5,
                      seed: int = 42,
                      num_workers: int = 4,
                      pin_memory: bool = True,
                      shuffle_train: bool = True):

    label_map = load_label_map(label_csv, subject_col="subject", label_col="Ischemia")

    files_all = gather_pickle_files(pickle_folder)

    # 提取 subject ID
    subjects_all = []
    for p in files_all:
        try:
            subjects_all.append(int(p.stem))
        except Exception:
            subjects_all.append(None)

    # 选择有 label 的文件
    idxs_with_label = [i for i, s in enumerate(subjects_all) if s in label_map]
    if len(idxs_with_label) == 0:
        raise RuntimeError("No pickle filenames matched labels in CSV")

    subs = [subjects_all[i] for i in idxs_with_label]
    ys = [label_map[s] for s in subs]

    # --------- 关键：使用 StratifiedKFold ---------
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    dataloaders_per_fold = []

    for fold, (train_pos, val_pos) in enumerate(skf.split(subs, ys)):
        # train_pos / val_pos 是在 subs 中的下标
        train_idx = [idxs_with_label[i] for i in train_pos]
        val_idx   = [idxs_with_label[i] for i in val_pos]

        train_files = [files_all[i] for i in train_idx]
        val_files   = [files_all[i] for i in val_idx]

        train_ds = FilesListDataset(train_files)
        val_ds   = FilesListDataset(val_files)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train,
                                  collate_fn=collate_fn_indexed, num_workers=num_workers,
                                  pin_memory=pin_memory)

        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn_indexed, num_workers=num_workers,
                                pin_memory=pin_memory)

        dataloaders_per_fold.append((train_loader, val_loader))

    # 返回：长度为 n_splits 的列表，每个元素是 (train_loader, val_loader)
    return dataloaders_per_fold, label_map
