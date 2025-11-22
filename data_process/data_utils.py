# data_utils.py
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import random
import pickle
import json

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

# -----------------------
# Dataset / collate
# -----------------------
class FilesListDataset(Dataset):
    """加载指定列表的pickle文件，返回 (data_dict, subject_id)"""
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
    Dataloader 的 collate 函数
    batch: list of (raw_dict, subject)
    返回:
      X_tensor: (B,6,6,t_max) padded
      subjects: list[int]
      raws: list[dict]
    注意：raws 保留原始字典，便于在训练脚本中取多个标签字段
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


# -----------------------
# 标签规范化工具
# -----------------------
def _normalize_binary_label_value(v) -> int:
    """
    将 CSV / raws 中的标签值统一为 0/1。
    约定（可扩展）：
      - 标识阳性/有病的：1, "1", "yes", "y", "true", "t", "有", "positive", "pos" -> 1
      - 标识阴性/无病的：2, "2", "0", "no", "n", "false", "f", "无", "negative", "neg" -> 0
    对于数字：1->1，其他数字（例如2）->0。
    若为 NaN/None/无法识别 -> 默认为 0。
    """
    if v is None:
        return 0
    # float nan
    try:
        if isinstance(v, float) and np.isnan(v):
            return 0
    except Exception:
        pass
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "yes", "y", "true", "t", "有", "hf", "positive", "pos"):
            return 1
        if s in ("2", "0", "no", "n", "false", "f", "无", "negative", "neg"):
            return 0
        try:
            iv = int(s)
            return 1 if iv == 1 else 0
        except Exception:
            return 0
    if isinstance(v, (int, float, bool, np.integer, np.floating)):
        try:
            iv = int(v)
            return 1 if iv == 1 else 0
        except Exception:
            return 0
    return 0


# -----------------------
# 加载 CSV -> label_map (subject -> dict of labels)
# -----------------------
def load_label_map(csv_path: str, subject_col: str = "subject", label_cols: Optional[Union[str, List[str]]] = None) -> Dict[int, dict]:
    """
    读取 CSV 并返回 mapping: subject_id -> {label_col1: value1, label_col2: value2, ...}
    - subject_col: CSV 中表示 subject id 的列名
    - label_cols: None -> 读取 CSV 中除 subject_col 外的所有列作为标签
                  如果是字符串 -> 读取单列；如果是列表 -> 按列表读取多个列
    返回值：每个 label 都用 _normalize_binary_label_value 规范化为 0/1（适用于二分类标签）。
    """
    df = pd.read_csv(csv_path)
    if subject_col not in df.columns:
        raise ValueError(f"subject_col '{subject_col}' not found in CSV columns: {df.columns.tolist()}")

    # 选择标签列
    if label_cols is None:
        label_cols_list = [c for c in df.columns if c != subject_col]
    else:
        if isinstance(label_cols, str):
            label_cols_list = [label_cols]
        else:
            label_cols_list = list(label_cols)
        for c in label_cols_list:
            if c not in df.columns:
                raise ValueError(f"label column '{c}' not found in CSV columns")

    mapping: Dict[int, dict] = {}
    for _, row in df.iterrows():
        sid = int(row[subject_col])
        mapping[sid] = {}
        for col in label_cols_list:
            raw_v = row[col]
            normalized = _normalize_binary_label_value(raw_v)
            mapping[sid][col] = int(normalized)
    return mapping


# -----------------------
# 其它工具
# -----------------------
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


# -----------------------
# 构造 dataloaders（支持分层交叉验证）
# -----------------------
def build_dataloaders(pickle_folder: str,
                      label_csv: str,
                      batch_size: int = 8,
                      n_splits: int = 5,
                      seed: int = 42,
                      num_workers: int = 4,
                      pin_memory: bool = True,
                      shuffle_train: bool = True,
                      subject_col: str = "subject",
                      label_cols: Optional[Union[str, List[str]]] = None,
                      stratify_by: Optional[str] = None):
    """
    构造 dataloaders 并返回 (dataloaders_per_fold, label_map)
    - label_cols: 指定要读取的 CSV 标签列（默认读取除 subject 列外的所有列）
    - stratify_by: 在 StratifiedKFold 中使用的标签列名（默认使用 label_cols 的第一个）
    返回:
      dataloaders_per_fold: list of (train_loader, val_loader)
      label_map: dict subject -> {label_col: 0/1, ...}
    """
    # 读取标签（subject -> dict）
    label_map = load_label_map(label_csv, subject_col=subject_col, label_cols=label_cols)

    files_all = gather_pickle_files(pickle_folder)

    # 提取 subject ID（按文件名 stem）
    subjects_all = []
    for p in files_all:
        try:
            subjects_all.append(int(p.stem))
        except Exception:
            subjects_all.append(None)

    # 保留那些在 label_map 中有标签的文件索引
    idxs_with_label = [i for i, s in enumerate(subjects_all) if s in label_map]
    if len(idxs_with_label) == 0:
        raise RuntimeError("No pickle filenames matched labels in CSV")

    subs = [subjects_all[i] for i in idxs_with_label]

    # 选择 stratify key
    sample_label_keys = list(next(iter(label_map.values())).keys())
    if stratify_by is None:
        stratify_key = sample_label_keys[0]
    else:
        stratify_key = stratify_by
    if stratify_key not in sample_label_keys:
        print(f"[warn] stratify_by='{stratify_key}' not found among label columns {sample_label_keys}. Falling back to first key.")
        stratify_key = sample_label_keys[0]

    # 构造 ys（用于 StratifiedKFold）
    ys = [label_map[s][stratify_key] for s in subs]

    # 如果 ys 中只有单一类别，Skf 会失败 -> 降级为随机切分
    unique_vals = set(ys)
    fold_indices = []
    if len(unique_vals) <= 1:
        print(f"[warn] stratify_by column '{stratify_key}' has only one unique value ({unique_vals}). Falling back to random splits.")
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

    dataloaders_per_fold = []
    for (train_idx, val_idx) in fold_indices:
        train_files = [files_all[i] for i in train_idx]
        val_files = [files_all[i] for i in val_idx]

        train_ds = FilesListDataset(train_files)
        val_ds = FilesListDataset(val_files)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train,
                                  collate_fn=collate_fn_indexed, num_workers=num_workers,
                                  pin_memory=pin_memory)

        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn_indexed, num_workers=num_workers,
                                pin_memory=pin_memory)

        dataloaders_per_fold.append((train_loader, val_loader))

    return dataloaders_per_fold, label_map
