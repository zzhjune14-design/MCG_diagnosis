# data_utils.py
from pathlib import Path
from typing import List, Dict, Optional, Union
import random
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# 定义公开接口
__all__ = [
    "FilesListDataset",
    "collate_fn_indexed",
    "load_label_map",
    "set_seed",
    "gather_pickle_files",
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


# -------------------------
# 辅助：从 batch raws 或 label_map 获取二分类标签（0/1/-1）
# -------------------------
def _normalize_binary_label_value(v) -> int:
    """
    将可能的输入值标准化为 0/1/-1：
      - 有效标签：1 / 0
      - 缺失标签：None / NaN -> -1
    """
    if v is None:
        return -1
    try:
        if isinstance(v, float) and np.isnan(v):
            return -1
    except Exception:
        pass
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "yes", "y", "true", "t", "有", "positive", "pos"):
            return 1
        if s in ("2", "0", "no", "n", "false", "f", "无", "negative", "neg"):
            return 0
        try:
            iv = int(s)
            return 1 if iv == 1 else 0
        except Exception:
            return -1
    if isinstance(v, (int, float, bool, np.integer, np.floating)):
        try:
            iv = int(v)
            return 1 if iv == 1 else 0
        except Exception:
            return -1
    return -1


def _get_binary_labels_from_raws_or_map(raws, subjects, field_name: str, label_map: dict):
    """
    返回 numpy array shape (B,) of ints (0/1/-1)
    -1 表示该样本无标签
    """
    B = len(subjects)
    labels = []

    def _try_get_from_raw(r, key):
        if not isinstance(r, dict):
            return None
        if key in r:
            return r.get(key)
        lk = key.lower()
        uk = key.upper()
        for k in (key, lk, uk):
            if k in r:
                return r.get(k)
        return None

    if isinstance(raws, (list, tuple)) and len(raws) > 0 and isinstance(raws[0], dict):
        example_val = _try_get_from_raw(raws[0], field_name)
        if example_val is not None:
            for r in raws:
                v = _try_get_from_raw(r, field_name)
                labels.append(_normalize_binary_label_value(v))
            return np.array(labels, dtype=int)

    for s in subjects:
        lm_entry = label_map.get(s, None)
        if lm_entry is None:
            labels.append(-1)
            continue
        if isinstance(lm_entry, dict):
            raw_v = lm_entry.get(field_name, lm_entry.get(field_name.lower(), -1))
        else:
            raw_v = lm_entry
        labels.append(_normalize_binary_label_value(raw_v))
    return np.array(labels, dtype=int)


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



