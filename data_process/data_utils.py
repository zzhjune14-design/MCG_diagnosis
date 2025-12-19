# data_process/data_utils.py
from pathlib import Path
from typing import List, Dict, Optional, Union
import random
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = [
    "FilesListDataset",
    "collate_fn_indexed",
    "load_label_map",
    "set_seed",
    "gather_pickle_files",
    "_get_binary_labels_from_raws_or_map",  # 导出这个给 main.py 用
    "_normalize_binary_label_value"
]


# -----------------------
# 数据增强 (保持您之前的版本)
# -----------------------
def mcg_augment(data):
    # 1. 随机缩放
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.8, 1.2)
        data = data * scale

    # 2. 随机噪声
    if np.random.rand() < 0.5:
        sig_std = np.std(data)
        if sig_std == 0: sig_std = 1.0
        noise_level = np.random.uniform(0.01, 0.03)
        sigma = noise_level * sig_std
        noise = np.random.normal(0, sigma, data.shape)
        data = data + noise

    # 3. 随机平移
    if np.random.rand() < 0.3:
        max_shift = data.shape[-1] // 20
        shift = np.random.randint(-max_shift, max_shift)
        data = np.roll(data, shift, axis=-1)

    return data


# -----------------------
# Dataset
# -----------------------
class FilesListDataset(Dataset):
    def __init__(self, files_list: List[Path], augment: bool = False):
        self.files = files_list
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        try:
            with open(p, "rb") as f:
                data_dict = pickle.load(f)

            if 'amcg' not in data_dict:
                raise ValueError(f"Key 'amcg' not found in {p}")

            data = data_dict['amcg']
            if isinstance(data, np.ndarray):
                data = data.astype(np.float32)

            if self.augment:
                data = mcg_augment(data)
                data_dict['amcg'] = data

            try:
                subject = int(p.stem)
            except Exception:
                subject = idx + 1

            return data_dict, subject

        except Exception as e:
            print(f"Error loading {p}: {e}")
            raise e


# -----------------------
# Collate Function
# -----------------------
def collate_fn_indexed(batch):
    raws, subjects = zip(*batch)
    processed = []
    max_t = 0

    for d in raws:
        amcg = np.asarray(d["amcg"])
        if amcg.ndim == 3:
            if amcg.shape[0] == 6 and amcg.shape[1] == 6:
                arr = amcg
            elif amcg.shape[-1] == 6 and amcg.shape[-2] == 6:
                arr = np.transpose(amcg, (1, 2, 0))
            else:
                shape = amcg.shape
                idx6 = [i for i, v in enumerate(shape) if v == 6]
                if len(idx6) >= 2:
                    dims = (0, 1, 2)
                    other = [i for i in dims if i not in idx6][0]
                    perm = list(dims)
                    perm.remove(other)
                    perm.append(other)
                    arr = np.transpose(amcg, perm)
                else:
                    raise ValueError(f"amcg shape {amcg.shape} not compatible")
        elif amcg.ndim == 2 and amcg.shape[0] == 36:
            arr = amcg.reshape(6, 6, -1)
        else:
            raise ValueError(f"amcg must be 3D or (36, T), got {amcg.shape}")

        processed.append(arr.astype(np.float32))
        max_t = max(max_t, arr.shape[2])

    B = len(processed)
    X = np.zeros((B, 6, 6, max_t), dtype=np.float32)
    for i, arr in enumerate(processed):
        t = arr.shape[2]
        X[i, :, :, :t] = arr

    X_tensor = torch.tensor(X)
    return X_tensor, list(subjects), list(raws)


# -----------------------
# 【核心修改】标签解析函数
# -----------------------
def _normalize_binary_label_value(v) -> int:
    """
    解析标签值，确保：
    - 1 / 1.0 -> 1 (心衰/阳性)
    - 2 / 2.0 -> 0 (非心衰/阴性)
    - NaN / 空 -> -1 (忽略)
    """
    # 1. 处理空值
    if v is None:
        return -1
    try:
        if isinstance(v, float) and np.isnan(v):
            return -1
    except Exception:
        pass

    # 2. 转字符串统一处理 (处理 1.0, 2.0 这种情况)
    s = str(v).strip().lower()

    # --- 您的特定逻辑 ---
    if s in ('1', '1.0'):
        return 1  # 心衰 -> 1
    if s in ('2', '2.0'):
        return 0  # 非心衰 -> 0 (注意：这里必须映射为0，因为二分类模型输出0代表负类)

    if s == 'nan':
        return -1

    # --- 兼容其他情况 ---
    if s in ("yes", "y", "true", "t", "有", "positive", "pos"):
        return 1
    if s in ("0", "0.0", "no", "n", "false", "f", "无", "negative", "neg"):
        return 0

    return -1


def _get_binary_labels_from_raws_or_map(raws, subjects, field_name: str, label_map: dict):
    labels = []
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


def load_label_map(csv_path: str, subject_col: str = "subject", label_cols: Optional[Union[str, List[str]]] = None) -> \
Dict[int, dict]:
    df = pd.read_csv(csv_path)
    df.dropna(how='all', inplace=True)

    if subject_col not in df.columns:
        raise ValueError(f"subject_col '{subject_col}' not found")

    if label_cols is None:
        label_cols_list = [c for c in df.columns if c != subject_col]
    else:
        label_cols_list = [label_cols] if isinstance(label_cols, str) else list(label_cols)

    mapping: Dict[int, dict] = {}
    for _, row in df.iterrows():
        try:
            sid = int(row[subject_col])
            mapping[sid] = {}
            for col in label_cols_list:
                mapping[sid][col] = int(_normalize_binary_label_value(row[col]))
        except ValueError:
            continue
    return mapping


# -----------------------
# 工具函数
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gather_pickle_files(pickle_folder: str, exts: Optional[set] = None) -> List[Path]:
    folder = Path(pickle_folder)
    if exts is None: exts = {".pkl", ".pickle", ".dat"}
    files_all = sorted([p for p in folder.iterdir() if p.suffix in exts])
    return files_all