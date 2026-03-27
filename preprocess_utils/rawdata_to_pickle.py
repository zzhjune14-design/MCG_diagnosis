import os
import numpy as np
import pandas as pd
import pickle
from scipy import signal  # 必须导入 signal 用于滤波

# ==========================================
# 1. 核心去噪函数 (严格复现论文 3.1 节方法)
# ==========================================
def apply_mcg_denoising(data, fs=1000.0):
    """
    严格按照论文方法进行去噪：
    Reference: "removing low- and high-frequency noise ... using a 1-45 Hz bandpass filter,
    followed by removing the industrial frequency noise using a 50 Hz trap filter"
    """
    # 确保数据格式为 float32
    clean_data = data.astype(np.float32).copy()

    # --- 步骤 A: 1-45 Hz 带通滤波 (论文核心参数) ---
    # 1Hz: 去除基线漂移 (呼吸等)
    # 45Hz: 强力去除高频噪声 (使得波形平滑)
    b_band, a_band = signal.butter(4, [1.0, 45.0], btype='bandpass', fs=fs)
    clean_data = signal.filtfilt(b_band, a_band, clean_data, axis=1)

    # --- 步骤 B: 50Hz 工频陷波 ---
    # 虽然 45Hz 的低通已经衰减了 50Hz，但为了保险起见，
    # 按照论文描述再次显式去除 50Hz 工频干扰
    b_notch, a_notch = signal.iirnotch(50.0, 30.0, fs)
    clean_data = signal.filtfilt(b_notch, a_notch, clean_data, axis=1)

    # --- 步骤 C: 去除直流分量 ---
    clean_data -= np.mean(clean_data, axis=1, keepdims=True)

    return clean_data


# ==========================================
# 2. 原有的读取工具函数 (保持不变)
# ==========================================
def tools_read_file(path, idx=1, interval=0):
    """
    读取指定文件夹中的多种数据
    """
    subfolder_list = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if idx < 1 or idx > len(subfolder_list):
        raise IndexError(f"子文件夹索引 {idx} 超出范围，文件夹总数为 {len(subfolder_list)}")

    data_idx = idx - 1
    subfolder_name = subfolder_list[data_idx]
    root_path = os.path.join(path, subfolder_name)

    raw_path = os.path.join(root_path, f'{subfolder_name}.baseDate') if os.path.exists(
        os.path.join(root_path, f'{subfolder_name}.baseDate')) else None
    ls_path = os.path.join(root_path, f'{subfolder_name}.LS')
    pk_path = os.path.join(root_path, f'{subfolder_name}.PK')
    bfd_path = os.path.join(root_path, f'{subfolder_name}.BFD')

    # 读取原始数据
    if raw_path:
        raw_data, total_time = read_rawdata(raw_path, interval)
    else:
        raw_data, total_time = None, 0

    # 读取 BFD 数据
    bfd_data = read_bfddata(bfd_path)

    # 读取 LS 数据
    ls_data = read_file_as_int(ls_path)

    # 读取 PK 数据
    pk_data = read_file_as_int(pk_path) if os.path.exists(pk_path) else 0

    return raw_data, ls_data, pk_data, bfd_data, total_time, root_path, subfolder_name


def read_rawdata(path, interval):
    with open(path, 'rb') as f:
        read_data = np.fromfile(f, dtype=np.float32)

    total_time = len(read_data) / 36000

    if interval == 0:
        time_start = 1
        time_step = total_time
        total_nums = round(time_step * 1000)
    elif isinstance(interval, (int, float)):
        if interval >= total_time:
            raise ValueError("超过读取时间范围")
        time_start = interval
        time_step = total_time - time_start + 1
        total_nums = round(time_step * 1000)
    elif isinstance(interval, list) and len(interval) == 2:
        if interval[0] > total_time:
            raise ValueError("读取起始点超过时间范围")
        elif sum(interval) > total_time:
            raise ValueError("读取终止点超过时间范围")
        time_start = interval[0]
        time_step = interval[1]
        total_nums = round(time_step * 1000)

    time_start = int(time_start)
    time_step = int(time_step)

    raw_data = np.zeros((36, total_nums))
    for i in range(36):
        for j in range(time_start, time_start + time_step):
            save_start = (j - time_start) * 1000
            save_end = (j - time_start + 1) * 1000

            read_start = (i) * 1000 + (j - 1) * 36000
            read_end = (i + 1) * 1000 + (j - 1) * 36000
            raw_data[i, save_start:save_end] = read_data[read_start:read_end]

    return raw_data, total_time


def read_bfddata(path):
    if not os.path.exists(path):
        return np.zeros((36, 1000))  # 防止报错

    with open(path, 'rb') as f:
        bfd = np.fromfile(f, dtype=np.float32)

    bfd_data = np.zeros((36, 1000))
    # 增加长度检查，防止文件损坏导致索引越界
    if len(bfd) < 36000:
        limit = len(bfd) // 1000
        for i in range(min(36, limit)):
            selected_start = (i) * 1000
            bfd_data[i, :] = bfd[selected_start:selected_start + 1000]
    else:
        for i in range(36):
            selected_start = (i) * 1000
            bfd_data[i, :] = bfd[selected_start:selected_start + 1000]

    return bfd_data


def read_file_as_int(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = np.fromfile(f, dtype=np.int32)
        return data
    return []


def save_data_to_pickle(data_dict, filename, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = os.path.join(save_folder, filename.replace('.csv', '.pkl'))

    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    # print(f"Pickle 数据已保存: {save_path}") # 减少打印刷屏


# ==========================================
# 3. 主程序逻辑
# ==========================================
if __name__ == "__main__":
    # 配置路径
    folder_path = r"E:\Pythonpro\MCG_quexue_xinshuai\data_raw\quexue801-986"
    save_folder = r'E:\Pythonpro\MCG_quexue_xinshuai\raw_data_pickle\sick'

    # 确保保存目录存在
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    subfolders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    print(f"开始处理 {len(subfolders)} 个文件夹...")
    print("应用论文预处理标准: 1-45Hz 带通 + 50Hz 陷波")

    for i, subfolder in enumerate(subfolders, 1):
        try:
            subfolder_path = os.path.join(folder_path, subfolder)

            # 1. 读取数据 (包含 raw_data, ls_data 等)
            raw_data, ls_data, pk_data, bfd_data, total_time, root_path, subfolder_name = tools_read_file(
                subfolder_path, idx=1, interval=0)

            # 如果 raw_data 读取失败，跳过
            if raw_data is None:
                print(f"跳过 {subfolder_name}: 缺少 baseDate 文件")
                continue

            # 2. 【关键步骤】执行去噪处理 (论文方法)
            # 只有 raw_data 需要去噪，bfd_data 是背景通常不动，或者用来做参考
            clean_raw_data = apply_mcg_denoising(raw_data, fs=1000.0)

            # 3. 封装字典 (保存的是干净数据 clean_raw_data)
            data_to_save = {
                'id': subfolder_name,
                'raw_data': clean_raw_data,  # 注意：这里保存的是去噪后的数据！
                'bfd_data': bfd_data,
                'ls_data': ls_data,
                'sampling_rate': 1000,
                'total_time': total_time
            }

            # 4. 保存
            pickle_filename = f'{subfolder_name}.pkl'
            save_data_to_pickle(data_to_save, pickle_filename, save_folder)

            print(f"[{i}/{len(subfolders)}] 完成: {subfolder_name} | 时长: {total_time:.1f}s | 状态: 已去噪并保存")

        except Exception as e:
            print(f"❌ 处理 {subfolder} 时出错: {e}")

    print("=" * 50)
    print(f"所有处理完成！干净的数据已保存在: {save_folder}")