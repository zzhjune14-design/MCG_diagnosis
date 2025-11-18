import os
import numpy as np
import pandas as pd


# ===================读取做格式转换========================
def tools_read_file(path, idx=1, interval=0):
    """
    读取指定文件夹中的多种数据
    :param path: 主文件夹路径
    :param idx: 子文件夹索引，从 1 开始
    :param interval: 时间间隔，支持单个数值或区间
    :return: 读取的多种数据
    """
    # 获取文件夹中的所有子文件夹
    subfolder_list = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

    if idx < 1 or idx > len(subfolder_list):
        raise IndexError(f"子文件夹索引 {idx} 超出范围，文件夹总数为 {len(subfolder_list)}")

    # 文件夹索引位置
    data_idx = idx - 1  # 转换为0索引
    subfolder_name = subfolder_list[data_idx]
    root_path = os.path.join(path, subfolder_name)

    raw_path = os.path.join(root_path, f'{subfolder_name}.baseDate') if os.path.exists(os.path.join(root_path, f'{subfolder_name}.baseDate')) else None

    ls_path = os.path.join(root_path, f'{subfolder_name}.LS')
    pk_path = os.path.join(root_path, f'{subfolder_name}.PK')
    bfd_path = os.path.join(root_path, f'{subfolder_name}.BFD')

    # 读取原始数据
    if raw_path:
        raw_data, total_time = read_rawdata(raw_path, interval)
    else:
        raw_data, total_time = None, 0  # 默认值

    # 读取 BFD 数据
    bfd_data = read_bfddata(bfd_path)

    # 读取 LS 数据
    ls_data = read_file_as_int(ls_path)

    # 读取 PK 数据
    pk_data = read_file_as_int(pk_path) if os.path.exists(pk_path) else 0

    return raw_data, ls_data, pk_data, bfd_data, total_time, root_path, subfolder_name


# ===================读取basedata文件========================
def read_rawdata(path, interval):
    """
    读取原始数据
    :param path: 数据文件路径
    :param interval: 时间间隔，支持单个数值或区间
    :return: 原始数据数组，总时间
    """
    with open(path, 'rb') as f:
        read_data = np.fromfile(f, dtype=np.float32)

    total_time = len(read_data) / 36000

    # 处理 interval 参数
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

    # 将 time_start 和 time_step 转为整数
    time_start = int(time_start)
    time_step = int(time_step)

    # 读取原始数据并调整维度
    raw_data = np.zeros((36, total_nums))
    for i in range(36):
        for j in range(time_start, time_start + time_step):
            save_start = (j - time_start) * 1000
            save_end = (j - time_start + 1) * 1000

            read_start = (i) * 1000 + (j - 1) * 36000
            read_end = (i + 1) * 1000 + (j - 1) * 36000
            raw_data[i, save_start:save_end] = read_data[read_start:read_end]

    return raw_data, total_time



# ===================读取bfd文件========================
def read_bfddata(path):
    """
    读取 BFD 数据
    :param path: BFD 文件路径
    :return: BFD 数据数组
    """
    with open(path, 'rb') as f:
        bfd = np.fromfile(f, dtype=np.float32)

    bfd_data = np.zeros((36, 1000))
    for i in range(36):
        selected_start = (i) * 1000
        bfd_data[i, :] = bfd[selected_start:selected_start + 1000]

    return bfd_data


# ===================读取LS或PK文件========================
def read_file_as_int(path):
    """
    读取文件并将其转换为整数列表
    :param path: 文件路径
    :return: 整数数组
    """
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = np.fromfile(f, dtype=np.int32)
        return data
    return []



# ===================保存为CSV文件========================
def save_data_to_csv(data, filename, save_folder):
    """
    保存数据为 CSV 文件到指定路径
    :param data: 数据数组
    :param filename: 文件名
    :param save_folder: 保存路径
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # 如果路径不存在，则创建
    save_path = os.path.join(save_folder, filename)
    if isinstance(data, (np.ndarray, list)) and len(data) > 0:
        pd.DataFrame(data).to_csv(save_path, index=False, header=False)
        print(f"数据已保存到: {save_path}")
    else:
        print(f"数据为空，未保存: {save_path}")


# 主程序
folder_path = r'E:\Pythonpro\MCG_quexue_xinshuai\data_raw\jiankang\801-900'  # 修改为你的主文件夹路径
save_folder = r'E:\Pythonpro\MCG_quexue_xinshuai\data_csv'  # 自定义保存路径

# 获取主文件夹内的所有子文件夹
subfolders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])


for i, subfolder in enumerate(subfolders, 1787):
    subfolder_path = os.path.join(folder_path, subfolder)

    # 调用 tools_read_file 函数读取数据
    raw_data, ls_data, pk_data, bfd_data, total_time, root_path, subfolder_name = tools_read_file(
        subfolder_path, idx=1, interval=0)

    # 保存 BFD 数据
    bfd_filename = f'{i}.csv'
    save_data_to_csv(bfd_data, bfd_filename, save_folder)

    # 保存 PK 数据（如果 PK 数据不为空）
    pk_filename = f'{i}.1.csv'
    save_data_to_csv(pk_data, pk_filename, save_folder)

    print(f"处理完成: 文件夹 {subfolder}")
    print("=" * 50)
