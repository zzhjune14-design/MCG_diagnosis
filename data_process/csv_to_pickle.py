import numpy as np
import pandas as pd
import pickle
import os

# 设置文件夹路径
csv_folder_path = r"E:\Pythonpro\MCG_quexue_xinshuai\data_csv"  # CSV 文件路径
pickle_folder_path = r"E:\Pythonpro\MCG_quexue_xinshuai\data_pickle"  # 指定保存 pickle 文件的路径

# 确保指定的 pickle 文件夹存在，如果不存在则创建
if not os.path.exists(pickle_folder_path):
    os.makedirs(pickle_folder_path)

# 循环读 csv 文件
for i in range(801, 901):
    # 生成 amcg 数据和特征点数据的文件路径
    amcg_file_path = os.path.join(csv_folder_path, f"{i}.csv")
    features_file_path = os.path.join(csv_folder_path, f"{i}.1.csv")  # 特征点存储文件

    # 读取 amcg 数据
    amcg_data = pd.read_csv(amcg_file_path, header=None).values

    # 确保 amcg 数据形状是 36×1000
    print(f"处理文件: {amcg_file_path}")
    print(f"amcg 数据形状: {amcg_data.shape}")
    if amcg_data.shape != (36, 1000):
        raise ValueError(f"amcg 数据形状不匹配，应该是 36×1000（文件: {amcg_file_path}）")

    # 重塑 amcg 数据为 6×6×1000
    amcg = amcg_data.reshape((6, 6, 1000))
    print(f"重塑后的 amcg 数据形状: {amcg.shape}")

    # 读取特征点数据
    if os.path.exists(features_file_path):
        features_data = pd.read_csv(features_file_path, header=None).values.flatten()  # 展平为一维数组
        print(f"读取的特征点数据: {features_data}")

        # 检查特征点数据长度
        if len(features_data) < 7:
            raise ValueError(f"特征点数据长度不足 7（文件: {features_file_path}）")

        # 计算所需的特征点
        R = features_data[2]  # 第三个数据（初始为第 0 个）
        Q = (features_data[2] + features_data[1]) / 2  # （第三个 - 第二个）/ 2
        S = (features_data[3] + features_data[2]) / 2  # （第四个 - 第三个）/ 2
        T = features_data[5]  # 第六个数据
        dQ = dR = dS = (features_data[3] - features_data[1]) / 3  # （第四个 - 第二个）/ 3
        dT = features_data[6] - features_data[4]  # 第七个 - 第五个

        # 特征点字典
        features = {
            'Q': Q,
            'R': R,
            'S': S,
            'T': T,
            'dQ': dQ,
            'dR': dR,
            'dS': dS,
            'dT': dT,
        }
        print(f"计算的特征点: {features}")
    else:
        print(f"未找到特征点文件: {features_file_path}")
        features = {}

    # 构造保存数据的字典
    data_to_save = {
        'amcg': amcg,
        **features  # 解包特征点数据字典，合并到保存数据中
    }

    # 生成 pickle 文件路径（保存到指定文件夹）
    pickle_file_path = os.path.join(pickle_folder_path, f"{i}.pickle")

    # 保存为 pickle 文件
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"数据已保存为 pickle 文件: {pickle_file_path}")
    print("=" * 50)
