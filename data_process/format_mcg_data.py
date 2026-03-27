import os
import pickle
import numpy as np
from tqdm import tqdm


def format_mcg_data(pkl_folder, save_folder, window_size=700, pre_r_points=250):
    """
    将 6x6x1000 的标准心拍提取为网络所需的 [1, 36, 700] numpy 矩阵。
    window_size: 截取总长度 700
    pre_r_points: R波前保留的点数 (默认 250，这样 R 波后保留 450)
    """
    os.makedirs(save_folder, exist_ok=True)

    pkl_files = [f for f in os.listdir(pkl_folder) if f.endswith('.pickle')]
    print(f"🚀 开始转换 {len(pkl_files)} 个心拍文件...")

    success_count = 0

    for fname in tqdm(pkl_files):
        pkl_path = os.path.join(pkl_folder, fname)
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            # 1. 提取信号并展平: (6, 6, 1000) -> (36, 1000)
            amcg = data['amcg']
            flattened_amcg = amcg.reshape(36, -1)

            # 2. 获取 R 波位置
            r_peak = int(data['R'])  # 例如 361

            # 3. 计算截取窗口
            start_idx = r_peak - pre_r_points
            end_idx = start_idx + window_size

            # 初始化一个全 0 矩阵 (防越界兜底)
            final_beat = np.zeros((36, window_size), dtype=np.float32)

            # 计算实际可以截取的有效区间
            valid_start = max(0, start_idx)
            valid_end = min(flattened_amcg.shape[1], end_idx)

            # 计算要填入 final_beat 的对应位置
            insert_start = valid_start - start_idx
            insert_end = insert_start + (valid_end - valid_start)

            # 执行截取与填充
            final_beat[:, insert_start:insert_end] = flattened_amcg[:, valid_start:valid_end]

            # 4. 增加一个 batch 维度，变成 [1, 36, 700]
            # (这样完全兼容我们之前写好的 Dataset 逻辑)
            final_beat = np.expand_dims(final_beat, axis=0)

            # 5. 提取文件名中的 subject id (例如 subject_1.pkl -> subject_1.npy)
            # 你可以根据实际文件名格式微调这里
            save_name = fname.replace('.pickle', '.npy')
            save_path = os.path.join(save_folder, save_name)

            np.save(save_path, final_beat)
            success_count += 1

        except Exception as e:
            print(f"❌ 解析 {fname} 失败: {e}")

    print(f"\n✅ 转换完成！成功处理: {success_count}/{len(pkl_files)}")


if __name__ == "__main__":
    # 请修改为你的实际路径
    # 例如：先把缺血和心衰的所有 pkl 放进去，转换出 .npy
    INPUT_PKL_DIR = r"E:\Pythonpro\MCG_quexue_xinshuai\data_pickle"
    OUTPUT_NPY_DIR = r"E:\Pythonpro\MCG_quexue_xinshuai\all_data_folder"

    format_mcg_data(INPUT_PKL_DIR, OUTPUT_NPY_DIR)