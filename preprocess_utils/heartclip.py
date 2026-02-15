import pickle
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import os
from tqdm import tqdm


# 修改后的心拍分割函数（支持多导联）
def segment_multilead_beats(multi_lead_ecg, ref_rpeaks, sampling_rate=1000,
                            before=0.25, after=0.45):
    """
    多导联心拍分割（基于参考导联R峰）
    参数：
        multi_lead_ecg: 多导联数据 (36, N)
        ref_rpeaks: R峰索引列表
        before: R峰前截取时间(s)
        after: R峰后截取时间(s)
    """
    num_leads, total_samples = multi_lead_ecg.shape

    samples_before = int(before * sampling_rate)
    samples_after = int(after * sampling_rate)
    target_length = samples_before + samples_after

    all_beats = []
    for r in ref_rpeaks:
        start = r - samples_before
        end = r + samples_after

        # 边界检查 (虽然去掉了头尾，但保留这个检查以防万一)
        if start < 0 or end > total_samples:
            continue

        # 提取多导联心拍
        multilead_beat = multi_lead_ecg[:, start:end]

        # 长度校验与填充 (处理可能存在的微小长度差异)
        if multilead_beat.shape[1] != target_length:
            # 如果截取长度不足（通常发生在边界），进行填充
            if multilead_beat.shape[1] < target_length:
                pad_width = ((0, 0), (0, target_length - multilead_beat.shape[1]))
                multilead_beat = np.pad(multilead_beat, pad_width)
            else:
                # 如果超长（理论上不会发生，除非索引错乱），截断
                multilead_beat = multilead_beat[:, :target_length]

        all_beats.append(multilead_beat)

    return np.array(all_beats)


def process_all_pickles(input_folder, output_folder):
    """
    批量读取 Pickle 并切割心拍
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有 pickle 文件
    pickle_files = [f for f in os.listdir(input_folder) if f.endswith('.pkl')]
    length_dic = {}

    print(f"开始处理 {len(pickle_files)} 个样本...")

    for file_name in tqdm(pickle_files):
        try:
            # 1. 加载数据
            file_path = os.path.join(input_folder, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            patient_id = data['id']
            # 注意：确保这里取的是你去噪后的数据，如果之前保存的是 'raw_data' 键且已去噪
            multi_lead_mcg = data['raw_data']
            sampling_rate = data.get('sampling_rate', 1000)

            # 2. R峰检测
            try:
                # 使用 neurokit2 寻找 R 峰
                # 为了提高R峰检测准确率，建议先对参考导联进行简单的带通清洗
                # 即使数据已经去噪，nk.ecg_clean 也能进一步规范化信号增强R峰特征
                clean_ref = nk.ecg_clean(multi_lead_mcg[26], sampling_rate=sampling_rate)
                _, rpeak_sets = nk.ecg_peaks(clean_ref, sampling_rate=sampling_rate)

                # 获取原始 R 峰列表
                rpeaks = rpeak_sets["ECG_R_Peaks"]

                # ==================================================
                # 核心修改：去掉第一个和最后一个 R 峰
                # ==================================================
                if len(rpeaks) > 2:
                    # 只有当心拍数大于2时才切除，避免切完没数据了
                    # 目的：去除开头不稳定的波形和结尾可能截断的波形
                    rpeaks = rpeaks[1:-1]
                    # print(f"  {patient_id}: 已剔除首尾心拍，剩余 {len(rpeaks)} 个")
                elif len(rpeaks) == 0:
                    print(f"警告: {patient_id} 未检测到 R 峰")
                    continue
                # ==================================================

                # 3. 执行分割
                good_beats = segment_multilead_beats(
                    multi_lead_mcg,
                    rpeaks,
                    sampling_rate=sampling_rate,
                    before=0.25,  # 根据你的需求保持参数
                    after=0.45
                )

                if len(good_beats) == 0:
                    print(f"警告: {patient_id} 分割后无有效数据")
                    continue

                # 4. 保存切割后的心拍 (Shape: [N, 36, 700])
                # 文件名建议保持简单，方便后续读取
                save_path = os.path.join(output_folder, f"{patient_id}_beats.npy")
                np.save(save_path, good_beats)

                # 记录该病人的心拍数量
                length_dic[patient_id] = len(good_beats)

            except Exception as e:
                print(f"处理病人 {patient_id} 时发生错误 (R峰检测/分割失败): {e}")

        except Exception as e:
            print(f"读取文件 {file_name} 失败: {e}")

    # 5. 保存心拍统计信息
    with open(os.path.join(output_folder, 'beats_length_info.pkl'), 'wb') as f:
        pickle.dump(length_dic, f)

    print(f"✅ 处理完成！数据保存在: {output_folder}")


if __name__ == "__main__":
    # 请确保路径正确
    INPUT_DIR = r'E:\Pythonpro\MCG_quexue_xinshuai\raw_data_pickle'
    OUTPUT_DIR = r'E:\Pythonpro\MCG_quexue_xinshuai\heartbeat_segments'

    process_all_pickles(INPUT_DIR, OUTPUT_DIR)