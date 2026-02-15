import numpy as np
import matplotlib.pyplot as plt
import os
import random

# ==========================================
# 1. 配置绘图风格 (期刊标准)
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.titlesize': 12,
    'figure.dpi': 500
})


def check_and_plot_segments(folder_path):
    # 获取所有 .npy 文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    if not files:
        print(f"❌ 文件夹为空: {folder_path}")
        return

    # 随机选一个文件 (或者你可以指定 filename = 'xxx.npy')
    target_file = random.choice(files)
    file_path = os.path.join(folder_path, target_file)

    try:
        # 加载数据: [N, 36, 700]
        beats_data = np.load(file_path)
        num_beats, num_channels, seq_len = beats_data.shape

        print(f"📄 当前检查文件: {target_file}")
        print(f"📊 数据维度: {beats_data.shape} (包含 {num_beats} 个心拍)")
        print(f"📈 幅度范围: Min={np.min(beats_data):.2f}, Max={np.max(beats_data):.2f} pT")

        if np.max(np.abs(beats_data)) > 1000:
            print("⚠️ 警告: 幅度依然很大 (>1000 pT)，可能去噪未生效！")
        else:
            print("✅ 幅度正常，去噪似乎已生效。")

        # 随机选一个心拍进行绘制
        beat_idx = random.randint(0, num_beats - 1)
        beat_data = beats_data[beat_idx]  # Shape: [36, 700]

        # --- 绘图 ---
        fig = plt.figure(figsize=(4.72, 3.64))
        ax = fig.add_subplot(111)

        # 生成时间轴 (假设 1000Hz, 700点 = 700ms)
        # 将 x轴 设为 -250ms 到 +450ms (假设R峰在第250点)
        # 或者直接用 0-700ms
        time_axis = np.arange(seq_len)

        # 绘制 36 通道叠加
        for i in range(36):
            ax.plot(time_axis, beat_data[i, :],
                    linewidth=0.8,
                    alpha=0.8)  # 彩色叠加

        # 设置标签
        ax.set_xlabel("Time (ms)", labelpad=2,
                      fontdict={'fontname': 'Times New Roman', 'fontweight': 'bold', 'fontsize': 12})
        ax.set_ylabel("Amplitude (pT)", labelpad=2,
                      fontdict={'fontname': 'Times New Roman', 'fontweight': 'bold', 'fontsize': 12})

        ax.set_title(f"{target_file.split('_')[0]} (Beat {beat_idx})",
                     fontdict={'fontname': 'Times New Roman', 'fontsize': 12, 'fontweight': 'bold'},
                     pad=8)

        # 0点基线
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

        plt.tight_layout(pad=0.8)
        plt.show()

    except Exception as e:
        print(f"❌ 读取失败: {e}")


if __name__ == "__main__":
    # 修改为你的 .npy 文件夹路径
    DATA_DIR = r'E:\Pythonpro\MCG_quexue_xinshuai\heartbeat_segments'

    check_and_plot_segments(DATA_DIR)