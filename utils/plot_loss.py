import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_chinese_loss_curve(csv_path):
    # 1. 读取数据
    df = pd.read_csv(csv_path)

    # 动态寻找 epoch 和 loss 列
    epoch_col = [c for c in df.columns if 'epoch' in c.lower()][0]
    loss_col = [c for c in df.columns if 'loss' in c.lower()][0]

    # 2. 开始绘图
    plt.figure(figsize=(10, 6))

    # ==========================================
    # 🌟 核心修复：必须先设置 Seaborn 风格，再注入中文字体
    # ==========================================
    sns.set_style("whitegrid")  # 第一步：应用美观的白底网格风格（这步会重置底层配置）

    # 第二步：强行覆盖刚刚被重置的字体配置（Windows 绝配是 SimHei）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # 画线
    sns.lineplot(data=df, x=epoch_col, y=loss_col, color='#d62728', linewidth=2.5)

    # 3. 设置中文标题和标签
    plt.title('模型训练损失下降曲线', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('训练轮次 (Epoch)', fontsize=14, labelpad=10)
    plt.ylabel('平均损失 (Average Loss)', fontsize=14, labelpad=10)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 4. 增加网格线和美化
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 5. 高清保存 (600 dpi 适合顶级期刊)
    save_path = r'D:\New_python_project\MCG_diagnosis\output\loss_curve_cn_high_res.png'
    plt.savefig(save_path, dpi=600)
    print(f"✅ 中文Loss曲线已生成并保存至: {save_path}")

    # plt.show()


if __name__ == "__main__":
    plot_chinese_loss_curve(r'D:\New_python_project\MCG_diagnosis\checkpoints\fuxian_training_loss.csv')