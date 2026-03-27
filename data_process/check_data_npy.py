import numpy as np


def inspect_npy_file(npy_path):
    print(f"🔍 正在检查文件: {npy_path}")
    print("-" * 40)

    try:
        # 1. 加载数据
        data = np.load(npy_path)

        # 2. 打印基本属性
        print(f"📦 数据维度 (Shape): {data.shape}  <-- 期待是 [1, 36, 700]")
        print(f"🏷️ 数据类型 (Dtype): {data.dtype}")

        # 3. 致命错误检查 (NaN 和 Inf)
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        print(f"🦠 包含 NaN (缺失值): {'是 ❌ (会导致Loss崩塌)' if has_nan else '否 ✅'}")
        print(f"💥 包含 Inf (无穷大): {'是 ❌ (会导致Loss崩塌)' if has_inf else '否 ✅'}")

        # 4. 打印统计信息 (评估数值分布是否正常)
        # 忽略 NaN 计算统计值，防止因为有 NaN 导致整体统计出来的也是 NaN
        print("\n📊 数值统计:")
        print(f"   最小值 (Min):  {np.nanmin(data):.6f}")
        print(f"   最大值 (Max):  {np.nanmax(data):.6f}")
        print(f"   平均值 (Mean): {np.nanmean(data):.6f}")

        # 5. 打印局部真实数据看看长什么样
        print("\n👀 局部数据截取 (第 1 个心拍, 第 1 个通道, 前 10 个采样点):")
        if data.ndim == 3:
            print(data[0, 0, :10])
        elif data.ndim == 2:
            print(data[0, :10])
        else:
            print(data[:10])

    except Exception as e:
        print(f"❌ 读取文件失败: {e}")


if __name__ == "__main__":
    # 请替换为你刚刚生成的任意一个 .npy 文件的真实路径
    FILE_PATH = r"E:\Pythonpro\MCG_quexue_xinshuai\all_data_folder\1.npy"

    inspect_npy_file(FILE_PATH)