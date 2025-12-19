import json
import numpy as np
import pandas as pd
import os

# 配置：你的模型名字和输出路径
MODEL_NAME = "MODEL_ST"  # 或者是你的 config 里的名字
JSON_PATH = os.path.join("output", MODEL_NAME, r"D:\New_python_project\MCG_diagnosis\output\ST_lstm2\cv_summary.json")


def print_5fold_results():
    if not os.path.exists(JSON_PATH):
        print(f"找不到文件: {JSON_PATH}")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 将 JSON 转换为 DataFrame，方便计算
    df = pd.DataFrame(data)

    print(f"\n{'=' * 60}")
    print(f"   5-Fold Cross Validation Summary: {MODEL_NAME}")
    print(f"{'=' * 60}")

    # 1. 打印每一折的详细数据 (可选)
    # print("Raw Data per Fold:")
    # print(df.round(4))
    # print("-" * 60)

    # 2. 自动识别有哪些指标列 (以 _final_ 结尾的列)
    metric_cols = [c for c in df.columns if "_final_" in c]

    # 按任务分组打印
    tasks = set([c.split('_final_')[0] for c in metric_cols])

    for task in sorted(list(tasks)):
        print(f"\nTask: 【 {task} 】")
        print(f"{'-' * 30}")
        print(f"{'Metric':<10} | {'Mean ± Std':<20}")
        print(f"{'-' * 30}")

        # 遍历该任务下的 5 个指标
        for m in ['acc', 'sens', 'spec', 'auc', 'f1']:
            col_name = f"{task}_final_{m}"
            if col_name in df.columns:
                values = df[col_name]
                mean_val = np.mean(values)
                std_val = np.std(values)
                # 格式化输出: 0.8523 ± 0.0123
                print(f"{m.upper():<10} | {mean_val:.4f} ± {std_val:.4f}")

    print(f"\n{'=' * 60}")
    # 打印最佳平均 F1 (用于早停那个指标)
    if "best_avg_val_f1" in df.columns:
        best_f1s = df["best_avg_val_f1"]
        print(f"Best Avg F1 (History): {np.mean(best_f1s):.4f} ± {np.std(best_f1s):.4f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    print_5fold_results()