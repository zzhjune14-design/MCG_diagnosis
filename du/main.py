from data_loader import DataLoader
from model_trainer import ModelTrainer
from result_interpreter import ResultInterpreter
import os

# 你的文件路径
file_path = r"D:\vxchat\xwechat_files\wxid_cik2jouejwed22_89fa\msg\file\2025-12\zuoshifeihou.xlsx"

if __name__ == "__main__":
    # 1. 加载数据
    print("Step 1: Loading Data...")
    loader = DataLoader(file_path)
    X, y = loader.load_process()

    # 2. 训练模型
    print("\nStep 2: Training Model...")
    trainer = ModelTrainer(X, y)
    best_model_pipeline = trainer.run_cv_experiment()

    # 3. 解释结果
    print("\nStep 3: Interpreting Results...")
    # 确保 interpreter 代码里也没有 plt.show()，如果有报错，请注释掉解释器的绘图部分
    interpreter = ResultInterpreter(best_model_pipeline, X.columns)
    try:
        interpreter.plot_feature_importance(top_n=10)
    except Exception as e:
        print(f"绘图跳过，直接输出文字结果。错误: {e}")