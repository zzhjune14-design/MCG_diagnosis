import matplotlib.pyplot as plt
import numpy as np

class ResultInterpreter:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def plot_feature_importance(self, top_n=10):
        # 从 Pipeline 中提取随机森林模型
        rf_model = self.model.named_steps['clf']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print(f"\n=== Top {top_n} 关键特征 ===")
        for i in range(top_n):
            idx = indices[i]
            print(f"{i + 1}. {self.feature_names[idx]}: {importances[idx]:.4f}")

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(top_n), importances[indices[:top_n]], align="center")
        plt.xticks(range(top_n), [self.feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()