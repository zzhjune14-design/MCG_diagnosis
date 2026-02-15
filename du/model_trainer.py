import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.best_model = None

    def build_pipeline(self):
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            # class_weight='balanced' 对不平衡数据非常关键
            ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
        ])

    def run_cv_experiment(self, n_splits=5):
        print(f"正在进行 {n_splits} 折交叉验证 (计算 准确率/特异性/敏感性)...")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        param_grid = {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [3, 5, 7],
            'clf__min_samples_leaf': [2, 4]
        }

        # 用于存储每一折的指标
        metrics = {
            'auc': [],
            'accuracy': [],
            'specificity': [],
            'sensitivity': []
        }

        tprs = []
        mean_fpr = np.linspace(0, 1, 100)

        plt.figure(figsize=(10, 8))

        fold_idx = 0
        for train_idx, val_idx in cv.split(self.X, self.y):
            fold_idx += 1
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            # --- 1. 训练 ---
            pipeline = self.build_pipeline()
            grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=1)
            grid.fit(X_train, y_train)

            if fold_idx == 1:
                self.best_model = grid.best_estimator_

            # --- 2. 预测 ---
            # 获取概率用于计算 AUC
            probas = grid.best_estimator_.predict_proba(X_val)[:, 1]
            # 获取直接的类别预测 (0或1) 用于计算准确率和混淆矩阵
            y_pred = grid.best_estimator_.predict(X_val)

            # --- 3. 计算核心指标 ---
            # A. AUC
            fpr, tpr, _ = roc_curve(y_val, probas)
            roc_auc = auc(fpr, tpr)
            metrics['auc'].append(roc_auc)
            tprs.append(np.interp(mean_fpr, fpr, tpr))

            # B. 准确率 (Accuracy)
            acc = accuracy_score(y_val, y_pred)
            metrics['accuracy'].append(acc)

            # C. 特异性 (Specificity) & 敏感性 (Sensitivity)
            # 混淆矩阵: tn (真阴性), fp (假阳性), fn (假阴性), tp (真阳性)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

            metrics['specificity'].append(specificity)
            metrics['sensitivity'].append(sensitivity)

            print(
                f"Fold {fold_idx}: AUC={roc_auc:.3f} | 准确率={acc:.3f} | 特异性={specificity:.3f} | 敏感性={sensitivity:.3f}")
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {fold_idx} AUC={roc_auc:.2f}')

        # --- 4. 汇总结果 ---
        print("\n" + "=" * 30)
        print("最终平均结果 (Mean Results):")
        print("=" * 30)
        print(f"平均 AUC      : {np.mean(metrics['auc']):.3f} (±{np.std(metrics['auc']):.3f})")
        print(f"平均 准确率   : {np.mean(metrics['accuracy']):.3f} (±{np.std(metrics['accuracy']):.3f})")
        print(
            f"平均 特异性   : {np.mean(metrics['specificity']):.3f} (±{np.std(metrics['specificity']):.3f}) -> 识别'非肥厚'的能力")
        print(
            f"平均 敏感性   : {np.mean(metrics['sensitivity']):.3f} (±{np.std(metrics['sensitivity']):.3f}) -> 识别'肥厚'的能力")
        print("=" * 30)

        # 绘图收尾
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(metrics['auc'])

        plt.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f'Mean AUC = {mean_auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r')
        plt.title('ROC Curve (with Accuracy/Specificity Analysis)')
        plt.legend(loc="lower right")

        plt.savefig('ROC_Result_With_Metrics.png')
        plt.close()
        print(f"结果图已保存为: ROC_Result_With_Metrics.png")

        return self.best_model