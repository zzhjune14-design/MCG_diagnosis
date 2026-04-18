import pandas as pd

df = pd.read_csv('d:/New_python_project/MCG_diagnosis/cls_head/finetune_checkpoints/xinshuai_training_log.csv')
res = []
for f in range(1, 6):
    fold_df = df[df['Fold'] == f]
    best_idx = fold_df['Val_AUC'].idxmax()
    if pd.isna(best_idx):
        best_idx = fold_df.index[0] # fallback
    best = fold_df.loc[best_idx]
    res.append(f"Fold {f}: AUC={best['Val_AUC']:.3f}, Acc={best['Val_Acc']:.3f}, Sens={best['Val_Sens']:.3f}, Spec={best['Val_Spec']:.3f}, F1={best['Val_F1']:.3f}")

max_aucs = df.groupby('Fold')['Val_AUC'].max()
res.append(f"Avg AUC: {max_aucs.mean():.3f} +/- {max_aucs.std():.3f}")

with open('res.txt', 'w') as f:
    f.write('\n'.join(res))
print("Summary written to res.txt")
