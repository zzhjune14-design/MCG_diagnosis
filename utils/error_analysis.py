import pandas as pd
import torch
from tqdm import tqdm

from data_process.data_utils import _get_binary_labels_from_raws_or_map


@torch.no_grad()
def save_validation_results(model, loader, device, label_map,
                            field_isch, field_xin,
                            output_path):
    """
    运行一次验证集，将每个病人的详细预测结果保存为 CSV。
    包含：SubjectID, 真值, 预测概率, 预测类别, 是否错误
    """
    model.eval()
    results = []

    for Xb, subjects, raws in tqdm(loader, desc="Saving Val Results", leave=False):
        Xb = Xb.to(device)

        # 1. 获取模型预测
        out = model(Xb)
        if isinstance(out, dict):
            logit_isch = out.get(field_isch)
            logit_xin = out.get(field_xin)
        else:
            logit_isch, logit_xin = out

        # 维度对齐
        if logit_isch.dim() == 1: logit_isch = logit_isch.view(-1, 1)
        if logit_xin.dim() == 1: logit_xin = logit_xin.view(-1, 1)

        # 转概率
        prob_isch = torch.sigmoid(logit_isch).cpu().numpy().flatten()
        prob_xin = torch.sigmoid(logit_xin).cpu().numpy().flatten()

        # 2. 获取真实标签 (和 train_epoch 里的逻辑一样)
        y_isch_np = _get_binary_labels_from_raws_or_map(raws, subjects, field_isch, label_map)
        y_xin_np = _get_binary_labels_from_raws_or_map(raws, subjects, field_xin, label_map)

        # 3. 逐个样本组装数据
        for i, sub_id in enumerate(subjects):
            # --- 缺血任务 ---
            true_i = y_isch_np[i]
            pred_prob_i = prob_isch[i]
            pred_cls_i = 1 if pred_prob_i > 0.5 else 0

            # 判断是否错误 (排除 -1 的情况)
            is_wrong_i = False
            if true_i != -1:
                is_wrong_i = (true_i != pred_cls_i)

            # --- 心衰任务 ---
            true_x = y_xin_np[i]
            pred_prob_x = prob_xin[i]
            pred_cls_x = 1 if pred_prob_x > 0.5 else 0

            is_wrong_x = False
            if true_x != -1:
                is_wrong_x = (true_x != pred_cls_x)

            # --- 构造行数据 ---
            row = {
                "Subject_ID": sub_id,
                # 缺血信息
                f"{field_isch}_True": true_i,
                f"{field_isch}_Prob": f"{pred_prob_i:.4f}",  # 保留4位小数
                f"{field_isch}_Pred": pred_cls_i,
                f"{field_isch}_Error": "❌" if is_wrong_i else "✅",

                # 心衰信息
                f"{field_xin}_True": true_x,
                f"{field_xin}_Prob": f"{pred_prob_x:.4f}",
                f"{field_xin}_Pred": pred_cls_x,
                f"{field_xin}_Error": "❌" if is_wrong_x else ("Ignore" if true_x == -1 else "✅"),
            }
            results.append(row)

    # 4. 保存为 CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')  # sig 防止中文乱码
    print(f"  [Analysis] Error report saved to: {output_path}")