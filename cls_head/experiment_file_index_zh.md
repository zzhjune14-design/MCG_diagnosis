# `cls_head` 实验文件中文索引

这份文档用来说明目前 `cls_head` 目录下保留下来的重要文件分别记录了什么实验、对应什么架构、适合拿来做什么图。

## 一、当前建议优先保留和使用的结果

### 1. 缺血任务主线结果
- `finetune_checkpoints/`
  - 含义：原始 `Ischemia` 微调结果目录。
  - 对应架构：`HybridFineTuneModel`
  - 架构说明：`预训练 MCGEncoder + 波形手工特征 + 门控融合`
  - 用途：作为缺血任务主线结果和后续微调初始化 backbone 的来源。

### 2. 心衰任务基线结果
- `finetune_checkpoints_exp_A/`
  - 含义：`xinshuai` 的 `Exp A` 基线结果。
  - 对应架构：`HybridFineTuneModel`
  - 用途：作为“未做 high=0.1 噪声处理、未做老师修标”的 patient-level 基线结果。
  - 关键文件：
    - `xinshuai_training_log.csv`

### 3. 心衰任务高疑似错标样本降权主线
- `finetune_checkpoints_xinshuai_downweight_high_01/`
  - 含义：只对 `high` 优先级疑似错标样本降权到 `0.1` 的训练结果。
  - 对应架构：`HybridFineTuneModel`
  - 用途：当前“未人工改标签”条件下最重要的单模型结果。
  - 关键文件：
    - `xinshuai_training_log.csv`
    - `seed_1_fold_*_best_predictions.csv`

### 4. 老师确认前 50 个修正标签后的心衰结果
- `finetune_checkpoints_xinshuai_teacher_confirmed_top50/`
  - 含义：老师确认前 50 个最可疑样本后，改到标签副本再训练的结果。
  - 对应架构：`HybridFineTuneModel`
  - 用途：当前最强的“人工确认修标后”主线结果。
  - 当前结果水平：`5-fold mean Patient AUC ≈ 0.7442`
  - 关键文件：
    - `xinshuai_training_log.csv`
    - `seed_1_fold_*_best_predictions.csv`
    - `xinshuai_teacher_confirmed_top50_roc.png`
    - `xinshuai_teacher_confirmed_top50_roc_points.csv`

## 二、`high=0.1` 条件下的异构集成结果

### 1. 异构成员模型目录
- `hetero_high01_mlp_A/`
  - 含义：`high=0.1` 条件下的纯波形手工特征 MLP 成员。
  - 对应架构：`MLPWaveformModel`

- `hetero_high01_mlp_bce/`
  - 含义：`high=0.1` 条件下的纯波形手工特征 `MLP + BCE` 成员。
  - 对应架构：`MLPWaveformModel`
  - 说明：这是当前最有价值的异构辅助成员之一。

- `hetero_high01_spectral_mlp/`
  - 含义：`high=0.1` 条件下的频域特征 MLP 成员。
  - 对应架构：`MLPSpectralModel`
  - 说明：主要作为辅助异构成员保留。

### 2. 异构集成汇总文件
- `hetero_high01_member_summary.csv`
  - 含义：`high=0.1` 条件下各异构成员的单模型指标汇总。
  - 适合画：单模型 AUC / F1 / Acc 对比图。

- `hetero_high01_ensemble_summary.csv`
  - 含义：`high=0.1` 条件下各个简单平均集成组合的指标汇总。
  - 适合画：简单平均 ensemble 对比图。

- `hetero_high01_best_ensemble_oof.csv`
  - 含义：`high=0.1` 条件下最佳简单平均集成的 OOF 概率。
  - 适合画：ROC 曲线、PR 曲线。

- `hetero_high01_weighted_ensemble_summary.csv`
  - 含义：`high=0.1` 条件下粗粒度加权搜索结果汇总。

- `hetero_high01_weighted_ensemble_weights.csv`
  - 含义：粗粒度加权搜索得到的成员权重表。

- `hetero_high01_weighted_ensemble_oof.csv`
  - 含义：粗粒度加权集成的 OOF 概率。

- `hetero_high01_refined_weighted_summary.csv`
  - 含义：精细加权搜索后的最终结果汇总。
  - 当前最好组合：
    - `Hybrid-A-high01 = 0.70`
    - `MLP-BCE-high01 = 0.30`
  - 当前最好加权集成 AUC：
    - `AUC ≈ 0.6997`

- `hetero_high01_refined_best_oof.csv`
  - 含义：当前最优精细加权集成的 OOF 概率。
  - 适合画：最终 ensemble ROC 曲线。

## 三、标签复核与老师确认相关文件

### 1. 原始标签复核文件
- `label_review_copy.csv`
  - 含义：整张标签表的复核副本。
  - 内容：包含原始标签、模型概率、分歧分数、建议复核原因等。
  - 说明：不是只针对心衰，还包含全表信息。

- `label_review_candidates.csv`
  - 含义：整张表范围内的疑似错标候选。

- `label_review_candidates_xinshuai_only.csv`
  - 含义：只保留 `xinshuai` 有效样本后的疑似错标候选。
  - 用途：早期心衰标签清洗的入口文件。

### 2. 给老师用的心衰复核文件
- `xinshuai_teacher_review_copy.csv`
  - 含义：老师确认用的完整心衰复核表。
  - 内容：
    - 当前最强集成概率
    - 可疑程度优先级
    - 建议修正方向
    - `teacher_confirm_*` 人工确认字段
  - 当前状态：已经同步了前 50 个老师确认结果。

- `xinshuai_teacher_review_candidates.csv`
  - 含义：老师用的心衰疑似错标候选表。
  - 特点：按可疑程度排序，更适合继续人工核对。

### 3. 老师确认后生成的标签副本
- `label_xinshuai_teacher_confirmed_top50.csv`
  - 含义：把老师确认的前 50 个样本修正到完整标签副本后的文件。
  - 说明：原始 `label.csv` 没动。

- `label_xinshuai_teacher_confirmed_top50_changes.csv`
  - 含义：前 50 个样本改前改后的详细变更清单。
  - 用途：留档、答辩、回溯都很方便。

## 四、保留下来的历史轻量汇总文件

这些文件是之前探索阶段留下的轻量级 CSV，总结价值还在，但它们对应的大体积 checkpoint 目录已经清掉了。

- `hetero_member_summary.csv`
  - 含义：较早一轮异构成员汇总表。

- `hetero_ensemble_summary.csv`
  - 含义：较早一轮异构简单平均集成汇总表。

- `hetero_best_ensemble_oof.csv`
  - 含义：较早一轮最佳简单平均集成的 OOF。

- `hetero_member_summary_v2.csv`
  - 含义：扩展成员池之后的单模型汇总表。

- `hetero_ensemble_summary_v2.csv`
  - 含义：扩展成员池之后的集成汇总表。

- `hetero_best_ensemble_oof_v2.csv`
  - 含义：扩展成员池之后最佳集成的 OOF。

- `hetero_weighted_ensemble_summary.csv`
  - 含义：较早一轮加权集成汇总表。

- `hetero_weighted_ensemble_weights.csv`
  - 含义：较早一轮加权集成权重表。

- `hetero_weighted_ensemble_oof.csv`
  - 含义：较早一轮加权集成 OOF。

## 五、核心代码和辅助脚本

- `finetune_model.py`
  - 含义：模型定义文件。
  - 重点结构：
    - `HybridFineTuneModel`
    - `MLPWaveformModel`
    - `MLPSpectralModel`

- `train_finetune.py`
  - 含义：统一训练入口。
  - 功能：
    - patient-level 指标记录
    - label noise downweight
    - 多模型变体训练
    - A/B/C 实验配置

- `create_label_review_copy.py`
  - 含义：生成原始标签复核副本和候选表的脚本。

- `create_conservative_label_copy.py`
  - 含义：较早期保守 auto-flip 标签副本脚本。
  - 说明：现在主要保留作过程记录，不建议再作为主线。

- `create_xinshuai_teacher_review_copy.py`
  - 含义：生成心衰老师复核表和候选表的脚本。

- `create_xinshuai_teacher_confirmed_copy.py`
  - 含义：把老师确认的 top-N 样本写进新的标签副本的脚本。

- `sync_teacher_confirmed_top50.py`
  - 含义：把老师已确认的前 50 条结果同步回 `xinshuai_teacher_review_copy.csv` 的脚本。

## 六、后续画图最建议直接读取的文件

如果你接下来要专门整理绘图数据，最建议优先用这些：

- `finetune_checkpoints_exp_A/xinshuai_training_log.csv`
  - 用于画：心衰原始基线训练曲线 / 折间对比

- `finetune_checkpoints_xinshuai_downweight_high_01/xinshuai_training_log.csv`
  - 用于画：高优先级疑似错标样本降权后的训练结果

- `finetune_checkpoints_xinshuai_teacher_confirmed_top50/xinshuai_training_log.csv`
  - 用于画：老师确认修标后的训练结果

- `hetero_high01_member_summary.csv`
  - 用于画：不同成员单模型柱状图

- `hetero_high01_ensemble_summary.csv`
  - 用于画：简单平均 ensemble 柱状图

- `hetero_high01_refined_weighted_summary.csv`
  - 用于画：最终加权 ensemble 对比图

- `finetune_checkpoints_xinshuai_teacher_confirmed_top50/xinshuai_teacher_confirmed_top50_roc_points.csv`
  - 用于画：当前修标后主线 ROC 曲线
