# 实验任务说明

## 目标

在 sklearn 内置的乳腺癌二分类数据集（569 个样本、30 个数值特征）上构建分类器：

- 主优化指标：**验证集 AUC**（`val_auc`），决策只看 val；
- 最终结果：在固定的测试集上报告 `test_auc` / `test_f1`，避免指标“自我欺骗”；
- 运行预算：普通电脑单轮实验控制在秒级~分钟级。

## 可修改的实验配置（每轮由 Agent 决策）

- `model`：`logistic_regression` | `random_forest` | `mlp`
- `scaler`：是否先做 StandardScaler 特征标准化
- `hyperparams`：模型超参数（C、n_estimators、max_depth、hidden_layer_sizes 等）

## 迭代策略建议

1. 先建立可运行的 baseline（默认 Logistic Regression）；
2. 每轮只做一个有依据的修改（如特征缩放、换模型、调超参）；
3. 若新指标不低于当前最佳则采纳，否则保留最佳并记录回退；
4. 连续多轮无明显提升或达到预算时停止，并输出报告。
