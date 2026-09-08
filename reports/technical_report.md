# AutoResearch Agent for ML — Technical Report

> 日期：2026-09-08 ｜ 语言：中文 ｜ 页数目标：≤6 页
> 配套仓库：Autoresearch-agent-ml；本报告对应的运行记录见第 7 节复现索引。

## 1. 摘要与验收对应

本项目实现了一个“自动迭代机器学习实验的 Agent”：给定自然语言研究目标后，
系统自主读取任务配置与可用数据，提出下一步实验方案，真实调用 Python 执行
训练与评估，依据验证集指标独立判断采纳、回退或停止，并在收尾阶段自动把
“当前最佳实验”重训生成 Kaggle 提交文件。系统内置 rule（确定性）与
LLM（DeepSeek deepseek-v4-flash，effort=max）两种决策大脑。

当前进展：5 个 Kaggle 任务的数据、Baseline、规则/LLM 迭代与提交管线全部打通；
LLM 大脑在 5 个任务上各完成 3 轮真实迭代；5 份“全局最佳”提交文件已生成并
通过官方样例格式校验。

与任务书要求（init.md）的对应关系：

| init.md 要求 | 本报告章节 |
| --- | --- |
| 系统架构 | 第 2 节 |
| 核心流程 | 第 2 节 |
| 关键设计 | 第 3 节 |
| 实验/测试结果 | 第 4 节 |
| 失败案例 | 第 5 节 |
| 改进方向 | 第 6 节 |

## 2. 系统架构与核心流程

系统采用“单一主入口 + 可插拔决策大脑 + 任务级实验脚本”的分层结构：

```text
run_agent.py（唯一主入口）
├── agent/config.py      配置加载、路径解析、LLM 环境变量覆盖
├── agent/planner.py     决策大脑：rule / llm（OpenAI 兼容 / Anthropic）
├── agent/executor.py    参数快照落盘 + subprocess 真实运行实验
├── agent/evaluator.py   val 指标比较：accept / revert
├── agent/state.py       全量状态 JSONL + 可读 RUN_LOG
└── configs/<task_id>.yaml  每任务的运行配置
```

任务层将“评估实验”与“提交管线”分开：

- `experiments/tasks/<task_id>/train.py`：加载数据 → 切分 train/val/test →
  训练 → 输出 `val_<metric>` / `test_<metric>`；
- `experiments/tasks/<task_id>/submit.py`：用全部有标签数据重训并预测官方
  test，输出标准 Kaggle CSV；
- `tasks/tasks.yaml`：任务注册表（类型、官方指标、来源）。

一次 Agent 运行的闭环为：

```text
plan（rule/LLM 提出 params）
  → execute（写入快照，真实运行 train.py）
  → evaluate（对比 val 指标，accept / revert）
  → decision（继续 / 回退 / 停止）
  →（停止后）自动提交（best 快照 → submit.py → 格式校验）
```

每一步均追加到 `runs/<task_id>/<timestamp>/state.jsonl`，并同步产出
参数快照、stdout/stderr、metrics JSON、summary.json 与 RUN_LOG.md，满足
“完整过程可追溯”的要求。终止机制覆盖三类退出：达到最大步数
（`stop_max_steps`）、连续轮次提升不足（`stop_converged`）、连续致命错误
（`stop_failure`），退出原因均写入状态。

## 3. 关键设计

### 3.1 任务可配置性

- 每任务一个 `configs/<task_id>.yaml`：任务描述、train/submit 脚本、数据路径、
  固定 seed、train/val/test 比例、目标指标及方向（`higher_is_better`）。
- 模型名、API、数据路径、预算全部来自配置或命令行，不写死。

### 3.2 指标口径与防过拟合

- 决策只读取 `val_*`，最终结果只报告 `test_*`；回归类指标越低越好、分类
  指标越高越好，由方向配置驱动评估器。
- bike_sharing_demand 按时间顺序切分，避免邻近时刻泄漏；facial_keypoints
  按行切分，各坐标列用其可见样本分别训练，评估时按可见 cell 聚合 RMSE。

### 3.3 大脑与容错

- `rule` 大脑：确定性候选序列，无需 API key，适合复现与验证；
- `llm` 大脑：真实调用 DeepSeek；模型/端点/effort 通过 `.env` 注入；
- 空响应或 JSON 截断自动重试一次，仍失败则回退 rule；
- LLM 提出的旧版超参（如 `max_features: auto`）经统一清洗后再构造模型；
- 连续致命错误达到阈值即停止并记录原因，防止无限循环。

### 3.4 收尾自动提交

Agent 停止后自动执行“最佳参数快照 → 全量重训 → 生成提交 CSV → 与官方
sample 比对格式”，结果写入 state/summary，避免“文档最佳与提交文件不一致”。

## 4. 实验设计与结果

### 4.1 实验协议

- 统一随机种子 seed=42；无外部指定的比赛按 60/20/20 划分本地
  train/val/test；
- 运行环境为普通个人电脑 + scikit-learn / pandas / xgboost；
- 决策只看 val，最终表中同时给出 test；提交文件用全量训练数据重训。

### 4.2 Baseline 与全局最佳结果

“全局最佳”取该任务所有历史 run 中 val 最优的一轮（含 rule 与 LLM 大脑）：

| 任务 | Baseline（val / test） | 最佳配置（来源 run） | 最佳（val / test） |
| --- | --- | --- | --- |
| titanic | acc 0.8202 / 0.8101 | Logistic C=0.1（20260906_205335） | acc 0.8315 / 0.8045 |
| house_prices | rmsle 0.1358 / 0.1287 | Ridge α=0.1（20260906_163159） | rmsle 0.1344 / 0.1261 |
| bike_sharing_demand | rmsle 0.5166 / 0.5322 | RandomForest(800)（20260906_163313） | rmsle 0.5134 / 0.5166 |
| digit_recognizer | acc 0.9077 / 0.9015 | MLP(256)（20260907_194210） | acc 0.9638 / 0.9613 |
| facial_keypoints | rmse 3.0897 / 3.0839 | PCA(96)+Ridge（20260906_163343） | rmse 3.0429 / 3.0177 |

### 4.3 提交文件（全部通过官方样例格式校验）

| 任务 | 最佳参数来源 | 训练行数 | 提交行数 |
| --- | --- | --- | --- |
| titanic | runs/titanic/20260906_205335/snapshots/step003_attempt001_params.yaml | 891 | 418 |
| house_prices | runs/house_prices/20260906_163159/snapshots/step003_attempt001_params.yaml | 1460 | 1459 |
| bike_sharing_demand | runs/bike_sharing_demand/20260906_163313/snapshots/step002_attempt001_params.yaml | 10886 | 6493 |
| digit_recognizer | runs/digit_recognizer/20260907_194210/snapshots/step003_attempt001_params.yaml | 42000 | 28000 |
| facial_keypoints | runs/facial_keypoints/20260906_163343/snapshots/step002_attempt001_params.yaml | 7049 | 27124（按 IdLookupTable 映射） |

对应文件为 `submissions/best_<task_id>_submission.csv`，列名、行数与 id 顺序
均与官方 sample 一致。

### 4.4 titanic 单任务过程示例（LLM 大脑 3 轮）

来自 `runs/titanic/20260907_193841/RUN_LOG.md` 的真实决策链：

| 步 | 大脑提出的修改 | val_acc | 决策 |
| --- | --- | --- | --- |
| 1 | Logistic Regression + 标准化（baseline） | 0.8202 | accept |
| 2 | 切换 RandomForest（100 棵树） | 0.7697 | revert |
| 3 | MLP（16 隐单元） | 0.8258 | accept |

该示例展示 Agent 在无人工干预下完成“建模 → 对照 → 回退差方案 → 采纳更优方案”
的完整反思闭环；此后 run 收尾阶段自动生成了该 run 的最佳提交。

## 5. 失败案例与恢复

1. **LLM 空响应（自动重试）**：deepseek-v4-flash 偶发返回空 content 且
   `finish_reason='length'`（推理占用全部输出预算）。系统将该调用判定为
   可恢复错误并自动重试一次；在 `runs/titanic/20260906_205335` 等运行中
   重试后成功提出下一步方案，未中断迭代。
2. **非法超参（统一清洗）**：LLM 曾提出 `max_features: auto`，该取值在
   scikit-learn 1.6 已移除。早期运行 `runs/house_prices/20260907_193918`
   因此以 `stop_failure` 退出；随后在实验公共层加入超参清洗（auto → sqrt）
   并重跑成功，成为框架鲁棒性的真实改进案例。
3. **指标下降回退（revert）**：facial_keypoints 第 2 轮 MLP 的
   val_rmse=6.11，明显差于 Ridge baseline（3.09），评估器正确判定 revert，
   最佳结果保持 PCA(96)+Ridge。

以上三类失败分别覆盖“外部 API 异常”“参数合法性错误”“模型效果不佳”，
且均有状态日志与修复后的成功运行作为证据。

## 6. 局限与改进方向

- 5 个 Kaggle 竞赛均为历史比赛，真实线上提交未执行；提交文件仅完成本地格式
  校验；
- `rule` 大脑是确定性候选序列，不具备真正的“反思式实验设计”；
- 单次运行不支持断点续跑（暂无 `--resume`）；
- 依赖声明采用 `>=` 下限，全新环境会随时间解析到更新版本，需定期回归；
- 后续可扩展方向：自动生成并迭代“特征工程/数据处理代码”而不仅是模型参数、
  多 run 取均值汇报、将最佳提交回填到仓库的 submissions 目录、断点恢复。

## 7. 复现索引

### 7.1 零依赖冒烟（无需 Kaggle / API key）

```bash
python run_agent.py --config config.yaml --brain rule --max-steps 3
```

### 7.2 单任务运行与提交

```bash
python run_agent.py --config configs/titanic.yaml --brain rule --max-steps 3
python run_agent.py --config configs/titanic.yaml --brain llm --max-steps 3
python experiments/validate_submission.py \
    --submission submissions/best_titanic_submission.csv \
    --sample data/raw/titanic/gender_submission.csv \
    --id-col PassengerId --required PassengerId,Survived
```

### 7.3 本报告引用到的运行记录

| 内容 | 路径 |
| --- | --- |
| titanic 全局最佳 run | runs/titanic/20260906_205335 |
| house_prices 全局最佳 run | runs/house_prices/20260906_163159 |
| bike 全局最佳 run | runs/bike_sharing_demand/20260906_163313 |
| digit 全局最佳 run | runs/digit_recognizer/20260907_194210 |
| facial 全局最佳 run | runs/facial_keypoints/20260906_163343 |
| titanic LLM 3 轮过程示例 | runs/titanic/20260907_193841 |
| house 非法超参失败 run | runs/house_prices/20260907_193918 |
| 五份全局最佳提交 | submissions/best_*.csv |
