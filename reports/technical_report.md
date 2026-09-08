# AutoResearch Agent for ML — Technical Report

> 日期：2026-09-08 ｜ 语言：中文 ｜ 页数目标：≤6 页
> 配套仓库：Autoresearch-agent-ml；本报告对应的复现说明见第 6 节复现索引。

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
| 改进方向 | 第 5 节 |

## 2. 系统架构与核心流程

系统采用“单一主入口 + 可插拔决策大脑 + 任务级实验脚本”的分层结构：

```text
run_agent.py（单任务入口）+ run_all.py（一键批量入口）
├── agent/config.py      配置加载、路径解析、LLM 环境变量覆盖
├── agent/planner.py     决策大脑：rule / llm（OpenAI 兼容 / Anthropic）
├── agent/executor.py    参数快照落盘 + subprocess 真实运行实验
├── agent/evaluator.py   val 指标比较：accept / revert
├── agent/state.py       全量状态 JSONL + 可读 RUN_LOG
└── configs/<task_id>.yaml  每任务的运行配置
```

`run_all.py` 会按任务注册表顺序运行一个或全部任务：数据缺失时自动下载、
逐个调用 `run_agent.py`、收尾生成提交，并把汇总报告写入
`runs/batch/<timestamp>/`；单个任务失败不中断整批。

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
参数快照、stdout/stderr、metrics JSON、summary.json、checkpoint.json 与
RUN_LOG.md，满足
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
- LLM 输出的 params 会先做程序级校验：模型名必须在白名单内、顶层字段只保留
  配置允许的 key、scaler/hyperparams/任务级字段做类型与边界清洗
  （如 `max_features: auto → sqrt`、`n_estimators`/`max_iter` 上限）；
- `stop_failure`、0 轮成功或提交失败会映射为非零退出码，供 run_all/CI 判定；
- 每完成一步会落盘 `checkpoint.json`；`--resume [run_dir|latest]` 从
  next_step 恢复大脑游标、最佳参数与收敛状态，无 checkpoint 的旧 run 会从
  state.jsonl 兼容重建；
- 连续致命错误达到阈值即停止并记录原因，防止无限循环。

### 3.4 收尾自动提交

Agent 停止后自动执行“最佳参数快照 → 全量重训 → 生成提交 CSV → 与官方
sample 比对格式（含行序、数值/NaN/Inf 校验）”，结果写入 state/summary，
避免“文档最佳与提交文件不一致”。如需实际上传 Kaggle，可显式传入
`--kaggle-upload`（默认不访问网络）。

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
| titanic | submissions/best_titanic_params.yaml | 891 | 418 |
| house_prices | submissions/best_house_prices_params.yaml | 1460 | 1459 |
| bike_sharing_demand | submissions/best_bike_sharing_demand_params.yaml | 10886 | 6493 |
| digit_recognizer | submissions/best_digit_recognizer_params.yaml | 42000 | 28000 |
| facial_keypoints | submissions/best_facial_keypoints_params.yaml | 7049 | 27124（按 IdLookupTable 映射） |

对应文件为 `submissions/best_<task_id>_submission.csv`，列名、行数与 id 顺序
均与官方 sample 一致；上述参数 YAML 已随仓库提交，clone 后下载数据即可重生成。

### 4.4 titanic 单任务过程示例（LLM 大脑 3 轮）

来自 `reports/example_runs/titanic_llm_3_rounds_20260907_193841/RUN_LOG.md`
（该完整运行已随仓库提交）的真实决策链：

| 步 | 大脑提出的修改 | val_acc | 决策 |
| --- | --- | --- | --- |
| 1 | Logistic Regression + 标准化（baseline） | 0.8202 | accept |
| 2 | 切换 RandomForest（100 棵树） | 0.7697 | revert |
| 3 | MLP（16 隐单元） | 0.8258 | accept |

该示例展示 Agent 在无人工干预下完成“建模 → 对照 → 回退差方案 → 采纳更优方案”
的完整反思闭环；此后 run 收尾阶段自动生成了该 run 的最佳提交。

## 5. 局限与改进方向

- 5 个 Kaggle 竞赛均为历史比赛，真实线上提交未执行；提交文件仅完成本地格式
  校验；如需实际上传可显式 `--kaggle-upload`，是否仍开放提交取决于比赛状态；
- `rule` 大脑是确定性候选序列，不具备真正的“反思式实验设计”；
- 依赖声明采用 `>=` 下限，全新环境会随时间解析到更新版本，需定期回归；
- 后续可扩展方向：自动生成并迭代“特征工程/数据处理代码”而不仅是模型参数、
  多 run 取均值汇报、断点恢复。

## 6. 复现索引

### 6.1 零依赖冒烟（无需 Kaggle / API key）

```bash
python run_agent.py --config config.yaml --brain rule --max-steps 3
```

### 6.2 一键批量 / 单任务运行与提交

```bash
python run_agent.py --config configs/titanic.yaml --brain rule --max-steps 3
python run_agent.py --config configs/titanic.yaml --brain llm --max-steps 3
python experiments/validate_submission.py \
    --submission submissions/best_titanic_submission.csv \
    --sample data/raw/titanic/gender_submission.csv \
    --id-col PassengerId --required PassengerId,Survived

# 一键批量（数据缺失自动下载）
python run_all.py --brain llm --max-steps 3

# 断点续跑（自动最新 run / 指定 run / 批量续跑）
python run_agent.py --config configs/titanic.yaml --resume
python run_agent.py --config configs/titanic.yaml \
    --resume runs/titanic/<timestamp>
python run_all.py --resume --brain llm --max-steps 6
```

### 6.3 已入库的可复现产物

| 内容 | 路径 |
| --- | --- |
| 5 份全局最佳参数 | submissions/best_<task_id>_params.yaml |
| 5 份全局最佳提交 | submissions/best_<task_id>_submission.csv |
| titanic LLM 3 轮完整运行日志 | reports/example_runs/titanic_llm_3_rounds_20260907_193841/ |

说明：产生上述全局最佳的原始 run 目录（gitignore）仍在本地，报告第 4.2 节
给出的 run_id 可对应到本地 `runs/<task_id>/`；参数 YAML 与示例完整日志已提交，
因此新 clone 无需历史 run 目录即可复现提交文件与查看完整决策链。
