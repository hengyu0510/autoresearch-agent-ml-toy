# AGENTS.md

本文件供 Codex 在本仓库内工作时参考。任务的权威说明是 `init.md`；本文件
补充当前已确定的架构、工作约定、目录结构与验收状态。若与 `init.md` 冲突，
以本文件为准，并向用户说明差异。

## 1. 项目目标

在普通电脑上搭建一个"自动迭代机器学习实验的 Agent"。用户给出自然语言研究
目标后，系统需自主完成：

1. 读取研究目标、现有代码/配置及可用数据，并检查当前实验状态；
2. 提出下一步可执行的修改方案；
3. 修改代码或配置，通过 Python / Shell 真实运行实验；
4. 读取日志与指标，独立判断修改是否有效，决定继续、回退或停止；
5. 至少完成 3 轮真实实验迭代，最终输出包含实验过程、最佳结果与局限性的报告。

## 2. 已确定的设计决策

| 主题 | 当前决策 |
| --- | --- |
| 实验任务 | 5 个 Kaggle 任务（`tasks/tasks.yaml`）：titanic、house_prices、bike_sharing_demand、digit_recognizer、facial_keypoints；`experiments/train.py` 保留 sklearn 乳腺癌数据作为通用示例 |
| 系统形态 | Python 自治进程（单任务入口 `run_agent.py`，批量入口 `run_all.py`）；决策大脑可插拔：`rule`（无 Key、确定性候选）与 `llm`（真实 API，默认 DeepSeek deepseek-v4-flash） |
| 数据与指标 | 每个任务固定 seed 划分；决策只看 `val_*` 指标，最终只报 `test_*`；指标方向由 `run.higher_is_better` 配置（bike 按时间顺序切分，facial 按行切分并逐坐标训练） |
| 迭代闭环 | planner → executor → evaluator → state → 决策（继续/回退/停止）；每步参数快照与 stdout/stderr 落盘 |
| 失败与恢复 | LLM 空响应/截断 JSON 自动重试一次，再失败回退 rule；步骤错误先读 traceback 再重试；连续致命错误按阈值退出 |
| 终止机制 | 各任务 config 默认 `max_steps=6`、`min_iterations=3`；提升不足达到收敛轮数、达到最大步数、连续失败三类退出均写入 state；`stop_failure`/0 轮成功/提交失败映射为非零退出码 |
| 状态记录 | `runs/<task_id>/<timestamp>/state.jsonl` 全量事件 + `checkpoint.json` 断点；含输入、决策、工具调用、输出、错误与下一步动作 |
| 断点续跑 | `run_agent --resume [run_dir|latest]` / `run_all --resume`：从 checkpoint 的 next_step 恢复大脑游标、最佳参数、收敛状态；无 checkpoint 旧 run 从 state.jsonl 重建；已正常完成 run 拒绝恢复 |
| 可配置项 | 任务与 `rule_candidates`、指标方向、LLM provider/model/base_url/api_key_env/reasoning_effort/max_tokens、运行预算与阈值 |
| 提交管线 | 每任务 `submit.py` 用全部有标签数据重训并输出 Kaggle 格式文件；run_agent 收尾自动用最佳快照生成提交并调用 `experiments/validate_submission.py` 校验（行序/数值/NaN/Inf）；可显式 `--kaggle-upload` 实际上传，默认不访问网络 |
| 可复现性 | 单任务：`python run_agent.py --config configs/<task_id>.yaml --brain rule|llm --max-steps 3`；全局最佳参数提交为 `submissions/best_<task_id>_params.yaml`，示例完整运行入库于 `reports/example_runs/` |
| 批量入口 | `python run_all.py [--tasks <id,...>] [--brain rule|llm] [--max-steps N]`：数据缺失自动下载，按序运行一个或全部任务，单任务失败不中断，汇总报告写入 `runs/batch/<timestamp>/` |

### 2.1 断点续跑语义

每个 step 真正“完成”（成功采纳/回退、失败已消耗并继续、触发停止或耗尽候选）后，
run_agent 会把内部状态写入 `runs/<task_id>/<timestamp>/checkpoint.json`：

- 下一步应执行的 `next_step`、成功轮数、连续失败计数；
- `best_metrics`、`last_metrics`、最佳参数文件路径；
- 收敛所需的 `previous_value` 与 `improvements` 历史；
- 决策大脑游标：rule 的候选 index；LLM 场景下其 fallback rule 的 index。

恢复入口与约束：

- `run_agent --resume`（= latest）或 `--resume <run_dir>`；
- `run_all --resume` 对每个任务续跑其各自最新 run；
- config 文件路径与 `brain.type` 必须与 checkpoint 一致；
- 已正常完成（exit_code=0）的 run 拒绝恢复；
- 旧 run 无 checkpoint 时从 state.jsonl 重建（成功轮数/最佳参数/rule 游标，
  收敛历史近似重置）。

## 3. 工作约定

1. **真实执行**：所有"运行/修改"必须是真实工具调用（文件读写、Python / Shell
   执行），禁止仅在对话中声称已运行。
2. **先读后改**：动手前阅读 `init.md`、本文件与现有代码；不破坏 `init.md`。
3. **状态先行**：每步修改或实验先写状态，出错先记录 traceback 再修复/回退。
   每个已完成 step 必须同步更新 checkpoint.json，保证任意时刻被杀都可恢复；
4. **独立验证**：检查进程退出码、输出文件与指标合理性；每轮加入反思，不以
   训练集指标下结论。
5. **小步有据**：每轮基于明确假设或上轮结论，避免盲目反复调参。
6. **不写死**：模型、API、数据路径、预算一律来自配置文件或命令行。
7. **密钥安全**：密钥仅存 `.env`；注意系统注入的 `OPENAI_API_KEY` 会覆盖
   `.env` 同名值，外部 LLM 密钥应使用独立变量（如 `DEEPSEEK_API_KEY`）并在
   `brain.api_key_env` 显式指定。
8. **兼容性清洗**：LLM 可能提出旧版超参（如 `max_features: auto`），
   `experiments/common.sanitize_model_params` 统一清洗后再构造模型。
9. **文档语言**：面向用户的文档默认中文。

## 4. 当前目录结构

```text
Autoresearch-agent-ml/
├── AGENTS.md              # 本文件（架构与验收状态）
├── init.md                # 原始任务说明（只读）
├── README.md              # 环境、安装、运行、目录、已知限制
├── config.yaml            # 乳腺癌示例的默认配置
├── configs/<task_id>.yaml # 5 个 Kaggle 任务的 Agent 配置
├── run_agent.py           # 单任务主入口
├── run_all.py             # 一键批量入口（可选：全部/部分任务顺序运行）
├── agent/                 # config/logger/state/planner/executor/evaluator/resume
├── data/
│   ├── fetch.py           # Kaggle 下载与缓存
│   ├── kaggle_upload.py   # 可选的 Kaggle CLI 提交上传
│   ├── raw/               # 解压后的原始数据（gitignore）
│   └── archives/          # 压缩包缓存（gitignore）
├── tasks/                 # tasks.yaml + registry.py
├── experiments/
│   ├── common.py          # 共享 CLI/参数/指标/超参清洗
│   ├── validate_submission.py
│   ├── train.py           # 乳腺癌示例
│   └── tasks/<task_id>/   # train.py + baseline.yaml + submit.py
├── submissions/           # 提交文件、指标与最佳参数 YAML（已校验）
├── runs/<task_id>/        # 各任务运行产物 + checkpoint.json（gitignore）
└── reports/               # technical_report.md + example_runs/（完整运行日志已入库）
```

## 5. 验收状态（Definition of Done）

已完成：

- [x] README 提供环境安装到运行的完整命令，单条主命令可复现；
- [x] `run_all.py` 一键批量入口：数据缺失自动下载 → 逐任务真实迭代 →
      run_agent 收尾自动提交 → `runs/batch/<timestamp>/BATCH_REPORT.md` 汇总；
      `--dry-run` 可预览、`--tasks` 可只跑子集、单任务失败不中断整批；
- [x] 5 个任务在 LLM 大脑下均完成 3 轮真实迭代；rule 大脑在
      titanic/house_prices/bike_sharing_demand 完成 3 轮、在 digit/facial
      完成闭环冒烟；日志与 `state.jsonl`/`summary.json` 可对应；
- [x] 真实失败与恢复：LLM 空响应/截断 JSON 自动重试成功、指标下降回退、
      `stop_failure` 退出原因均真实出现过；
- [x] 独立验证：固定 train/val/test，决策只看 val，最终报告 test；
- [x] 终止机制三类（max_steps / 收敛 / 连续失败）均已触发并被记录；
- [x] 5 个任务提交管线打通，提交文件通过官方 sample 格式校验；
- [x] run_agent 收尾阶段已加入"最佳参数快照 → 自动生成提交 → 格式校验"，
      产物写入 `runs/<task_id>/<timestamp>/submissions/`；
- [x] Agent 失败态可检测：`stop_failure`/0 轮成功/提交失败 -> run_agent 非零退出，
      run_all 汇总为 `failed` 而非 `ok`；
- [x] `--resume` 断点续跑：checkpoint 每完成一步落盘；规则/LLM 大脑游标、最佳
      参数、收敛与失败计数可恢复；旧 run 兼容从 state.jsonl 重建；run_all 可
      批量续跑各任务最新 run；
- [x] LLM params 程序级白名单与资源/超参边界校验，非白名单字段自动忽略；
- [x] 提交管线修复：bike/digit 保留原始行序与 ImageId；validator 增加数值/NaN/Inf
      校验；`--kaggle-upload` 可显式实际上传；
- [x] `reports/technical_report.md`、最佳参数 YAML 与示例完整运行已入库，clone 后可复现；
- [x] 无密钥提交、路径可配置、无临时调试残留。

待办：

- [ ] 为本次 LLM 迭代前的历史 run 补生成提交，或直接运行新的 run 由收尾
      阶段自动生成；
- [ ] 可选：git 集成、更多轮次的分任务深入研究。

## 6. 当前运行参考（2026-09-08，LLM = deepseek-v4-flash, effort=max）

| 任务 | 最佳实验 | val | test |
| --- | --- | --- | --- |
| titanic | MLP(16), scaler | acc 0.8258 | acc 0.8268 |
| house_prices | XGBoost | rmsle 0.1438 | rmsle 0.1249 |
| bike_sharing_demand | XGBoost | rmsle 0.5163 | rmsle 0.4949 |
| digit_recognizer | MLP(256) | acc 0.9637 | acc 0.9613 |
| facial_keypoints | PCA(64)+Ridge | rmse 3.0897 | rmse 3.0839 |
