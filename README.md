# AutoResearch Agent for ML

自动迭代机器学习实验的 Agent：用户给定任务后，系统读取配置与实验状态、
提出下一步修改方案、真实执行 Python 实验、读取指标并独立判断继续/回退/停止，
最终产出可追溯的运行记录、提交文件与报告。

> 当前状态（2026-09-07）：5 个 Kaggle 任务的数据/Baseline/提交管线已打通；
> LLM（deepseek-v4-flash, effort=max）在 5 个任务各完成 3 轮真实迭代，
> rule 大脑完成确定性闭环复现；`reports/technical_report.md` 尚待整理。

## 环境与安装

- Python 3.10+
- 依赖：`scikit-learn`、`numpy`、`pandas`、`xgboost`、`PyYAML`、
  `python-dotenv`、`kaggle==1.8.4`

```bash
python -m venv .venv
# Windows:
.venv\Scripts\python -m pip install -r requirements.txt
# macOS / Linux:
# .venv/bin/python -m pip install -r requirements.txt

# 复制环境变量模板并填入密钥（Kaggle / LLM）
copy .env.example .env     # Windows
# cp .env.example .env     # macOS / Linux

# 激活虚拟环境（后续命令统一使用 python）
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
# source .venv/bin/activate      # macOS / Linux / bash
# 若 PowerShell 提示禁止执行脚本，先运行：
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

密钥统一放在根目录 `.env`（gitignore）。注意：系统/Codex 运行时若已注入同名的
`OPENAI_API_KEY`，会覆盖 `.env` 中的同名值（本项目采用"系统变量优先"），因此
外部 LLM 密钥建议使用独立变量并在 config 中显式指定：

```text
DEEPSEEK_API_KEY=sk-...
OPENAI_MODEL=deepseek-v4-flash
OPENAI_BASE_URL=https://api.deepseek.com/
OPENAI_REASONING_EFFORT=max
```

Kaggle 使用新版 token：Settings → API → Generate New Token，将 `KGAT_xxx`
填入 `KAGGLE_API_TOKEN`（旧版 `kaggle.json` / `KAGGLE_USERNAME`+`KAGGLE_KEY`
仍可选，见 `.env.example`）。

## 3 分钟零依赖冒烟（推荐先跑这个）

该命令**不需要 Kaggle 凭证、不需要 API key、不需要下载任何数据**：它使用
sklearn 内置乳腺癌数据完成 3 轮 rule 大脑真实迭代，用于验证环境/安装是否成功。

```bash
python run_agent.py --config config.yaml --brain rule --max-steps 3

# 若未激活 venv，用全路径也一样：
# .venv\Scripts\python run_agent.py --config config.yaml --brain rule --max-steps 3
```

看到 `成功迭代: 3 轮` 与 `运行目录` 即说明环境可运行；产物在
`runs/<timestamp>/`。

## 从 GitHub 克隆运行（给别人使用时）

```bash
git clone <your-repo-url> Autoresearch-agent-ml
cd Autoresearch-agent-ml
# 然后按上文创建 venv 并安装 requirements.txt
```

注意三点：

- `data/raw/`、`data/archives/`、`runs/`、`.env` 已被 gitignore，clone 后
  不存在；`python -m data.fetch --task <id>` 之前需要 Kaggle 凭证并在网页端
  接受规则；
- 零依赖冒烟（`config.yaml` + rule）不需要 Kaggle 和 API key，clone 后即可
  验证环境；
- `experiments/tasks/facial_keypoints` 与 `digit_recognizer` 数据较大，
  baseline 运行建议保留约 1–2 GB 可用内存。

### 兼容性说明

- Python 3.10+；已在 Windows + Python 3.13 的全新 venv 上验证通过；
- 依赖按“已验证区间”约束（见 `requirements.txt`），避免未来大版本意外
  破坏；未在 macOS/Linux 实测，代码均为纯 Python，理论上可直接运行。

## 快速开始

> 前置条件：Kaggle 账号 + API token（`.env`），并已在网页端接受对应竞赛规则。

```bash
# 0) 一键批量运行全部任务（推荐：配置好后的一行主命令）
#    数据缺失时自动下载 -> 逐个任务真实迭代 -> 收尾自动生成提交
#    -> 汇总报告写到 runs/batch/<timestamp>/BATCH_REPORT.md
python run_all.py --brain llm --max-steps 3

# 只跑部分任务 / rule 大脑（无需 API key）
python run_all.py --tasks titanic,house_prices --brain llm --max-steps 3
python run_all.py --tasks titanic --brain rule --max-steps 3

# 先预览将要执行的下载与训练命令
python run_all.py --dry-run

# ---------- 以下为单任务/手动等价命令（run_all.py 会逐个执行） ----------
# 1) 查看/下载任务数据（需 Kaggle 凭证且已在网页接受规则）
python -m data.fetch --list
python -m data.fetch --task titanic --dry-run
python -m data.fetch --task house_prices

# 2) 运行单个任务 baseline
python experiments/tasks/titanic/train.py \
    --params experiments/tasks/titanic/baseline.yaml --out runs/titanic_baseline.json

# 3) rule 大脑（无需 API key，确定性可复现）
python run_agent.py --config configs/titanic.yaml --brain rule --max-steps 3

# 4) LLM 大脑（DeepSeek deepseek-v4-flash, effort=max）
python run_agent.py --config configs/titanic.yaml --brain llm --max-steps 3
```

运行产物：

```text
runs/<task_id>/<timestamp>/
├── state.jsonl        # 每步决策/工具调用/结果/错误
├── summary.json       # 最佳结果与停止原因
├── RUN_LOG.md         # 可读运行日志
├── snapshots/         # 每轮参数快照（可复现/可喂给 submit.py）
├── metrics/           # 每轮指标 JSON
├── logs/              # 每轮 stdout/stderr
└── agent.log
```

根目录 `config.yaml`（乳腺癌示例）的产物写 `runs/<timestamp>/`。

## 目录结构

```text
Autoresearch-agent-ml/
├── AGENTS.md              # 架构/约定/验收状态
├── init.md                # 原始任务说明（只读）
├── README.md
├── config.yaml            # 乳腺癌示例配置
├── configs/<task_id>.yaml # 5 个 Kaggle 任务的 Agent 配置
├── run_agent.py           # 唯一主入口
├── agent/                 # config / logger / state / planner / executor / evaluator
├── data/                  # fetch.py + raw/ + archives/
├── tasks/                 # tasks.yaml + registry.py
├── experiments/
│   ├── common.py          # 共享 CLI/参数/指标/超参清洗
│   ├── validate_submission.py
│   ├── train.py           # 乳腺癌示例
│   └── tasks/<task_id>/   # train.py + baseline.yaml + submit.py
├── submissions/           # 生成的 Kaggle 提交文件
├── runs/                  # 运行产物（gitignore）
│   └── <task_id>/<timestamp>/submissions/best_submission.csv  # 自动提交产物
└── reports/               # 规划中
```

## 实验任务池

当前 5 个 Kaggle 任务（见 [tasks/tasks.yaml](tasks/tasks.yaml)）：

| 任务 | 类型 | 指标 | 方向 |
| --- | --- | --- | --- |
| `titanic` | 表格二分类 | Accuracy | 越高越好 |
| `house_prices` | 表格回归 | RMSLE | 越低越好 |
| `bike_sharing_demand` | 时序/计数回归 | RMSLE | 越低越好 |
| `digit_recognizer` | 图像分类 | Accuracy | 越高越好 |
| `facial_keypoints` | 图像关键点回归 | RMSE | 越低越好 |

数据在 `data/raw/<task_id>/`；指标方向在 `configs/<task_id>.yaml` 的
`run.higher_is_better` 中配置。

## Baseline 实测（seed=42）

| 任务 | 默认模型 | 切分 | val | test |
| --- | --- | --- | --- | --- |
| `titanic` | Logistic Regression + scaler | 分层随机 | val_acc 0.8202 | test_acc 0.8101 |
| `house_prices` | Ridge + scaler | 随机 | val_rmsle 0.1358 | test_rmsle 0.1287 |
| `bike_sharing_demand` | Random Forest | 按时间顺序 | val_rmsle 0.5166 | test_rmsle 0.5322 |
| `digit_recognizer` | Logistic Regression（20k 行） | 分层随机 | val_acc 0.9077 | test_acc 0.9015 |
| `facial_keypoints` | PCA(64)+Ridge（逐坐标） | 随机（按行） | val_rmse 3.0897 | test_rmse 3.0839 |

## LLM 大脑配置

五个任务 config 的 `brain` 统一使用 OpenAI 兼容接口调用 DeepSeek：

```yaml
brain:
  type: llm                # rule | llm
  provider: openai
  model: ""                # 由 OPENAI_MODEL=deepseek-v4-flash 提供
  base_url: ""             # 由 OPENAI_BASE_URL=https://api.deepseek.com/ 提供
  api_key_env: DEEPSEEK_API_KEY
  reasoning_effort: ""     # 由 OPENAI_REASONING_EFFORT=max 提供
  max_tokens: 32768
  fallback_to_rule: true
```

行为说明：

- `--brain llm` 会在 CLI 覆盖后重新读取 `.env`，因此模型/端点/effort 均生效；
- LLM 空响应或 JSON 截断会自动重试一次，仍失败才回退 rule；
- LLM 提出的旧版超参（如 `max_features: auto`）会被
  `experiments/common.sanitize_model_params` 清洗。

## LLM 实测结果（2026-09-07，每任务 3 轮）

| 任务 | run_id | 最佳实验 | val | test |
| --- | --- | --- | --- | --- |
| `titanic` | `runs/titanic/20260907_193841` | MLP(16), scaler | acc 0.8258 | acc 0.8268 |
| `house_prices` | `runs/house_prices/20260907_194024` | XGBoost | rmsle 0.1438 | rmsle 0.1249 |
| `bike_sharing_demand` | `runs/bike_sharing_demand/20260907_194113` | XGBoost | rmsle 0.5163 | rmsle 0.4949 |
| `digit_recognizer` | `runs/digit_recognizer/20260907_194210` | MLP(256) | acc 0.9637 | acc 0.9613 |
| `facial_keypoints` | `runs/facial_keypoints/20260907_195002` | PCA(64)+Ridge | rmse 3.0897 | rmse 3.0839 |

## 提交管线（Kaggle Submission）

每任务 `submit.py` 用全部有标签数据重训并预测官方 `test.csv`，默认输出到
`submissions/<task_id>_submission.csv`：

| 任务 | 提交列 | 行数 |
| --- | --- | --- |
| `titanic` | `PassengerId,Survived` | 418 |
| `house_prices` | `Id,SalePrice` | 1459 |
| `bike_sharing_demand` | `datetime,count` | 6493 |
| `digit_recognizer` | `ImageId,Label` | 28000 |
| `facial_keypoints` | `RowId,Location` | 27124 |

```bash
python experiments/tasks/titanic/submit.py --out submissions/titanic_submission.csv
python experiments/validate_submission.py \
    --submission submissions/titanic_submission.csv \
    --sample data/raw/titanic/gender_submission.csv \
    --id-col PassengerId --required PassengerId,Survived
```

Agent 正常运行到收尾阶段后会自动执行“最佳快照 → 全量重训 → 生成提交 →
格式校验”，产物与校验结果写入 `runs/<task_id>/<timestamp>/submissions/` 和
`state.jsonl`/`summary.json`（不需要手动指定参数）。

当前 `submissions/` 中的是 baseline 参数产物；用某轮最佳参数重生成：

```bash
# 将 <task_id>/<run_id>/<step> 替换为你自己 run 目录中的实际值，例如：
python experiments/tasks/titanic/submit.py \
    --params runs/<task_id>/<run_id>/snapshots/<step>_params.yaml
```

## 可配置项

| 配置段 | 关键项 | 说明 |
| --- | --- | --- |
| `task` | `description`/`train_script`/`data_path`/`seed`/`split` | 任务目标、脚本、数据与固定划分 |
| `task` | `rule_candidates` | rule 大脑候选方案（空则用内置通用序列） |
| `task` | `submit_script`/`sample_submission`/`submission_id_col`/`submission_required_cols` | 收尾自动提交与格式校验配置 |
| `brain` | `type`/`provider` | `rule` 或 `llm`；`openai`/`anthropic` |
| `brain` | `model`/`base_url`/`api_key_env` | LLM 模型、端点与密钥变量名 |
| `brain` | `reasoning_effort`/`max_tokens` | 思考强度与输出预算 |
| `run` | `target_metric`/`higher_is_better` | 优化目标及方向 |
| `run` | `improve_min`/`convergence_rounds` | 收敛判定 |
| `run` | `max_steps`/`min_iterations` | 预算与最少迭代轮数 |
| `run` | `max_consecutive_errors`/`max_attempts_per_step`/`timeout_seconds` | 失败与超时阈值 |

优先级：命令行 > 环境变量（`.env`/shell）> 配置文件。

## 已知限制

- 提交脚本已验证格式，但 5 个 Kaggle 老竞赛是否仍开放线上提交取决于账号与比赛状态；
- `rule` 大脑是确定性候选序列，无真正"反思式实验设计"；
- 单次运行不支持断点续跑（暂无 `--resume`）；
- `reports/technical_report.md` 与最终交付报告尚未整理；
- `llm` 大脑依赖外部 API：调用失败自动重试一次后回退 rule。
