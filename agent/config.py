"""配置加载与校验。

配置来源：项目根目录 config.yaml + 命令行覆盖参数。
路径均相对项目根目录解析，便于从任意工作目录启动。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


def load_env_file(project_root: Path | None = None) -> Path | None:
    """加载项目根目录下的 .env（若存在）。

    不覆盖系统环境中已有的同名变量，保证 shell 显式注入的配置优先。
    """
    root = Path(project_root) if project_root else Path.cwd()
    env_path = root / ".env"
    if not env_path.exists():
        return None
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确提示
        raise RuntimeError(
            "缺少 python-dotenv，请先执行: pip install -r requirements.txt"
        ) from exc
    load_dotenv(env_path, override=False)
    return env_path


def _find_project_root(start: Path) -> Path:
    """从配置文件所在目录向上找包含 run_agent.py 的仓库根。"""
    cur = start.resolve()
    while True:
        if (cur / "run_agent.py").exists():
            return cur
        if cur.parent == cur:
            return start
        cur = cur.parent


def _pick(cls, data: dict) -> dict:
    """只保留 dataclass 已声明的字段，忽略未知键。"""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in (data or {}).items() if k in names}


@dataclass
class BrainConfig:
    type: str = "rule"
    provider: str = "openai"      # openai | anthropic（type=llm 时生效）
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""          # 留空则按 provider 自动取 OPENAI/ANTHROPIC_API_KEY
    reasoning_effort: str = ""     # 可选：low | medium | high | max（空则不传）
    temperature: float = 0.2
    max_tokens: int = 800
    fallback_to_rule: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "BrainConfig":
        return cls(**_pick(cls, data))


@dataclass
class ExperimentConfig:
    description: str = ""
    dir: str = "experiments"
    train_script: str = "experiments/train.py"
    data_path: str = ""
    # 提交管线（可选）：运行结束后用最佳快照自动生成提交文件
    submit_script: str = ""
    sample_submission: str = ""
    submission_id_col: str = ""
    submission_required_cols: str = ""
    seed: int = 42
    split_train: float = 0.6
    split_val: float = 0.2
    split_test: float = 0.2
    # rule 大脑的候选方案；为空时使用 planner.py 内置的通用序列。
    rule_candidates: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        data = dict(data or {})
        split = data.pop("split", {}) or {}
        cfg = cls(**_pick(cls, data))
        cfg.split_train = float(split.get("train", cfg.split_train))
        cfg.split_val = float(split.get("val", cfg.split_val))
        cfg.split_test = float(split.get("test", cfg.split_test))
        return cfg

    def validate(self) -> None:
        total = self.split_train + self.split_val + self.split_test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train/val/test 比例之和必须为 1，当前 {total:.4f}"
            )
        if not all(v > 0 for v in
                   (self.split_train, self.split_val, self.split_test)):
            raise ValueError("train/val/test 比例必须全部大于 0")


@dataclass
class RunConfig:
    runs_root: str = "runs"
    max_steps: int = 12
    min_iterations: int = 3
    target_metric: str = "val_auc"
    improve_min: float = 0.003
    convergence_rounds: int = 2
    success_threshold: float | None = None
    max_consecutive_errors: int = 2
    max_attempts_per_step: int = 2
    timeout_seconds: int = 300
    higher_is_better: bool = True   # 指标越大越好还是越小越好

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        return cls(**_pick(cls, data))

    def validate(self) -> None:
        if self.max_steps < 1:
            raise ValueError("run.max_steps 必须 >= 1")
        if self.min_iterations < 1:
            raise ValueError("run.min_iterations 必须 >= 1")
        if self.improve_min < 0:
            raise ValueError("run.improve_min 不能为负")
        if self.max_consecutive_errors < 1:
            raise ValueError("run.max_consecutive_errors 必须 >= 1")


@dataclass
class AgentConfig:
    project_root: Path
    experiment: ExperimentConfig
    brain: BrainConfig
    run: RunConfig
    config_file: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        cfg_path = Path(path).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
        project_root = _find_project_root(cfg_path.parent)
        load_env_file(project_root)
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        experiment = ExperimentConfig.from_dict(raw.get("task", {}))
        brain = BrainConfig.from_dict(raw.get("brain", {}))
        run = RunConfig.from_dict(raw.get("run", {}))
        cfg = cls(
            project_root=project_root,
            experiment=experiment,
            brain=brain,
            run=run,
            config_file=str(cfg_path),
        )
        cfg.apply_llm_env_overrides()
        cfg.validate()
        return cfg

    def validate(self) -> None:
        self.experiment.validate()
        self.run.validate()
        if self.brain.type not in ("rule", "llm"):
            raise ValueError(
                f"brain.type 仅支持 rule/llm，当前: {self.brain.type}"
            )
        if self.brain.provider not in ("openai", "anthropic"):
            raise ValueError(
                "brain.provider 仅支持 openai/anthropic，"
                f"当前: {self.brain.provider}"
            )
        if self.brain.type == "llm" and self.brain.fallback_to_rule is False \
                and not self.brain.model:
            raise ValueError("llm 大脑需要配置 brain.model 或允许 rule 回退")

    @property
    def experiment_dir(self) -> Path:
        return self.project_root / self.experiment.dir

    @property
    def train_script_path(self) -> Path:
        p = Path(self.experiment.train_script)
        return p if p.is_absolute() else (self.project_root / p)

    @property
    def submit_script_path(self) -> Path:
        p = Path(self.experiment.submit_script)
        return p if p.is_absolute() else (self.project_root / p)

    @property
    def sample_submission_path(self) -> Path:
        p = Path(self.experiment.sample_submission)
        return p if p.is_absolute() else (self.project_root / p)

    def apply_llm_env_overrides(self) -> None:
        """允许 .env / 系统环境变量覆盖 LLM 大脑的模型、effort 与端点。"""
        if self.brain.type != "llm":
            return
        prefix = "OPENAI" if self.brain.provider == "openai" else "ANTHROPIC"
        self.brain.model = (
            os.environ.get(f"{prefix}_MODEL")
            or os.environ.get("LLM_MODEL")
            or self.brain.model
        )
        self.brain.reasoning_effort = (
            os.environ.get(f"{prefix}_REASONING_EFFORT")
            or os.environ.get("LLM_REASONING_EFFORT")
            or self.brain.reasoning_effort
        )
        self.brain.base_url = (
            os.environ.get(f"{prefix}_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or self.brain.base_url
        )
