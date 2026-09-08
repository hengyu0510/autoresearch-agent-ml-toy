"""断点续跑：checkpoint 读写与旧 run 的兼容恢复。

新版本 run_agent 会在每个“已完成的 step”落盘 `runs/.../checkpoint.json`：

    {
      "schema_version": 1,
      "run_id": ...,
      "config_file": ...,
      "brain": ...,
      "next_step": ...,
      "successful_rounds": ...,
      "consecutive_failed_steps": ...,
      "best_metrics": ...,
      "best_params_file": ...,
      "previous_value": ...,
      "improvements": [...],
      "brain_state": ...,
    }

没有 checkpoint 的旧 run 会尝试从 state.jsonl 重建可恢复状态；规则大脑的
候选游标按历史 plan 事件（source=rule）次数重建。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import AgentConfig

CHECKPOINT_FILE = "checkpoint.json"
RUN_ID_FORMAT = "%Y%m%d_%H%M%S"


@dataclass
class ResumeState:
    next_step: int = 1
    successful_rounds: int = 0
    consecutive_failed_steps: int = 0
    best_metrics: dict[str, Any] | None = None
    last_metrics: dict[str, Any] | None = None
    best_params_file: str | None = None
    previous_value: float | None = None
    improvements: list[float] = field(default_factory=list)
    brain_state: dict[str, Any] = field(default_factory=dict)
    config_file: str = ""
    brain: str = ""

    def to_dict(self, run_dir: Path, brain: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": run_dir.name,
            "config_file": self.config_file,
            "brain": brain,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "next_step": self.next_step,
            "successful_rounds": self.successful_rounds,
            "consecutive_failed_steps": self.consecutive_failed_steps,
            "best_metrics": self.best_metrics,
            "last_metrics": self.last_metrics,
            "best_params_file": self.best_params_file,
            "previous_value": self.previous_value,
            "improvements": self.improvements,
            "brain_state": self.brain_state,
            "brain": self.brain or brain,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResumeState":
        def _int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        return cls(
            next_step=max(1, _int(data.get("next_step"), 1)),
            successful_rounds=max(0, _int(data.get("successful_rounds"), 0)),
            consecutive_failed_steps=max(
                0, _int(data.get("consecutive_failed_steps"), 0)
            ),
            best_metrics=data.get("best_metrics"),
            last_metrics=data.get("last_metrics"),
            best_params_file=data.get("best_params_file"),
            previous_value=data.get("previous_value"),
            improvements=list(data.get("improvements") or []),
            brain_state=dict(data.get("brain_state") or {}),
            config_file=str(data.get("config_file") or ""),
            brain=str(data.get("brain") or ""),
        )


def save_checkpoint(
    run_dir: Path, state: ResumeState, *, brain: str
) -> Path:
    state.brain = brain
    path = run_dir / CHECKPOINT_FILE
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(state.to_dict(run_dir, brain), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def load_checkpoint(run_dir: Path) -> ResumeState | None:
    path = run_dir / CHECKPOINT_FILE
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ResumeState.from_dict(data)


def _parse_run_id(name: str) -> datetime | None:
    try:
        return datetime.strptime(name, RUN_ID_FORMAT)
    except ValueError:
        return None


def resolve_resume_dir(cfg: AgentConfig, resume_arg: str) -> Path:
    """把 --resume 参数解析为具体 run 目录。"""
    runs_root = Path(cfg.run.runs_root)
    runs_root = (
        runs_root if runs_root.is_absolute() else cfg.project_root / runs_root
    )
    if resume_arg == "latest":
        if not runs_root.is_dir():
            raise FileNotFoundError(f"runs 根目录不存在，无法 --resume latest: {runs_root}")
        candidates = [
            d for d in runs_root.iterdir()
            if d.is_dir() and (d / "state.jsonl").exists()
        ]
        candidates.sort(
            key=lambda d: (
                _parse_run_id(d.name) or datetime.min,
                d.stat().st_mtime,
            ),
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"{runs_root} 下没有可恢复的 run 目录")
        return candidates[0]

    path = Path(resume_arg)
    if not path.is_absolute():
        path = cfg.project_root / path
    if not path.is_dir():
        raise FileNotFoundError(f"要恢复的 run 目录不存在: {path}")
    if not (path / "state.jsonl").exists():
        raise FileNotFoundError(f"{path} 缺少 state.jsonl，无法恢复")
    return path


def _legacy_brain_state(
    entries: list[dict[str, Any]], brain_type: str
) -> dict[str, Any]:
    rule_plan_count = sum(
        1
        for e in entries
        if e.get("phase") == "plan" and e.get("source") == "rule"
    )
    rule_state = {"type": "rule", "rule_index": rule_plan_count}
    if brain_type == "llm":
        return {
            "type": "llm",
            "fallback_rule": rule_state,
        }
    return rule_state


def rebuild_from_state(
    run_dir: Path, cfg: AgentConfig, brain_type: str
) -> ResumeState:
    """从旧 state.jsonl 尽力恢复（无 checkpoint 时的兼容路径）。"""
    entries: list[dict[str, Any]] = []
    for line in (run_dir / "state.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    target = cfg.run.target_metric
    metrics_by_step: dict[int, tuple[int, dict[str, Any]]] = {}
    for e in entries:
        if e.get("phase") == "execute" and e.get("success") is True:
            metrics_by_step[int(e["step"])] = (
                int(e.get("attempt") or 1),
                e.get("metrics") or {},
            )

    best_metrics: dict[str, Any] | None = None
    best_params_file: str | None = None
    previous_value: float | None = None
    evaluated_steps: set[int] = set()
    for e in entries:
        if e.get("phase") != "evaluate":
            continue
        step = int(e["step"])
        evaluated_steps.add(step)
        metrics = e.get("metrics") or {}
        if e.get("decision") == "accept":
            best_metrics = metrics
            attempt, _ = metrics_by_step.get(step, (1, {}))
            snapshot = (
                run_dir
                / "snapshots"
                / f"step{step:03d}_attempt{attempt:03d}_params.yaml"
            )
            best_params_file = (
                str(snapshot.resolve()) if snapshot.exists() else None
            )
        if target in metrics:
            try:
                previous_value = float(metrics[target])
            except (TypeError, ValueError):
                pass

    completed_steps = {
        int(e["step"])
        for e in entries
        if e.get("phase") == "decision" and e.get("step") is not None
    }
    next_step = max(completed_steps, default=0) + 1
    if next_step == 1 and evaluated_steps:
        next_step = max(evaluated_steps)  # 中断在 evaluate 之后、decision 之前
    successful_rounds = len(evaluated_steps)

    return ResumeState(
        next_step=next_step,
        successful_rounds=successful_rounds,
        consecutive_failed_steps=0,
        best_metrics=best_metrics,
        last_metrics=(
            max(metrics_by_step.items(), key=lambda kv: kv[0])[1][1]
            if metrics_by_step else None
        ),
        best_params_file=best_params_file,
        previous_value=previous_value,
        improvements=[],
        brain_state=_legacy_brain_state(entries, brain_type),
        config_file=str(Path(cfg.config_file).resolve()),
    )
