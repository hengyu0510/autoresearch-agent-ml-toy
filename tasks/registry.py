"""任务注册表：集中管理 Agent 可用的 Kaggle 实验任务元数据。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

TASKS_FILE = Path(__file__).resolve().parent / "tasks.yaml"


@dataclass(frozen=True)
class TaskSource:
    provider: str          # kaggle_competition | kaggle_dataset | url
    slug: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "TaskSource":
        data = data or {}
        return cls(
            provider=data.get("provider", ""),
            slug=data.get("slug", ""),
        )


@dataclass(frozen=True)
class MLTask:
    id: str
    name: str
    domain: str
    metric: str
    source: TaskSource
    files: tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "MLTask":
        task_id = data["id"]
        return cls(
            id=task_id,
            name=data.get("name", task_id),
            domain=data.get("domain", ""),
            metric=data.get("metric", ""),
            source=TaskSource.from_dict(data.get("source", {})),
            files=tuple(data.get("files", [])),
            notes=data.get("notes", ""),
        )


def load_tasks(path: str | Path = TASKS_FILE) -> tuple[MLTask, ...]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"任务注册表不存在: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{p} 顶层必须是任务列表")
    return tuple(MLTask.from_dict(item) for item in data)


def get_task(task_id: str, tasks: tuple[MLTask, ...] | None = None) -> MLTask:
    if tasks is None:
        tasks = load_tasks()
    for task in tasks:
        if task.id == task_id:
            return task
    available = ", ".join(t.id for t in tasks)
    raise KeyError(f"未知任务 {task_id!r}；可用任务: {available}")
