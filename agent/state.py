"""运行状态记录：每一步输入、决策、工具调用、输出与错误全部落盘 JSONL。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .logger import Logger


class RunState:
    def __init__(self, run_dir: Path, log: Logger):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.state_file = self.run_dir / "state.jsonl"
        self.entries: list[dict[str, Any]] = []

    def append(
        self,
        phase: str,
        *,
        step: int | None = None,
        message: str = "",
        **fields: Any,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "phase": phase,
        }
        if step is not None:
            entry["step"] = step
        if message:
            entry["message"] = message
        entry.update(fields)
        self.entries.append(entry)
        with self.state_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry

    def last_phase(self, phase: str) -> dict[str, Any] | None:
        for entry in reversed(self.entries):
            if entry.get("phase") == phase:
                return entry
        return None

    def events(self, phase: str) -> list[dict[str, Any]]:
        return [e for e in self.entries if e.get("phase") == phase]

    def successful_runs(self) -> list[dict[str, Any]]:
        """返回每次成功运行的 execute 事件（含 metrics）。"""
        return [
            e for e in self.entries
            if e.get("phase") == "execute" and e.get("success") is True
        ]

    def render_markdown(self, title: str = "AutoResearch Agent 运行日志") -> str:
        lines = [f"# {title}", ""]
        if not self.entries:
            lines.append("（无事件）")
        for e in self.entries:
            ts = e.get("ts", "")[11:19]
            step = e.get("step")
            step_txt = f"step={step}" if step is not None else "     "
            phase = e.get("phase", "?")
            message = e.get("message", "")
            details = message
            if not details:
                compact = {
                    k: v for k, v in e.items()
                    if k not in ("ts", "phase", "step", "message")
                }
                details = json.dumps(compact, ensure_ascii=False, default=str)
            lines.append(f"- `{ts}` {step_txt} **{phase}** — {details}")
        return "\n".join(lines) + "\n"
