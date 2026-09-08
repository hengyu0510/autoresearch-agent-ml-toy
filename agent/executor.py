"""Executor：把方案落到真实文件，并用 subprocess 执行实验。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import AgentConfig
from .logger import Logger
from .planner import Action


@dataclass
class ExecutionResult:
    success: bool
    step: int
    attempt: int
    params_file: Path | None = None
    metrics: dict[str, Any] | None = None
    stdout_log: Path | None = None
    stderr_log: Path | None = None
    error_tail: str = ""
    duration_sec: float = 0.0


class Executor:
    def __init__(self, cfg: AgentConfig, run_dir: Path, log: Logger):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.log = log
        self.snapshot_dir = self.run_dir / "snapshots"
        self.metrics_dir = self.run_dir / "metrics"
        self.logs_dir = self.run_dir / "logs"
        for d in (self.snapshot_dir, self.metrics_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    def run_experiment(
        self,
        action: Action,
        *,
        step: int,
        attempt: int,
    ) -> ExecutionResult:
        tag = f"step{step:03d}_attempt{attempt:03d}"
        snapshot = self.snapshot_dir / f"{tag}_params.yaml"
        metrics_path = self.metrics_dir / f"{tag}_metrics.json"
        stdout_log = self.logs_dir / f"{tag}_stdout.log"
        stderr_log = self.logs_dir / f"{tag}_stderr.log"

        snapshot.write_text(
            yaml.safe_dump(action.params, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self.log.info(f"参数快照: {snapshot}")

        exp = self.cfg.experiment
        cmd = [
            sys.executable,
            str(self.cfg.train_script_path),
            "--params", str(snapshot),
            "--out", str(metrics_path),
            "--seed", str(exp.seed),
            "--val-size", str(exp.split_val),
            "--test-size", str(exp.split_test),
        ]
        if exp.data_path:
            cmd += ["--data-path", str(exp.data_path)]

        started = time.perf_counter()
        try:
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            proc = subprocess.run(
                cmd,
                cwd=str(self.cfg.project_root),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.cfg.run.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            duration = round(time.perf_counter() - started, 2)
            self.log.error(f"实验超时（>{self.cfg.run.timeout_seconds}s）")
            return ExecutionResult(
                success=False, step=step, attempt=attempt,
                stdout_log=stdout_log, stderr_log=stderr_log,
                error_tail=f"timeout > {self.cfg.run.timeout_seconds}s",
                duration_sec=duration,
            )
        except Exception as exc:
            duration = round(time.perf_counter() - started, 2)
            self.log.error(f"执行进程启动失败: {exc}")
            return ExecutionResult(
                success=False, step=step, attempt=attempt,
                error_tail=str(exc), duration_sec=duration,
            )

        duration = round(time.perf_counter() - started, 2)
        stdout_log.write_text(proc.stdout or "", encoding="utf-8")
        stderr_log.write_text(proc.stderr or "", encoding="utf-8")

        if proc.returncode != 0:
            raw_tail = (proc.stderr or proc.stdout or "").strip()
            error_tail = raw_tail[-3000:]
            self.log.error(
                f"实验失败 rc={proc.returncode}，stderr 尾部已记录到 {stderr_log}"
            )
            return ExecutionResult(
                success=False, step=step, attempt=attempt,
                stdout_log=stdout_log, stderr_log=stderr_log,
                error_tail=error_tail, duration_sec=duration,
            )

        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.log.error(f"指标文件解析失败: {exc}")
            return ExecutionResult(
                success=False, step=step, attempt=attempt,
                stdout_log=stdout_log, stderr_log=stderr_log,
                error_tail=f"metrics 解析失败: {exc}",
                duration_sec=duration,
            )

        target = self.cfg.run.target_metric
        value = metrics.get(target)
        val_txt = (
            f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
        )
        self.log.info(
            f"实验完成 step={step} attempt={attempt}，耗时 {duration}s，"
            f"{target}={val_txt}"
        )
        return ExecutionResult(
            success=True, step=step, attempt=attempt,
            params_file=snapshot,
            metrics=metrics, stdout_log=stdout_log, stderr_log=stderr_log,
            duration_sec=duration,
        )
