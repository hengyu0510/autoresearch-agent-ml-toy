"""统一日志：同时输出到控制台与运行目录下的 agent.log。"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


def configure_stdio() -> None:
    """将 stdout/stderr 固定为 UTF-8，避免 Windows 管道编码造成乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


class Logger:
    def __init__(self, log_file: Path | None = None, verbose: bool = True):
        self.verbose = verbose
        self._fh = None
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self._fh = log_file.open("a", encoding="utf-8")

    def _emit(self, level: str, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] {message}"
        if self.verbose:
            print(line, flush=True)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()

    def info(self, message: str) -> None:
        self._emit("INFO", message)

    def warning(self, message: str) -> None:
        self._emit("WARN", message)

    def error(self, message: str) -> None:
        self._emit("ERROR", message)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
