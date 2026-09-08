"""共享工具：各任务 train.py 的统一 CLI、参数加载与指标落盘。

约定：
- 每个任务脚本均接受 --params/--out/--seed/--val-size/--test-size/--data-path；
- params 结构统一为 {model, scaler, hyperparams, ...}；
- 输出 JSON 同时包含 val_<metric> 与 test_<metric>，供 Agent 独立评估。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


def configure_utf8_stdio() -> None:
    """将 stdout/stderr 固定为 UTF-8，避免 Windows 管道乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--params", type=str, default="",
        help="参数文件（YAML/JSON）；留空使用内置默认 baseline",
    )
    parser.add_argument(
        "--out", type=str, default="metrics.json", help="指标 JSON 输出路径"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--data-path", type=str, default="",
        help="任务数据 CSV 路径；留空使用默认原始数据文件",
    )


def load_params(path: str, defaults: dict[str, Any]) -> dict[str, Any]:
    """读取参数并合并到任务默认值。

    顶层做浅合并；若文件提供了 hyperparams 则整体替换默认 hyperparams，
    避免把某一模型的默认超参泄漏到另一个模型（例如 ridge 的 alpha
    不应传给 random_forest）。
    """
    params = dict(defaults)
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"params 文件不存在: {p}")
        text = p.read_text(encoding="utf-8")
        data = (
            json.loads(text)
            if p.suffix.lower() == ".json"
            else yaml.safe_load(text)
        )
        if not isinstance(data, dict):
            raise ValueError(f"params 必须是映射结构: {p}")
        params.update(data)
    return params


def write_metrics(out: str, metrics: dict[str, Any]) -> None:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[train] 指标已写入: {p.resolve()}")


def split_index(
    total: int,
    *,
    seed: int,
    val_size: float,
    test_size: float,
    stratify: Sequence[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按固定 seed 划分 (train, val, test) 行索引。"""
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError(f"无效的 val/test 比例: {val_size}/{test_size}")
    from sklearn.model_selection import train_test_split

    idx = np.arange(total)
    tr, rest = train_test_split(
        idx,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=stratify,
    )
    rest_strat = None
    if stratify is not None:
        arr = np.asarray(stratify)
        rest_strat = arr[rest]
    va, te = train_test_split(
        rest,
        test_size=test_size / (val_size + test_size),
        random_state=seed + 1,
        stratify=rest_strat,
    )
    return tr, va, te


def describe_imputation(df: Any) -> dict[str, int]:
    """返回每列缺失数量（小工具，便于日志）。"""
    return {str(c): int(v) for c, v in df.isna().sum().items() if v}


def sanitize_model_params(model_name: str, hyperparams: dict[str, Any]) -> dict[str, Any]:
    """清洗 LLM 可能提出的与当前 sklearn 版本不兼容的超参数。"""
    params = dict(hyperparams or {})
    if model_name == "random_forest" and params.get("max_features") == "auto":
        # sklearn >=1.4 移除了 RandomForest 的 max_features='auto'
        params["max_features"] = "sqrt"
    return params
