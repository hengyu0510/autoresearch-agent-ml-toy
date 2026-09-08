"""提交文件格式校验：与 Kaggle 官方 sample 对比。

用法：
    python experiments/validate_submission.py \
        --submission submissions/titanic_submission.csv \
        --sample data/raw/titanic/gender_submission.csv \
        --id-col PassengerId --required PassengerId,Survived

检查项：必需列齐全、行数与 sample 一致、id 列顺序与值一致。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=str, required=True)
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--id-col", type=str, default="")
    parser.add_argument("--required", type=str, default="")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    sub = pd.read_csv(args.submission)
    sample = pd.read_csv(args.sample)

    required = [c for c in args.required.split(",") if c]
    missing = [c for c in required if c not in sub.columns]
    if missing:
        raise SystemExit(f"[fail] 提交缺少必需列: {missing}")
    if len(sub) != len(sample):
        raise SystemExit(
            f"[fail] 行数不一致: 提交 {len(sub)} vs sample {len(sample)}"
        )
    if args.id_col:
        if args.id_col not in sub.columns or args.id_col not in sample.columns:
            raise SystemExit(
                f"[fail] id 列 {args.id_col!r} 在提交或 sample 中不存在"
            )
        sub_ids = sub[args.id_col].tolist()
        sample_ids = sample[args.id_col].tolist()
        if sub_ids != sample_ids:
            raise SystemExit("[fail] id 列顺序或值不一致")

    # 对照 sample 的数值列做基础类型/NaN/Inf 校验（id 列除外）。
    for col in required:
        if col == args.id_col or col not in sample.columns:
            continue
        if not pd.api.types.is_numeric_dtype(sample[col]):
            continue
        try:
            numeric = pd.to_numeric(sub[col], errors="raise")
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"[fail] 提交列 {col!r} 含非数值: {exc}"
            ) from exc
        if numeric.isna().any():
            raise SystemExit(f"[fail] 提交列 {col!r} 含 NaN/空值")
        if numeric.isin([float("inf"), float("-inf")]).any():
            raise SystemExit(f"[fail] 提交列 {col!r} 含无穷值")
    if args.id_col and args.id_col in sub.columns and sub[args.id_col].isna().any():
        raise SystemExit(f"[fail] id 列 {args.id_col!r} 含 NaN/空值")
    print(
        f"[ok] {Path(args.submission).name}: 列={list(sub.columns)}, "
        f"行数={len(sub)}，与 sample 一致"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
