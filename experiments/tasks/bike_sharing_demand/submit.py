"""Bike Sharing Demand 提交管线：全量训练 + 官方 test.csv 预测。

用法：
    python experiments/tasks/bike_sharing_demand/submit.py \
        --params experiments/tasks/bike_sharing_demand/baseline.yaml \
        --out submissions/bike_sharing_demand_submission.csv

输出格式：datetime,count（与 sample 一致，count 取整为非负整数）。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import configure_utf8_stdio, write_metrics  # noqa: E402
from experiments.tasks.bike_sharing_demand import train as t  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "submissions" / "bike_sharing_demand_submission.csv"


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=str, default="")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--metrics-out", type=str, default="")
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--test-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    params = t.load_params(args.params, t.DEFAULT_PARAMS)
    train_df = t.load_data(args.data_path)
    y_train = train_df["count"].to_numpy(dtype=float)
    X_train = train_df.drop(columns=["count"])

    test_path = Path(args.test_path or PROJECT_ROOT / "data" / "raw" / "bike_sharing_demand" / "test.csv")
    test_raw = pd.read_csv(test_path)
    original = test_raw.copy()
    if "count" in test_raw.columns:
        test_raw = test_raw.drop(columns=["count"])
    # make_features 保持原始行顺序，避免 sort_values 后与 datetime 输出错位。
    X_test = t.make_features(test_raw)
    if len(test_raw) != len(X_test):
        raise ValueError("test 特征行数与原始 test.csv 不一致")

    scaler = bool(params.get("scaler", True))
    model = t.build_model(params["model"], params["hyperparams"], seed=args.seed)
    pre = t.make_preprocessor(X_train, scaler)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, np.log1p(y_train))

    pred = np.clip(np.rint(np.expm1(pipe.predict(X_test))), 0, None).astype(int)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"datetime": original["datetime"], "count": pred}).to_csv(
        out, index=False
    )
    print(f"[submit] bike_sharing_demand: 已写入 {out}（{len(pred)} 行）")

    metrics = {
        "task": "bike_sharing_demand",
        "mode": "submission",
        "model": params["model"],
        "scaler": scaler,
        "hyperparams": params["hyperparams"],
        "seed": args.seed,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "submission": str(out),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_metrics(args.metrics_out or str(out.with_suffix(".metrics.json")), metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
