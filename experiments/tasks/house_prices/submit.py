"""House Prices 提交管线：全量训练 + 官方 test.csv 预测。

用法：
    python experiments/tasks/house_prices/submit.py \
        --params experiments/tasks/house_prices/baseline.yaml \
        --out submissions/house_prices_submission.csv

输出格式：Id,SalePrice（与 sample 一致）。
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
from experiments.tasks.house_prices import train as t  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "submissions" / "house_prices_submission.csv"


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
    X_train, y_train = t.load_data(args.data_path)

    test_path = Path(args.test_path or PROJECT_ROOT / "data" / "raw" / "house_prices" / "test.csv")
    test_raw = pd.read_csv(test_path)
    X_test = test_raw.drop(columns=["Id"])
    if "SalePrice" in X_test.columns:
        X_test = X_test.drop(columns=["SalePrice"])

    scaler = bool(params.get("scaler", True))
    model = t.build_model(params["model"], params["hyperparams"], seed=args.seed)
    pre = t.make_preprocessor(X_train, scaler)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, np.log1p(y_train))

    pred = np.expm1(pipe.predict(X_test))
    pred = np.clip(pred, 0, None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Id": test_raw["Id"], "SalePrice": pred}).to_csv(
        out, index=False
    )
    print(f"[submit] house_prices: 已写入 {out}（{len(pred)} 行）")

    metrics = {
        "task": "house_prices",
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
