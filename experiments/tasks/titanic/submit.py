"""Titanic 提交管线：用全部训练数据重训并预测官方 test.csv。

用法：
    python experiments/tasks/titanic/submit.py \
        --params experiments/tasks/titanic/baseline.yaml \
        --out submissions/titanic_submission.csv

输出格式：PassengerId,Survived（与 sample 一致）。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import configure_utf8_stdio, write_metrics  # noqa: E402
from experiments.tasks.titanic import train as t  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "submissions" / "titanic_submission.csv"


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

    # 训练集：全量特征 + 目标
    train_df = t.load_data(args.data_path)
    y_train = train_df["Survived"].to_numpy(dtype=int)
    X_train = train_df.drop(columns=["Survived"])

    # 测试集：原始 PassengerId 用于写出；t.load_data 保留与训练一致的特征列
    test_path = Path(args.test_path or PROJECT_ROOT / "data" / "raw" / "titanic" / "test.csv")
    test_raw = pd.read_csv(test_path)
    X_test = t.load_data(str(test_path))
    if "Survived" in X_test.columns:
        X_test = X_test.drop(columns=["Survived"])
    if len(test_raw) != len(X_test):
        raise ValueError("test 特征行数与原始 test.csv 不一致")

    scaler = bool(params.get("scaler", True))
    model = t.build_model(params["model"], params["hyperparams"], seed=args.seed)
    pipe = Pipeline([("pre", t.make_preprocessor(scaler)), ("model", model)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test).astype(int)
    if set(pred) - {0, 1}:
        raise ValueError(f"预测值超出 {0, 1}: {sorted(set(pred))}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PassengerId": test_raw["PassengerId"], "Survived": pred}).to_csv(
        out, index=False
    )
    print(f"[submit] titanic: 已写入 {out}（{len(pred)} 行）")

    metrics = {
        "task": "titanic",
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
