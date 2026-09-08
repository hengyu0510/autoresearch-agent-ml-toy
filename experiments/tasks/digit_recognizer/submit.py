"""Digit Recognizer 提交管线：全量训练 + 官方 test.csv 预测。

用法：
    python experiments/tasks/digit_recognizer/submit.py \
        --params experiments/tasks/digit_recognizer/baseline.yaml \
        --out submissions/digit_recognizer_submission.csv

输出格式：ImageId,Label（与 sample 一致）。提交默认用全量 42000 行训练，
忽略 baseline 参数中的 max_rows 采样限制。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import configure_utf8_stdio, write_metrics  # noqa: E402
from experiments.tasks.digit_recognizer import train as t  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "submissions" / "digit_recognizer_submission.csv"


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
    feature_ops = t.validate_feature_ops(params.get("feature_ops") or [])

    # 提交模式始终使用全部训练数据，不使用 baseline 里的 max_rows 采样。
    X_train, y_train, n_train = t.load_data(args.data_path, None, seed=args.seed)
    X_train = X_train.astype(np.float32) / 255.0
    X_train = t.apply_pixel_ops(X_train, feature_ops)

    test_path = Path(args.test_path or PROJECT_ROOT / "data" / "raw" / "digit_recognizer" / "test.csv")
    test_raw = pd.read_csv(test_path)
    if "ImageId" in test_raw.columns:
        image_ids = test_raw["ImageId"].to_numpy()
        test_raw = test_raw.drop(columns=["ImageId"])
    else:
        image_ids = np.arange(1, len(test_raw) + 1)
    X_test = test_raw.to_numpy(dtype=np.uint8)
    X_test = X_test.astype(np.float32) / 255.0
    X_test = t.apply_pixel_ops(X_test, feature_ops)
    if len(image_ids) != len(X_test):
        raise ValueError("ImageId 行数与像素矩阵不一致")

    scaler = bool(params.get("scaler", False))
    if scaler:
        std = StandardScaler().fit(X_train)
        X_train = std.transform(X_train)
        X_test = std.transform(X_test)

    model = t.build_model(params["model"], params["hyperparams"], seed=args.seed)
    model.fit(X_train, y_train)
    pred = model.predict(X_test).astype(int)

    if not set(pred) <= set(range(10)):
        raise ValueError(f"预测类别超出 0-9: {sorted(set(pred))}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ImageId": image_ids, "Label": pred}).to_csv(
        out, index=False
    )
    print(f"[submit] digit_recognizer: 已写入 {out}（{len(pred)} 行）")

    metrics = {
        "task": "digit_recognizer",
        "mode": "submission",
        "model": params["model"],
        "scaler": scaler,
        "hyperparams": params["hyperparams"],
        "feature_ops": feature_ops,
        "seed": args.seed,
        "n_train": int(n_train),
        "n_test": int(len(X_test)),
        "submission": str(out),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_metrics(args.metrics_out or str(out.with_suffix(".metrics.json")), metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
