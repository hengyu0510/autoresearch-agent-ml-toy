"""Digit Recognizer (MNIST) baseline。

指标口径：Accuracy（Kaggle 官方）。像素已归一化到 [0,1]；
scaler=true 时额外做 StandardScaler。
baseline 只做本地 train/val/test 评估，不含提交。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import (  # noqa: E402
    add_common_args,
    configure_utf8_stdio,
    load_params,
    sanitize_model_params,
    split_index,
    write_metrics,
)

DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "digit_recognizer" / "train.csv"

DEFAULT_PARAMS = {
    "model": "logistic_regression",
    "scaler": False,
    "max_rows": 20000,
    "hyperparams": {"C": 1.0, "max_iter": 1000},
}

SUPPORTED_FEATURE_OPS = {
    "add_pixel_mean",
    "add_pixel_std",
    "add_horizontal_symmetry",
    "add_vertical_symmetry",
    "add_center_density",
}


def validate_feature_ops(feature_ops: list[str]) -> list[str]:
    ops = list(feature_ops or [])
    unknown = [op for op in ops if op not in SUPPORTED_FEATURE_OPS]
    if unknown:
        raise ValueError(
            f"不支持的 digit_recognizer 特征算子: {unknown}；"
            f"可用: {sorted(SUPPORTED_FEATURE_OPS)}"
        )
    return ops


def apply_pixel_ops(X: np.ndarray, feature_ops: list[str]) -> np.ndarray:
    """在 [0,1] 像素矩阵上追加白名单全局/对称特征。"""
    ops = validate_feature_ops(feature_ops)
    if not ops:
        return X
    side = int(round(X.shape[1] ** 0.5))
    if side * side != X.shape[1]:
        raise ValueError(f"像素数不是完全平方: {X.shape[1]}")
    img = X.reshape(X.shape[0], side, side).astype(np.float32)
    extras: list[np.ndarray] = []
    for op in ops:
        if op == "add_pixel_mean":
            extras.append(img.mean(axis=(1, 2)))
        elif op == "add_pixel_std":
            extras.append(img.std(axis=(1, 2)))
        elif op == "add_horizontal_symmetry":
            extras.append(np.abs(img - img[:, :, ::-1]).mean(axis=(1, 2)))
        elif op == "add_vertical_symmetry":
            extras.append(np.abs(img - img[:, ::-1, :]).mean(axis=(1, 2)))
        elif op == "add_center_density":
            h0, h1 = side // 4, side - side // 4
            center_mean = img[:, h0:h1, h0:h1].mean(axis=(1, 2))
            extras.append(center_mean - img.mean(axis=(1, 2)))
    extra = np.stack(extras, axis=1).astype(np.float32)
    return np.concatenate([X.astype(np.float32), extra], axis=1)


def load_data(path: str, max_rows: int | None, seed: int):
    p = Path(path) if path else DEFAULT_DATA
    if not p.exists():
        raise FileNotFoundError(f"Digit Recognizer 数据不存在: {p}")
    df = pd.read_csv(p, nrows=max_rows)
    y = df["label"].to_numpy(dtype=np.uint8)
    X = df.drop(columns=["label"]).to_numpy(dtype=np.uint8)
    return X, y, len(df)


def build_model(model_name: str, hyperparams: dict, seed: int):
    params = sanitize_model_params(model_name, hyperparams)
    if "random_state" not in params:
        params["random_state"] = seed
    builders = {
        "logistic_regression": lambda: LogisticRegression(**params),
        "random_forest": lambda: RandomForestClassifier(**params),
        "mlp": lambda: MLPClassifier(**params),
    }
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("未安装 xgboost，无法使用 model=xgboost") from exc
        return XGBClassifier(**params)
    if model_name not in builders:
        raise ValueError(
            f"未知模型 {model_name!r}，可用: {sorted(builders)} + xgboost"
        )
    return builders[model_name]()


def evaluate(model, X_val, y_val, X_test, y_test) -> dict:
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)
    return {
        "val_acc": float(accuracy_score(y_val, pred_val)),
        "test_acc": float(accuracy_score(y_test, pred_test)),
    }


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args(argv)
    params = load_params(args.params, DEFAULT_PARAMS)
    feature_ops = validate_feature_ops(params.get("feature_ops") or [])

    max_rows = params.get("max_rows")
    X, y, n_loaded = load_data(args.data_path, max_rows, seed=args.seed)
    X = X.astype(np.float32) / 255.0
    X = apply_pixel_ops(X, feature_ops)
    print(f"[train] digit_recognizer: 已加载 n={n_loaded}，"
          f"features={X.shape[1]}（含 feature_ops={feature_ops}）")

    tr, va, te = split_index(
        len(X), seed=args.seed,
        val_size=args.val_size, test_size=args.test_size, stratify=y,
    )
    X_tr, X_va, X_te = X[tr], X[va], X[te]
    y_tr, y_va, y_te = y[tr], y[va], y[te]
    print(f"[train] n_train={len(tr)}, n_val={len(va)}, n_test={len(te)}")

    scaler = bool(params.get("scaler", False))
    if scaler:
        std = StandardScaler().fit(X_tr)
        X_tr, X_va, X_te = (
            std.transform(X_tr),
            std.transform(X_va),
            std.transform(X_te),
        )

    model = build_model(params["model"], params["hyperparams"], seed=args.seed)
    started = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_seconds = round(time.perf_counter() - started, 4)
    scores = evaluate(model, X_va, y_va, X_te, y_te)
    print(f"[train] model={params['model']}, scaler={scaler}, "
          f"训练耗时={train_seconds}s")
    print(f"[train] val_acc={scores['val_acc']:.4f}, "
          f"test_acc={scores['test_acc']:.4f}")

    metrics = {
        "task": "digit_recognizer",
        "model": params["model"],
        "scaler": scaler,
        "hyperparams": params["hyperparams"],
        "feature_ops": feature_ops,
        "seed": args.seed,
        "max_rows": max_rows,
        "n_loaded": int(n_loaded),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "train_seconds": train_seconds,
        **scores,
    }
    write_metrics(args.out, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
