"""Bike Sharing Demand baseline。

指标口径：RMSLE（Kaggle 官方）。数据按时间排序，切分采用“前 train / 中 val /
后 test”的时序顺序划分，避免随机划分把相邻时刻泄漏进验证集。

注意：casual / registered 是 count 的直接组成部分（标签泄漏），不参与训练。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import (  # noqa: E402
    add_common_args,
    configure_utf8_stdio,
    load_params,
    sanitize_model_params,
    write_metrics,
)

DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "bike_sharing_demand" / "train.csv"

DEFAULT_PARAMS = {
    "model": "random_forest",
    "scaler": False,
    "hyperparams": {
        "n_estimators": 400,
        "max_depth": 10,
        "min_samples_leaf": 2,
    },
}

DROP_COLS = ["datetime", "casual", "registered"]
SUPPORTED_FEATURE_OPS = {
    "add_is_working_day",
    "add_is_peak_hour",
    "add_day_period",
    "add_bad_weather",
}


def validate_feature_ops(feature_ops: list[str]) -> list[str]:
    ops = list(feature_ops or [])
    unknown = [op for op in ops if op not in SUPPORTED_FEATURE_OPS]
    if unknown:
        raise ValueError(
            f"不支持的 bike_sharing_demand 特征算子: {unknown}；"
            f"可用: {sorted(SUPPORTED_FEATURE_OPS)}"
        )
    return ops


def make_features(
    df: pd.DataFrame, feature_ops: list[str] | None = None
) -> pd.DataFrame:
    """由原始行（保留顺序）派生 hour/weekday/month 并移除泄漏/标识列。"""
    ops = validate_feature_ops(feature_ops)
    parsed = pd.to_datetime(df["datetime"])
    out = df.copy()
    out["hour"] = parsed.dt.hour
    out["weekday"] = parsed.dt.weekday
    out["month"] = parsed.dt.month
    for op in ops:
        if op == "add_is_working_day":
            holiday = pd.to_numeric(out["holiday"], errors="coerce").fillna(0)
            out["IsWorkingDay"] = (
                (out["weekday"] < 5) & (holiday == 0)
            ).astype(int)
        elif op == "add_is_peak_hour":
            out["IsPeakHour"] = out["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        elif op == "add_day_period":
            out["DayPeriod"] = pd.cut(
                out["hour"],
                bins=[-1, 5, 11, 17, 23],
                labels=["Night", "Morning", "Afternoon", "Evening"],
            ).astype(str)
        elif op == "add_bad_weather":
            weather = pd.to_numeric(out["weather"], errors="coerce").fillna(1)
            out["BadWeather"] = (weather >= 3).astype(int)
    return out.drop(columns=[c for c in DROP_COLS if c in out.columns])


def load_data(
    path: str, feature_ops: list[str] | None = None
) -> pd.DataFrame:
    p = Path(path) if path else DEFAULT_DATA
    if not p.exists():
        raise FileNotFoundError(f"Bike Sharing 数据不存在: {p}")
    df = pd.read_csv(p, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return make_features(df, feature_ops)


def make_preprocessor(X: pd.DataFrame, scaler: bool) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler() if scaler else "passthrough"),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))
    return ColumnTransformer(transformers)


def build_model(model_name: str, hyperparams: dict, seed: int):
    params = sanitize_model_params(model_name, hyperparams)
    builders = {
        "ridge": lambda: Ridge(**params),
        "random_forest": lambda: RandomForestRegressor(**params),
        "mlp": lambda: MLPRegressor(**params),
    }
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("未安装 xgboost，无法使用 model=xgboost") from exc
        if "random_state" not in params:
            params["random_state"] = seed
        return XGBRegressor(**params)
    if model_name in ("random_forest", "mlp") and "random_state" not in params:
        params["random_state"] = seed
    if model_name not in builders:
        raise ValueError(
            f"未知模型 {model_name!r}，可用: {sorted(builders)} + xgboost"
        )
    return builders[model_name]()


def chronological_split(
    n: int, val_size: float, test_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按时间顺序切分：前 1-val-test 为 train，其后 val，最后 test。"""
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError(f"无效的 val/test 比例: {val_size}/{test_size}")
    tr_end = int(round(n * (1 - val_size - test_size)))
    va_end = int(round(n * (1 - test_size)))
    idx = np.arange(n)
    return idx[:tr_end], idx[tr_end:va_end], idx[va_end:]


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.sqrt(
            mean_squared_error(
                np.log1p(np.clip(y_true, 0, None)),
                np.log1p(np.clip(y_pred, 0, None)),
            )
        )
    )


def evaluate(model, X_val, y_val, X_test, y_test) -> dict:
    pred_val = np.expm1(model.predict(X_val))
    pred_test = np.expm1(model.predict(X_test))
    return {
        "val_rmsle": rmsle(y_val, pred_val),
        "test_rmsle": rmsle(y_test, pred_test),
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, pred_val))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
    }


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args(argv)
    params = load_params(args.params, DEFAULT_PARAMS)
    feature_ops = validate_feature_ops(params.get("feature_ops") or [])

    df = load_data(args.data_path, feature_ops)
    y = df["count"].to_numpy(dtype=float)
    X = df.drop(columns=["count"])
    tr, va, te = chronological_split(
        len(df), val_size=args.val_size, test_size=args.test_size
    )
    X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X.iloc[te]
    y_tr, y_va, y_te = y[tr], y[va], y[te]
    print(f"[train] bike_sharing_demand: n_train={len(tr)}, "
          f"n_val={len(va)}, n_test={len(te)}（按时间顺序）")

    scaler = bool(params.get("scaler", True))
    model = build_model(params["model"], params["hyperparams"], seed=args.seed)
    pre = make_preprocessor(X_tr, scaler)
    pipe = Pipeline([("pre", pre), ("model", model)])

    started = time.perf_counter()
    pipe.fit(X_tr, np.log1p(y_tr))
    train_seconds = round(time.perf_counter() - started, 4)
    scores = evaluate(pipe, X_va, y_va, X_te, y_te)
    print(f"[train] model={params['model']}, scaler={scaler}, "
          f"训练耗时={train_seconds}s")
    print(f"[train] val_rmsle={scores['val_rmsle']:.4f}, "
          f"test_rmsle={scores['test_rmsle']:.4f}")

    metrics = {
        "task": "bike_sharing_demand",
        "split": "chronological",
        "model": params["model"],
        "scaler": scaler,
        "hyperparams": params["hyperparams"],
        "feature_ops": feature_ops,
        "seed": args.seed,
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
