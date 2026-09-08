"""House Prices - Advanced Regression Techniques baseline。

指标口径：RMSLE（Kaggle 官方指标，对 SalePrice 做 log1p 后求 RMSE）。
baseline 只做本地 train/val/test 评估，不含提交。
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
    split_index,
    write_metrics,
)

DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "house_prices" / "train.csv"

DEFAULT_PARAMS = {
    "model": "ridge",
    "scaler": True,
    "hyperparams": {"alpha": 1.0},
}

SUPPORTED_FEATURE_OPS = {
    "add_total_sf",
    "add_total_bath",
    "add_has_pool",
    "add_has_garage",
    "add_house_age",
}


def validate_feature_ops(feature_ops: list[str]) -> list[str]:
    ops = list(feature_ops or [])
    unknown = [op for op in ops if op not in SUPPORTED_FEATURE_OPS]
    if unknown:
        raise ValueError(
            f"不支持的 house_prices 特征算子: {unknown}；"
            f"可用: {sorted(SUPPORTED_FEATURE_OPS)}"
        )
    return ops


def derive_features(
    df: pd.DataFrame, feature_ops: list[str] | None = None
) -> pd.DataFrame:
    """在原始行上应用白名单特征算子（不依赖 Id/target）。"""
    ops = validate_feature_ops(feature_ops)
    out = df.copy()
    for op in ops:
        if op == "add_total_sf":
            cols = ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]
            out["TotalSF"] = out[cols].fillna(0).sum(axis=1)
        elif op == "add_total_bath":
            out["TotalBath"] = (
                out["FullBath"].fillna(0)
                + 0.5 * out["HalfBath"].fillna(0)
                + out["BsmtFullBath"].fillna(0)
                + 0.5 * out["BsmtHalfBath"].fillna(0)
            )
        elif op == "add_has_pool":
            out["HasPool"] = (out["PoolArea"].fillna(0) > 0).astype(int)
        elif op == "add_has_garage":
            has_cars = out["GarageCars"].fillna(0) > 0
            has_area = out["GarageArea"].fillna(0) > 0
            out["HasGarage"] = (has_cars | has_area).astype(int)
        elif op == "add_house_age":
            out["HouseAge"] = (
                out["YrSold"].fillna(out["YearBuilt"])
                - out["YearBuilt"].fillna(out["YrSold"])
            )
    return out


def load_data(
    path: str, feature_ops: list[str] | None = None
) -> tuple[pd.DataFrame, np.ndarray]:
    p = Path(path) if path else DEFAULT_DATA
    if not p.exists():
        raise FileNotFoundError(f"House Prices 数据不存在: {p}")
    df = derive_features(pd.read_csv(p), feature_ops)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    y = df["SalePrice"].to_numpy(dtype=float)
    X = df.drop(columns=["SalePrice"])
    return X, y


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

    X, y = load_data(args.data_path, feature_ops)
    tr, va, te = split_index(
        len(X), seed=args.seed,
        val_size=args.val_size, test_size=args.test_size,
    )
    X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X.iloc[te]
    y_tr, y_va, y_te = y[tr], y[va], y[te]
    print(f"[train] house_prices: n_train={len(tr)}, n_val={len(va)}, "
          f"n_test={len(te)}")

    scaler = bool(params.get("scaler", True))
    model = build_model(params["model"], params["hyperparams"], seed=args.seed)
    pre = make_preprocessor(X_tr, scaler)
    pipe = Pipeline([("pre", pre), ("model", model)])

    started = time.perf_counter()
    # 对强偏态目标做 log1p 变换，训练目标即 log1p(SalePrice)。
    pipe.fit(X_tr, np.log1p(y_tr))
    train_seconds = round(time.perf_counter() - started, 4)
    scores = evaluate(pipe, X_va, y_va, X_te, y_te)
    print(f"[train] model={params['model']}, scaler={scaler}, "
          f"训练耗时={train_seconds}s")
    print(f"[train] val_rmsle={scores['val_rmsle']:.4f}, "
          f"test_rmsle={scores['test_rmsle']:.4f}")

    metrics = {
        "task": "house_prices",
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
