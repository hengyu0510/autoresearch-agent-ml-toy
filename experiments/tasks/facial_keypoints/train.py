"""Facial Keypoints Detection baseline。

指标口径：RMSE（Kaggle 官方，按所有可见关键点坐标 cell 聚合）。
标签按列稀疏缺失：这里按“行”切分 train/val/test，再对每个坐标列
用该列在 train 中可见的行单独训练模型，评估时只统计可见 cell。
默认模型为 PCA(64) + Ridge（普通电脑数秒~数十秒一轮）。
baseline 只做本地评估，不含提交。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
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

DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "facial_keypoints" / "training.csv"

DEFAULT_PARAMS = {
    "model": "ridge",
    "scaler": True,
    "pca_components": 64,
    "max_rows": None,
    "hyperparams": {"alpha": 1.0},
}

LABEL_COLS = [
    "left_eye_center_x", "left_eye_center_y",
    "right_eye_center_x", "right_eye_center_y",
    "left_eye_inner_corner_x", "left_eye_inner_corner_y",
    "left_eye_outer_corner_x", "left_eye_outer_corner_y",
    "right_eye_inner_corner_x", "right_eye_inner_corner_y",
    "right_eye_outer_corner_x", "right_eye_outer_corner_y",
    "left_eyebrow_inner_end_x", "left_eyebrow_inner_end_y",
    "left_eyebrow_outer_end_x", "left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x", "right_eyebrow_inner_end_y",
    "right_eyebrow_outer_end_x", "right_eyebrow_outer_end_y",
    "nose_tip_x", "nose_tip_y",
    "mouth_left_corner_x", "mouth_left_corner_y",
    "mouth_right_corner_x", "mouth_right_corner_y",
    "mouth_center_top_lip_x", "mouth_center_top_lip_y",
    "mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y",
]


def parse_images(values: pd.Series) -> np.ndarray:
    """把 Image 像素串逐行解析为 uint8 矩阵（n, 9216）。"""
    parsed = []
    for s in values:
        if not isinstance(s, str) or not s:
            raise ValueError("Image 列存在空值")
        parsed.append(np.array(s.split(" "), dtype=np.uint8))
    return np.stack(parsed)


def load_data(path: str, max_rows: int | None):
    p = Path(path) if path else DEFAULT_DATA
    if not p.exists():
        raise FileNotFoundError(f"Facial Keypoints 数据不存在: {p}")
    df = pd.read_csv(p)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
    labels = df[LABEL_COLS].to_numpy(dtype=np.float32)
    images = parse_images(df["Image"])
    del df
    return images, labels


def build_estimator(model_name: str, hyperparams: dict, seed: int):
    params = sanitize_model_params(model_name, hyperparams)
    if model_name == "ridge":
        return Ridge(**params)
    if "random_state" not in params:
        params["random_state"] = seed
    builders = {
        "random_forest": lambda: RandomForestRegressor(**params),
        "mlp": lambda: MLPRegressor(**params),
    }
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("未安装 xgboost，无法使用 model=xgboost") from exc
        return XGBRegressor(**params)
    if model_name not in builders:
        raise ValueError(
            f"未知模型 {model_name!r}，可用: "
            f"ridge/random_forest/mlp/xgboost"
        )
    return builders[model_name]()


def cell_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true)
    return float(
        np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    )


def fit_predict_coordinate(
    X_tr_pca: np.ndarray, X_va_pca: np.ndarray, X_te_pca: np.ndarray,
    y_all: np.ndarray, tr_idx: np.ndarray,
    coord: int, estimator, scaler: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """训练单个坐标模型，返回该列在 val/test 上的预测。"""
    y_col = y_all[:, coord]
    valid_tr = ~np.isnan(y_col[tr_idx])
    obs_local = np.flatnonzero(valid_tr)
    if len(obs_local) == 0:
        # 该列在 train 中无标签：返回列均值占位，避免后续 NaN 穿透。
        mu = float(np.nanmean(y_col))
        pred_va = np.full(X_va_pca.shape[0], mu)
        pred_te = np.full(X_te_pca.shape[0], mu)
        return pred_va, pred_te
    steps = []
    if scaler:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    pipe = Pipeline(steps)
    pipe.fit(X_tr_pca[obs_local], y_col[tr_idx][valid_tr])
    return pipe.predict(X_va_pca), pipe.predict(X_te_pca)


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args(argv)
    params = load_params(args.params, DEFAULT_PARAMS)

    max_rows = params.get("max_rows")
    images, labels = load_data(args.data_path, max_rows)
    print(f"[train] facial_keypoints: 已加载 n={len(images)}, "
          f"坐标列={labels.shape[1]}, 像素={images.shape[1]}")

    tr, va, te = split_index(
        len(images), seed=args.seed,
        val_size=args.val_size, test_size=args.test_size,
    )
    print(f"[train] n_train={len(tr)}, n_val={len(va)}, n_test={len(te)}")

    X_float = images.astype(np.float32) / 255.0
    pca = PCA(
        n_components=int(params.get("pca_components", 64)),
        svd_solver="randomized",
        random_state=args.seed,
    )
    pca_start = time.perf_counter()
    X_pca = pca.fit_transform(X_float[tr])
    X_va_pca = pca.transform(X_float[va])
    X_te_pca = pca.transform(X_float[te])
    pca_seconds = round(time.perf_counter() - pca_start, 4)
    print(f"[train] PCA: n_components={pca.components_.shape[0]}, "
          f"耗时={pca_seconds}s")

    scaler = bool(params.get("scaler", True))
    n_coords = labels.shape[1]
    pred_va = np.full((len(va), n_coords), np.nan, dtype=np.float32)
    pred_te = np.full((len(te), n_coords), np.nan, dtype=np.float32)
    started = time.perf_counter()
    for coord in range(n_coords):
        est = build_estimator(params["model"], params["hyperparams"], args.seed)
        pv, pt = fit_predict_coordinate(
            X_pca, X_va_pca, X_te_pca, labels, tr, coord, est, scaler
        )
        pred_va[:, coord] = pv
        pred_te[:, coord] = pt
    train_seconds = round(time.perf_counter() - started, 4)

    scores = {
        "val_rmse": cell_rmse(labels[va], pred_va),
        "test_rmse": cell_rmse(labels[te], pred_te),
        "val_cells": int(np.count_nonzero(~np.isnan(labels[va]))),
        "test_cells": int(np.count_nonzero(~np.isnan(labels[te]))),
    }
    print(f"[train] model={params['model']}, scaler={scaler}, "
          f"PCA={pca.components_.shape[0]}，坐标模型训练耗时={train_seconds}s")
    print(f"[train] val_rmse={scores['val_rmse']:.4f}, "
          f"test_rmse={scores['test_rmse']:.4f}")

    metrics = {
        "task": "facial_keypoints",
        "model": params["model"],
        "scaler": scaler,
        "pca_components": int(pca.components_.shape[0]),
        "hyperparams": params["hyperparams"],
        "seed": args.seed,
        "max_rows": max_rows,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "pca_seconds": pca_seconds,
        "train_seconds": train_seconds,
        **scores,
    }
    write_metrics(args.out, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
