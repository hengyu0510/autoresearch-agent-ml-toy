"""Titanic - Machine Learning from Disaster baseline。

指标口径：Accuracy（与 Kaggle 一致），附加 val_auc / f1 供参考。
目标列：Survived（0/1）。baseline 只做本地 train/val/test 评估，不含提交。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
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

DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "titanic" / "train.csv"

DEFAULT_PARAMS = {
    "model": "logistic_regression",
    "scaler": True,
    "hyperparams": {"C": 1.0, "max_iter": 2000},
}

NUMERIC_COLS = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
CATEGORICAL_COLS = ["Sex", "Embarked", "CabinLetter"]
DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]


def load_data(path: str) -> pd.DataFrame:
    p = Path(path) if path else DEFAULT_DATA
    if not p.exists():
        raise FileNotFoundError(f"Titanic 数据不存在: {p}")
    df = pd.read_csv(p)
    df["CabinLetter"] = df["Cabin"].fillna("None").str[0]
    drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=drop)


def make_preprocessor(scaler: bool) -> ColumnTransformer:
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
    return ColumnTransformer(
        [
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )


def build_model(model_name: str, hyperparams: dict, seed: int):
    params = sanitize_model_params(model_name, hyperparams)
    if "random_state" not in params:
        params["random_state"] = seed
    builders = {
        "logistic_regression": lambda: LogisticRegression(**params),
        "random_forest": lambda: RandomForestClassifier(**params),
        "mlp": lambda: MLPClassifier(**params),
    }
    if model_name not in builders:
        raise ValueError(
            f"未知模型 {model_name!r}，可用: {sorted(builders)}"
        )
    return builders[model_name]()


def evaluate(model, X_val, y_val, X_test, y_test) -> dict:
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)
    proba_val = model.predict_proba(X_val)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]
    return {
        "val_acc": float(accuracy_score(y_val, pred_val)),
        "test_acc": float(accuracy_score(y_test, pred_test)),
        "val_auc": float(roc_auc_score(y_val, proba_val)),
        "test_auc": float(roc_auc_score(y_test, proba_test)),
        "val_f1": float(f1_score(y_val, pred_val)),
        "test_f1": float(f1_score(y_test, pred_test)),
    }


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args(argv)
    params = load_params(args.params, DEFAULT_PARAMS)

    df = load_data(args.data_path)
    y = df["Survived"].to_numpy(dtype=int)
    X = df.drop(columns=["Survived"])

    tr, va, te = split_index(
        len(df), seed=args.seed,
        val_size=args.val_size, test_size=args.test_size, stratify=y,
    )
    X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X.iloc[te]
    y_tr, y_va, y_te = y[tr], y[va], y[te]
    print(f"[train] titanic: n_train={len(tr)}, n_val={len(va)}, "
          f"n_test={len(te)}")

    scaler = bool(params.get("scaler", True))
    model = build_model(params["model"], params["hyperparams"], seed=args.seed)
    pipe = Pipeline(
        [("pre", make_preprocessor(scaler)), ("model", model)]
    )
    started = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    train_seconds = round(time.perf_counter() - started, 4)
    scores = evaluate(pipe, X_va, y_va, X_te, y_te)
    print(f"[train] model={params['model']}, scaler={scaler}, "
          f"训练耗时={train_seconds}s")
    print(f"[train] val_acc={scores['val_acc']:.4f}, "
          f"test_acc={scores['test_acc']:.4f}")

    metrics = {
        "task": "titanic",
        "model": params["model"],
        "scaler": scaler,
        "hyperparams": params["hyperparams"],
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
