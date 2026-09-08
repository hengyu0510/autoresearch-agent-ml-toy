"""实验运行脚本：训练并评估单个 scikit-learn 分类器，输出指标 JSON。

用法：
    python experiments/train.py --params <params.yaml> --out <metrics.json>
                                [--seed N] [--val-size 0.2] [--test-size 0.2]

params 文件格式（YAML/JSON）：
    model: logistic_regression | random_forest | mlp
    scaler: true | false
    hyperparams: { ...模型构造函数参数 }
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

MODEL_BUILDERS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "mlp": MLPClassifier,
}

DEFAULT_PARAMS = {
    "model": "logistic_regression",
    "scaler": False,
    "hyperparams": {"C": 1.0, "max_iter": 2000},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=str, default="",
                        help="模型参数文件（YAML/JSON）；留空使用默认 baseline")
    parser.add_argument("--out", type=str, default="metrics.json",
                        help="指标输出路径")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--data-path", type=str, default="",
                        help="CSV 路径（须含 target 列）；留空使用 sklearn 内置数据")
    return parser.parse_args(argv)


def load_params(path: str) -> dict:
    if not path:
        return dict(DEFAULT_PARAMS)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"params 文件不存在: {p}")
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if p.suffix.lower() != ".json" else json.loads(text)
    if not isinstance(data, dict) or "model" not in data:
        raise ValueError(f"params 文件缺少 model 字段: {p}")
    data.setdefault("scaler", False)
    data.setdefault("hyperparams", {})
    return data


def load_data(data_path: str) -> tuple[list, list, list[str]]:
    """返回 (X, y, feature_names)；支持 sklearn 内置数据或 CSV。"""
    if data_path:
        p = Path(data_path)
        if not p.exists():
            raise FileNotFoundError(f"数据文件不存在: {p}")
        with p.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        if not rows or "target" not in rows[0]:
            raise ValueError("CSV 必须包含 'target' 列")
        cols = [c for c in rows[0] if c != "target"]
        X = [[float(row[c]) for c in cols] for row in rows]
        y = [int(float(row["target"])) for row in rows]
        return X, y, cols
    data = load_breast_cancer()
    return data.data, data.target, list(data.feature_names)


def split_data(X, y, seed: int, val_size: float, test_size: float):
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError(f"无效的数据划分: val={val_size}, test={test_size}")
    X_tr, X_rem, y_tr, y_rem = train_test_split(
        X, y, test_size=val_size + test_size,
        random_state=seed, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rem, y_rem,
        test_size=test_size / (val_size + test_size),
        random_state=seed + 1, stratify=y_rem,
    )
    return X_tr, X_val, X_test, y_tr, y_val, y_test


def build_model(model_name: str, hyperparams: dict, seed: int):
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"未知模型 {model_name!r}，可用: {sorted(MODEL_BUILDERS)}"
        )
    kwargs = dict(hyperparams)
    if "random_state" not in kwargs:
        kwargs["random_state"] = seed
    return MODEL_BUILDERS[model_name](**kwargs)


def evaluate_scores(model, X_val, y_val, X_test, y_test) -> dict:
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)
    proba_val = model.predict_proba(X_val)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]
    return {
        "val_auc": float(roc_auc_score(y_val, proba_val)),
        "test_auc": float(roc_auc_score(y_test, proba_test)),
        "val_f1": float(f1_score(y_val, pred_val)),
        "test_f1": float(f1_score(y_test, pred_test)),
        "val_acc": float(accuracy_score(y_val, pred_val)),
        "test_acc": float(accuracy_score(y_test, pred_test)),
    }


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    params = load_params(args.params)
    model_name = params["model"]
    scaler = bool(params["scaler"])
    hyperparams = params.get("hyperparams") or {}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    X, y, feature_names = load_data(args.data_path)
    X_tr, X_val, X_test, y_tr, y_val, y_test = split_data(
        X, y, args.seed, args.val_size, args.test_size
    )
    print(f"[train] 数据: n_train={len(y_tr)}, n_val={len(y_val)}, "
          f"n_test={len(y_test)}, features={len(feature_names)}")

    if scaler:
        scaler_obj = StandardScaler().fit(X_tr)
        X_tr = scaler_obj.transform(X_tr)
        X_val = scaler_obj.transform(X_val)
        X_test = scaler_obj.transform(X_test)

    model = build_model(model_name, hyperparams, seed=args.seed)
    started = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_seconds = round(time.perf_counter() - started, 4)
    scores = evaluate_scores(model, X_val, y_val, X_test, y_test)
    print(f"[train] 模型: {model_name}, scaler={scaler}, "
          f"训练耗时={train_seconds}s")
    print(f"[train] val_auc={scores['val_auc']:.4f}, "
          f"test_auc={scores['test_auc']:.4f}")

    metrics = {
        "model": model_name,
        "scaler": scaler,
        "hyperparams": hyperparams,
        "seed": args.seed,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train_seconds": train_seconds,
        **scores,
    }
    out.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[train] 指标已写入: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
