"""Facial Keypoints 提交管线：全量训练 + 官方 test.csv/IdLookupTable 预测。

用法：
    python experiments/tasks/facial_keypoints/submit.py \
        --params experiments/tasks/facial_keypoints/baseline.yaml \
        --out submissions/facial_keypoints_submission.csv

输出格式：RowId,Location（行数与 IdLookupTable 一致）。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import configure_utf8_stdio, write_metrics  # noqa: E402
from experiments.tasks.facial_keypoints import train as t  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "submissions" / "facial_keypoints_submission.csv"


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=str, default="")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--metrics-out", type=str, default="")
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--test-path", type=str, default="")
    parser.add_argument("--lookup-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    params = t.load_params(args.params, t.DEFAULT_PARAMS)
    feature_ops = t.validate_feature_ops(params.get("feature_ops") or [])
    images, labels = t.load_data(args.data_path, max_rows=None)
    n_coords = labels.shape[1]

    test_path = Path(args.test_path or PROJECT_ROOT / "data" / "raw" / "facial_keypoints" / "test.csv")
    lookup_path = Path(args.lookup_path or PROJECT_ROOT / "data" / "raw" / "facial_keypoints" / "IdLookupTable.csv")
    test_df = pd.read_csv(test_path)
    lookup = pd.read_csv(lookup_path)
    test_images = t.parse_images(test_df["Image"])
    if len(test_images) != lookup["ImageId"].max():
        raise ValueError("test 图像数小于 IdLookupTable 的最大 ImageId")
    if list(test_df["ImageId"]) != list(range(1, len(test_df) + 1)):
        raise ValueError("test.csv 的 ImageId 必须为 1..N 顺序")

    X_train = t.apply_pixel_ops(
        images.astype(np.float32) / 255.0, feature_ops
    )
    X_test = t.apply_pixel_ops(
        test_images.astype(np.float32) / 255.0, feature_ops
    )
    pca = PCA(
        n_components=int(params.get("pca_components", 64)),
        svd_solver="randomized",
        random_state=args.seed,
    )
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"[submit] facial_keypoints: PCA {pca.components_.shape[0]} "
          f"完成（train {len(X_train)} -> test {len(X_test)}）")

    scaler = bool(params.get("scaler", True))
    pred_test = np.empty((len(X_test), n_coords), dtype=np.float32)
    for coord in range(n_coords):
        y_col = labels[:, coord]
        obs = np.flatnonzero(~np.isnan(y_col))
        if len(obs) == 0:
            raise ValueError(f"坐标列 {t.LABEL_COLS[coord]} 在 train 中无标签")
        est = t.build_estimator(params["model"], params["hyperparams"], args.seed)
        steps = []
        if scaler:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", est))
        pipe = Pipeline(steps).fit(X_train_pca[obs], y_col[obs])
        pred_test[:, coord] = pipe.predict(X_test_pca)

    col_index = {name: i for i, name in enumerate(t.LABEL_COLS)}
    locations = np.empty(len(lookup), dtype=np.float32)
    missing = []
    for i, (image_id, feature) in enumerate(
        zip(lookup["ImageId"].to_numpy(), lookup["FeatureName"].to_numpy())
    ):
        col = col_index.get(feature)
        if col is None:
            missing.append(feature)
            continue
        locations[i] = pred_test[image_id - 1, col]
    if missing:
        raise ValueError(f"IdLookupTable 含未知特征: {sorted(set(missing))}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"RowId": lookup["RowId"], "Location": locations}).to_csv(
        out, index=False
    )
    print(f"[submit] facial_keypoints: 已写入 {out}（{len(locations)} 行）")

    metrics = {
        "task": "facial_keypoints",
        "mode": "submission",
        "model": params["model"],
        "scaler": scaler,
        "pca_components": int(pca.components_.shape[0]),
        "hyperparams": params["hyperparams"],
        "feature_ops": feature_ops,
        "seed": args.seed,
        "n_train": int(len(images)),
        "n_test": int(len(test_images)),
        "n_lookup": int(len(lookup)),
        "submission": str(out),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_metrics(args.metrics_out or str(out.with_suffix(".metrics.json")), metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
