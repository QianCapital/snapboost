"""Reproducible equal-split benchmark for SnapBoost and installed competitors.

Run with ``python benchmarks/benchmark.py``. XGBoost and LightGBM are included
when installed. Results are emitted as CSV so repeated runs can be aggregated
without notebook state.
"""

import argparse
import csv
import time

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from snapboost import SnapBoostClassifier, SnapBoostRegressor


def _competitors(task, seed):
    models = {}
    try:
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBClassifier if task == "classification" else XGBRegressor
        models["xgboost"] = cls(n_estimators=300, random_state=seed, n_jobs=1)
    except Exception as error:
        print(f"Skipping XGBoost: {error}")
        pass
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        cls = LGBMClassifier if task == "classification" else LGBMRegressor
        models["lightgbm"] = cls(n_estimators=300, random_state=seed, n_jobs=1)
    except Exception as error:
        print(f"Skipping LightGBM: {error}")
        pass
    return models


def run(seed):
    rows = []
    cases = [
        ("breast_cancer", "classification", load_breast_cancer(return_X_y=True)),
        ("diabetes", "regression", load_diabetes(return_X_y=True)),
    ]
    for dataset, task, (X, y) in cases:
        stratify = y if task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=stratify
        )
        X_fit, X_valid, y_fit, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=seed,
            stratify=y_train if task == "classification" else None,
        )
        if task == "classification":
            snap = SnapBoostClassifier(
                num_iterations=300, learning_rate=0.05,
                selection_strategy="greedy", line_search=True,
                subsample=0.8, max_features=0.8,
                early_stopping_rounds=25, random_state=seed,
            )
        else:
            snap = SnapBoostRegressor(
                num_iterations=300, learning_rate=0.05,
                selection_strategy="greedy", line_search=True,
                subsample=0.8, max_features=0.8,
                early_stopping_rounds=25, random_state=seed,
            )
        models = {"snapboost_adaptive": snap, **_competitors(task, seed)}
        for name, model in models.items():
            start = time.perf_counter()
            # All models train on the same rows. The hold-out split is only
            # used as SnapBoost's early-stopping eval set.
            if name == "snapboost_adaptive":
                model.fit(X_fit, y_fit, eval_set=(X_valid, y_valid))
            else:
                model.fit(X_fit, y_fit)
            train_seconds = time.perf_counter() - start
            if task == "classification":
                probability = model.predict_proba(X_test)[:, 1]
                metrics = {
                    "primary_metric": "log_loss",
                    "primary_value": log_loss(y_test, probability),
                    "secondary_metric": "roc_auc",
                    "secondary_value": roc_auc_score(y_test, probability),
                }
            else:
                prediction = model.predict(X_test)
                metrics = {
                    "primary_metric": "rmse",
                    "primary_value": float(
                        np.sqrt(mean_squared_error(y_test, prediction))
                    ),
                    "secondary_metric": "",
                    "secondary_value": np.nan,
                }
            rows.append({
                "seed": seed, "dataset": dataset, "task": task, "model": name,
                "train_seconds": train_seconds, **metrics,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 41])
    parser.add_argument("--output", default="benchmark_results.csv")
    args = parser.parse_args()
    rows = [row for seed in args.seeds for row in run(seed)]
    with open(args.output, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} results to {args.output}")


if __name__ == "__main__":
    main()
