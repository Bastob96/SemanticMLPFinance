"""Train Stage 1 models, save predictions, and create provisional results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import quarterly_split
from src.evaluation import calculate_classification_metrics
from src.features import FEATURE_SETS, build_stage1_features
from src.models import fit_logistic_regression, fit_random_forest

LR_SEED = 42
RF_SEEDS = (0, 1, 2)
TARGET = "target_next_day_up"


def prediction_frame(frame, split, experiment, model, seed, predicted, probability):
    return pd.DataFrame({
        "Date": frame["Date"].dt.strftime("%Y-%m-%d"),
        "target_date": frame["target_date"].dt.strftime("%Y-%m-%d"),
        "actual": frame[TARGET].to_numpy(),
        "predicted": predicted,
        "probability_up": probability,
        "split": split,
        "experiment": experiment,
        "model": model,
        "seed": seed,
    })


def run(data_path: Path, output_dir: Path, year: int = 2025):
    features = build_stage1_features(pd.read_csv(data_path))
    splits = quarterly_split(features, year)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(splits.values()).sort_values("Date").to_csv(output_dir / "stage1_features_2025.csv", index=False)

    rows, predictions = [], []
    y_train = splits["train"][TARGET]
    majority_class = int(y_train.mode().iloc[0])
    majority_probability = float(y_train.mean())

    for split_name in ("validation", "test"):
        frame = splits[split_name]
        actual = frame[TARGET]
        majority_pred = np.full(len(frame), majority_class)
        majority_prob = np.full(len(frame), majority_probability)
        rows.append({"split": split_name, "experiment": "baseline", "model": "majority_class", "seed": np.nan,
                     **calculate_classification_metrics(actual, majority_pred, majority_prob)})
        predictions.append(prediction_frame(frame, split_name, "baseline", "majority_class", np.nan, majority_pred, majority_prob))

        previous_pred = frame["previous_direction"].to_numpy()
        previous_prob = previous_pred.astype(float)
        rows.append({"split": split_name, "experiment": "baseline", "model": "previous_direction", "seed": np.nan,
                     **calculate_classification_metrics(actual, previous_pred, previous_prob)})
        predictions.append(prediction_frame(frame, split_name, "baseline", "previous_direction", np.nan, previous_pred, previous_prob))

    for experiment, columns in FEATURE_SETS.items():
        x_train = splits["train"][columns]
        logistic = fit_logistic_regression(x_train, y_train, LR_SEED)
        forests = [(seed, fit_random_forest(x_train, y_train, seed)) for seed in RF_SEEDS]

        for split_name in ("validation", "test"):
            frame = splits[split_name]
            actual = frame[TARGET]
            x_evaluate = frame[columns]

            probability = logistic.predict_proba(x_evaluate)[:, 1]
            predicted = (probability >= 0.5).astype(int)
            rows.append({"split": split_name, "experiment": experiment, "model": "logistic_regression", "seed": LR_SEED,
                         **calculate_classification_metrics(actual, predicted, probability)})
            predictions.append(prediction_frame(frame, split_name, experiment, "logistic_regression", LR_SEED, predicted, probability))

            for seed, forest in forests:
                probability = forest.predict_proba(x_evaluate)[:, 1]
                predicted = (probability >= 0.5).astype(int)
                rows.append({"split": split_name, "experiment": experiment, "model": "random_forest", "seed": seed,
                             **calculate_classification_metrics(actual, predicted, probability)})
                predictions.append(prediction_frame(frame, split_name, experiment, "random_forest", seed, predicted, probability))

    detailed = pd.DataFrame(rows)
    all_predictions = pd.concat(predictions, ignore_index=True)
    detailed.to_csv(output_dir / "stage1_metrics_provisional_detailed.csv", index=False)
    all_predictions.to_csv(output_dir / "stage1_predictions.csv", index=False)

    measures = ["accuracy", "balanced_accuracy", "auc", "precision", "recall", "f1"]
    summary = detailed.groupby(["split", "experiment", "model"], as_index=False)[measures].agg(["mean", "std"])
    summary.columns = ["_".join(filter(None, map(str, column))).rstrip("_") for column in summary.columns]
    summary.to_csv(output_dir / "stage1_results_summary.csv", index=False)

    run_info = {
        "year": year,
        "logistic_regression_seed": LR_SEED,
        "random_forest_seeds": list(RF_SEEDS),
        "threshold": 0.5,
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "note": "Metrics are provisional until passed through Abu's shared evaluation module."
    }
    (output_dir / "stage1_run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    return detailed, all_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/aligned_daily_ohlcv_5y.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()
    metrics, _ = run(args.data, args.output, args.year)
    print(metrics.to_string(index=False))
