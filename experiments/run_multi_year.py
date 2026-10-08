"""Repeat the Stage 1 pipeline on several years and collect comparable results.

For each year the same quarterly split is used (Q1-Q2 train, Q3 validation,
Q4 test), the same models and seeds are trained, predictions are scored with
src.evaluation, and the validation-selected configuration is backtested on Q4
with src.backtesting. Optionally, permutation importance is computed for all
22 features (feature set C) on each year's validation quarter.

Usage:
    python experiments/run_multi_year.py --output results/multi_year
    python experiments/run_multi_year.py --output results/multi_year --importance
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_stage1_training import LR_SEED, RF_SEEDS, TARGET, run
from src.backtesting import backtest_best
from src.data import quarterly_split
from src.evaluation import comparison_table
from src.features import FEATURE_SETS, build_stage1_features
from src.models import fit_logistic_regression, fit_random_forest

DEFAULT_DATA = ROOT / "data/processed/aligned_daily_ohlcv_5y.csv"
DEFAULT_YEARS = (2022, 2023, 2024, 2025)  # full calendar years in the 5-year file


def run_years(data_path: Path, output_dir: Path, years=DEFAULT_YEARS):
    """Train, evaluate and backtest every year; return (comparison, backtest summary, daily, selections)."""
    comparisons, summaries, dailies, selections = [], [], [], []
    for year in years:
        _, predictions = run(data_path, output_dir / str(year), year)
        comparison = comparison_table(predictions)
        selection, daily, summary = backtest_best(predictions, data_path)
        comparisons.append(comparison.assign(year=year))
        summaries.append(summary.assign(year=year))
        dailies.append(daily.assign(year=year))
        selections.append({"year": year, **selection})
    comparison = pd.concat(comparisons, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    daily = pd.concat(dailies, ignore_index=True)
    comparison.to_csv(output_dir / "comparison_all_years.csv", index=False)
    summary.to_csv(output_dir / "backtest_summary_all_years.csv", index=False)
    daily.to_csv(output_dir / "backtest_daily_all_years.csv", index=False)
    (output_dir / "backtest_selection_all_years.json").write_text(json.dumps(selections, indent=2) + "\n")
    return comparison, summary, daily, selections


def permutation_importance_by_year(data_path: Path, years=DEFAULT_YEARS, n_repeats: int = 30):
    """Drop in validation ROC AUC when one feature is shuffled (feature set C).

    Models are fitted on each year's training quarters only and scored on that
    year's validation quarter; the test quarter is never used. Random Forest
    importance is averaged over seeds 0, 1 and 2.
    """
    from sklearn.inspection import permutation_importance

    features = build_stage1_features(pd.read_csv(data_path))
    columns = FEATURE_SETS["C"]
    rows = []
    for year in years:
        splits = quarterly_split(features, year)
        x_train, y_train = splits["train"][columns], splits["train"][TARGET]
        x_val, y_val = splits["validation"][columns], splits["validation"][TARGET]
        models = [("logistic_regression", LR_SEED, fit_logistic_regression(x_train, y_train, LR_SEED))]
        models += [("random_forest", seed, fit_random_forest(x_train, y_train, seed)) for seed in RF_SEEDS]
        for name, seed, model in models:
            result = permutation_importance(model, x_val, y_val, scoring="roc_auc",
                                            n_repeats=n_repeats, random_state=0, n_jobs=-1)
            rows += [{"year": year, "model": name, "seed": seed, "factor": factor, "importance": value}
                     for factor, value in zip(columns, result.importances_mean)]
    detailed = pd.DataFrame(rows)
    per_year = detailed.groupby(["model", "year", "factor"], as_index=False).importance.mean()
    summary = (per_year.groupby(["model", "factor"]).importance
               .agg(mean_importance="mean", positive_years=lambda x: int((x > 0).sum()))
               .reset_index()
               .sort_values(["model", "mean_importance"], ascending=[True, False]))
    return detailed, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "results/multi_year")
    parser.add_argument("--importance", action="store_true", help="also compute permutation importance (slower)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    args.output.mkdir(parents=True, exist_ok=True)
    comparison, summary, _, selections = run_years(args.data, args.output)
    test = comparison[comparison.split.eq("test")]
    print(test.pivot_table(index=["experiment", "model"], columns="year", values="balanced_accuracy_mean").round(3))
    print(summary.round(4).to_string(index=False))
    if args.importance:
        detailed, importance = permutation_importance_by_year(args.data)
        detailed.to_csv(args.output / "permutation_importance_detailed.csv", index=False)
        importance.to_csv(args.output / "permutation_importance_summary.csv", index=False)
        print(importance.round(4).to_string(index=False))
