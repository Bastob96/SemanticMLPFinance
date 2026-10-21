"""Integrate quarterly LLM factors with Stage 1 features without data leakage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import quarterly_split
from src.evaluation import METRICS, select_best_configuration
from src.features import FEATURE_SETS as STAGE1_FEATURE_SETS, build_stage1_features
from src.models import fit_logistic_regression, fit_random_forest


TARGET = "target_next_day_up"
LR_SEED = 42
RF_SEEDS = (0, 1, 2)
PRIMARY_EXPERIMENTS = (
    "A_stage1",
    "B_stage1",
    "B_stage2",
    "C_stage1",
    "C_stage2",
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"Input file not found: {path}")
    frame = pd.read_csv(path)
    for column in ("Date", "target_date"):
        if column not in frame:
            raise ValueError(f"{path.name} is missing {column}")
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    if frame["Date"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate dates")
    return frame.sort_values("Date").reset_index(drop=True)


def _validate_numeric(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing feature columns: {missing}")
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-finite feature values")


def load_factor_feature_sets(path: Path) -> dict[str, list[str]]:
    """Load explicit candidate columns instead of selecting numeric columns."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != {"A", "B", "C"}:
        raise ValueError("Stage 2 feature config must contain exactly A, B and C")
    result: dict[str, list[str]] = {}
    for name, columns in config.items():
        if not isinstance(columns, list) or any(not isinstance(c, str) for c in columns):
            raise ValueError(f"Stage 2 feature set {name} must be a list of names")
        if len(columns) != len(set(columns)):
            raise ValueError(f"Stage 2 feature set {name} contains duplicates")
        result[name] = columns
    if result["A"] or len(result["B"]) != 4 or len(result["C"]) != 8:
        raise ValueError("Expected 0 A factors, 4 B factors and 8 C factors")
    if not set(result["B"]).issubset(result["C"]):
        raise ValueError("Stage 2 B factors must be a subset of C")
    return result


def build_experiment_specs(stage2_sets: dict[str, list[str]]) -> dict[str, list[str]]:
    """Define fair controls: candidate experiments only add LLM factors."""
    specs = {
        "A_stage1": list(STAGE1_FEATURE_SETS["A"]),
        "B_stage1": list(STAGE1_FEATURE_SETS["B"]),
        "B_stage2": list(STAGE1_FEATURE_SETS["B"]) + list(stage2_sets["B"]),
        "C_stage1": list(STAGE1_FEATURE_SETS["C"]),
        "C_stage2": list(STAGE1_FEATURE_SETS["C"]) + list(stage2_sets["C"]),
    }
    for name, columns in specs.items():
        if len(columns) != len(set(columns)):
            raise ValueError(f"Experiment {name} contains duplicate features")
    return specs


def load_integrated_splits(
    canonical_ohlcv_path: Path,
    stage2_quarterly_dir: Path,
    feature_sets_path: Path,
    include_test: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    """Build Stage 1 features and join the matching quarterly factor rows."""
    raw = pd.read_csv(canonical_ohlcv_path)
    stage1_splits = quarterly_split(build_stage1_features(raw), 2025)
    stage2_sets = load_factor_feature_sets(feature_sets_path)
    factors = sorted({column for values in stage2_sets.values() for column in values})
    requested = ["train", "validation"] + (["test"] if include_test else [])
    integrated: dict[str, pd.DataFrame] = {}

    for split_name in requested:
        baseline = stage1_splits[split_name]
        path = stage2_quarterly_dir / f"stage2_{split_name}_2025.csv"
        candidate = _read_csv(path)
        if "Split" not in candidate or not candidate["Split"].eq(split_name).all():
            raise ValueError(f"{path.name} has incorrect Split values")
        _validate_numeric(candidate, factors + [TARGET], path.name)
        merged = baseline.merge(
            candidate[["Date", "target_date", TARGET, *factors]],
            on="Date",
            how="inner",
            validate="one_to_one",
            suffixes=("_stage1", "_stage2"),
        )
        if len(merged) != len(baseline) or len(merged) != len(candidate):
            raise ValueError(f"Stage 1 and Stage 2 dates do not align for {split_name}")
        if not merged["target_date_stage1"].eq(merged["target_date_stage2"]).all():
            raise ValueError(f"target_date mismatch for {split_name}")
        if not merged[f"{TARGET}_stage1"].eq(merged[f"{TARGET}_stage2"]).all():
            raise ValueError(f"Target label mismatch for {split_name}")
        merged = merged.drop(columns=["target_date_stage2", f"{TARGET}_stage2"])
        merged = merged.rename(columns={
            "target_date_stage1": "target_date",
            f"{TARGET}_stage1": TARGET,
        })
        integrated[split_name] = merged.sort_values("Date").reset_index(drop=True)

    expected = {"train": 121, "validation": 63, "test": 63}
    for split_name, frame in integrated.items():
        if len(frame) != expected[split_name]:
            raise ValueError(f"Expected {expected[split_name]} {split_name} rows")
    if integrated["train"].target_date.max() >= integrated["validation"].Date.min():
        raise ValueError("Training and validation target windows overlap")
    if include_test and integrated["validation"].target_date.max() >= integrated["test"].Date.min():
        raise ValueError("Validation and test target windows overlap")
    return integrated, stage2_sets


def _prediction_frame(frame, split, experiment, model, seed, predicted, probability):
    """Use the exact schema consumed by the shared evaluator and backtester."""
    return pd.DataFrame({
        "Date": frame["Date"].dt.strftime("%Y-%m-%d"),
        "target_date": frame["target_date"].dt.strftime("%Y-%m-%d"),
        "actual": frame[TARGET].to_numpy(dtype=int),
        "predicted": np.asarray(predicted, dtype=int),
        "probability_up": np.asarray(probability, dtype=float),
        "split": split,
        "experiment": experiment,
        "model": model,
        "seed": seed,
    })


def build_predictions(
    splits: dict[str, pd.DataFrame],
    specs: dict[str, list[str]],
    evaluation_splits: tuple[str, ...],
) -> pd.DataFrame:
    """Fit on Q1-Q2 only and predict the requested later split(s)."""
    if "train" not in splits or any(name not in splits for name in evaluation_splits):
        raise ValueError("Requested split is unavailable")
    train = splits["train"]
    y_train = train[TARGET].astype(int)
    majority_class = int(y_train.mode().iloc[0])
    majority_probability = float(y_train.mean())
    predictions = []

    for split_name in evaluation_splits:
        frame = splits[split_name]
        majority_pred = np.full(len(frame), majority_class)
        predictions.append(_prediction_frame(
            frame, split_name, "baseline", "majority_class", np.nan,
            majority_pred, np.full(len(frame), majority_probability),
        ))
        previous_pred = frame["previous_direction"].to_numpy(dtype=int)
        predictions.append(_prediction_frame(
            frame, split_name, "baseline", "previous_direction", np.nan,
            previous_pred, previous_pred.astype(float),
        ))

    for experiment, columns in specs.items():
        _validate_numeric(train, columns, f"training data for {experiment}")
        logistic = fit_logistic_regression(train[columns], y_train, LR_SEED)
        forests = [
            (seed, fit_random_forest(train[columns], y_train, seed))
            for seed in RF_SEEDS
        ]
        for split_name in evaluation_splits:
            frame = splits[split_name]
            x_evaluate = frame[columns]
            probability = logistic.predict_proba(x_evaluate)[:, 1]
            predictions.append(_prediction_frame(
                frame, split_name, experiment, "logistic_regression", LR_SEED,
                probability >= 0.5, probability,
            ))
            for seed, forest in forests:
                probability = forest.predict_proba(x_evaluate)[:, 1]
                predictions.append(_prediction_frame(
                    frame, split_name, experiment, "random_forest", seed,
                    probability >= 0.5, probability,
                ))
    return pd.concat(predictions, ignore_index=True)


def select_primary_configuration(comparison: pd.DataFrame) -> dict:
    """Select by Q3 balanced accuracy; Q4 scores are never consulted."""
    eligible = comparison.loc[
        comparison["experiment"].isin(["baseline", *PRIMARY_EXPERIMENTS])
    ].copy()
    selection = select_best_configuration(eligible)
    selection["eligible_experiments"] = ["baseline", *PRIMARY_EXPERIMENTS]
    return selection


def incremental_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Measure the effect of adding factors while keeping each model fixed."""
    rows = []
    for split in sorted(comparison["split"].unique()):
        for ablation in ("B", "C"):
            for model in ("logistic_regression", "random_forest"):
                control = comparison.loc[
                    comparison.split.eq(split)
                    & comparison.experiment.eq(f"{ablation}_stage1")
                    & comparison.model.eq(model)
                ]
                candidate = comparison.loc[
                    comparison.split.eq(split)
                    & comparison.experiment.eq(f"{ablation}_stage2")
                    & comparison.model.eq(model)
                ]
                if len(control) != 1 or len(candidate) != 1:
                    raise ValueError(f"Missing comparison for {split}/{ablation}/{model}")
                row = {"split": split, "ablation": ablation, "model": model}
                for metric in METRICS:
                    before = float(control.iloc[0][f"{metric}_mean"])
                    after = float(candidate.iloc[0][f"{metric}_mean"])
                    row[f"stage1_{metric}"] = before
                    row[f"stage2_{metric}"] = after
                    row[f"delta_{metric}"] = after - before
                rows.append(row)
    return pd.DataFrame(rows)


def build_feature_explanation(
    train: pd.DataFrame,
    columns: list[str],
    stage2_factors: list[str],
) -> pd.DataFrame:
    """Return train-only standardized LR coefficients and RF importances."""
    y_train = train[TARGET].astype(int)
    logistic = fit_logistic_regression(train[columns], y_train, LR_SEED)
    coefficients = logistic.named_steps["classifier"].coef_[0]
    forests = [fit_random_forest(train[columns], y_train, seed) for seed in RF_SEEDS]
    importances = np.vstack([model.feature_importances_ for model in forests])
    result = pd.DataFrame({
        "feature": columns,
        "source": ["llm_candidate" if c in stage2_factors else "conventional" for c in columns],
        "logistic_standardized_coefficient": coefficients,
        "logistic_abs_coefficient": np.abs(coefficients),
        "rf_importance_mean": importances.mean(axis=0),
        "rf_importance_std": importances.std(axis=0, ddof=1),
    })
    return result.sort_values(["source", "rf_importance_mean"], ascending=[False, False]).reset_index(drop=True)
