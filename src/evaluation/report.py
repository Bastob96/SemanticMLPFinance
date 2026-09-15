"""Validate and evaluate saved Stage 1 predictions without retraining."""

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import calculate_classification_metrics

KEYS = ["split", "experiment", "model", "seed"]
COLUMNS = ["Date", "target_date", "actual", "predicted", "probability_up", *KEYS]


def load_predictions(source):
    """Read a CSV or copy a DataFrame; reject ambiguous runs and misaligned rows.

    A partial set of configurations is supported. Within each split all supplied
    configurations must have exactly the same dated targets and actual labels.
    """
    frame = pd.read_csv(source) if isinstance(source, (str, Path)) else source.copy()
    missing = set(COLUMNS) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Empty predictions or missing columns: {sorted(missing)}")
    for col in ["Date", "target_date"]:
        frame[col] = pd.to_datetime(frame[col], errors="raise")
        if frame[col].isna().any():
            raise ValueError(f"Missing {col}")
    if not (frame.target_date > frame.Date).all():
        raise ValueError("target_date must follow Date")
    if not frame.split.isin(["validation", "test"]).all():
        raise ValueError("Only validation and test predictions are supported")
    if frame[KEYS[:-1]].isna().any().any():
        raise ValueError("Missing run identifiers")
    if frame.duplicated(KEYS + ["Date"]).any():
        raise ValueError("Duplicate prediction dates within a run")
    baseline = frame.model.isin(["majority_class", "previous_direction"])
    lr = frame.model.eq("logistic_regression")
    rf = frame.model.eq("random_forest")
    if not (baseline | lr | rf).all():
        raise ValueError("Unknown model")
    if not frame.loc[baseline, "experiment"].eq("baseline").all() or not frame.loc[~baseline, "experiment"].isin(["A", "B", "C"]).all():
        raise ValueError("Invalid model/experiment combination")
    if frame.loc[baseline, "seed"].notna().any() or not frame.loc[lr, "seed"].eq(42).all() or not frame.loc[rf, "seed"].isin([0, 1, 2]).all():
        raise ValueError("Expected no baseline seed, LR seed 42, RF seeds 0/1/2")
    for _, group in frame.groupby(KEYS, dropna=False):
        calculate_classification_metrics(group.actual, group.predicted, group.probability_up)
    for _, split in frame.groupby("split"):
        reference = None
        for _, group in split.groupby(KEYS, dropna=False):
            dated = group.sort_values("Date")[["Date", "target_date", "actual"]].reset_index(drop=True)
            if reference is not None and not dated.equals(reference):
                raise ValueError("Configurations must share dates, targets, and actual labels")
            reference = dated
    if set(frame.split) == {"validation", "test"}:
        if frame.loc[frame.split.eq("validation"), "target_date"].max() >= frame.loc[frame.split.eq("test"), "Date"].min():
            raise ValueError("Validation and test windows overlap")
    return frame.sort_values(KEYS + ["Date"]).reset_index(drop=True)


def evaluate_predictions(source):
    """Return one row per split/experiment/model/seed, including observation count."""
    frame = load_predictions(source)
    return pd.DataFrame([
        {**dict(zip(KEYS, key)), "n_observations": len(group),
         **calculate_classification_metrics(group.actual, group.predicted, group.probability_up)}
        for key, group in frame.groupby(KEYS, dropna=False, sort=True)
    ])


def comparison_table(source):
    """Return six metric means and sample SDs; baselines appear once per split."""
    from .aggregate import aggregate_runs
    return aggregate_runs(evaluate_predictions(source))


def select_best_configuration(comparison):
    """Select using validation balanced accuracy only, including baselines.

    Ties resolve by experiment then model (lexical). RF is ranked by seed mean;
    seed 0 is fixed in advance for simulation, never chosen by test performance.
    """
    candidates = comparison.loc[comparison.split.eq("validation")].copy()
    if candidates.empty or not np.isfinite(candidates.balanced_accuracy_mean).all():
        raise ValueError("Finite validation balanced accuracy is required for selection")
    best = candidates.sort_values(
        ["balanced_accuracy_mean", "experiment", "model"], ascending=[False, True, True]
    ).iloc[0]
    return {"experiment": best.experiment, "model": best.model,
            "seed": 0 if best.model == "random_forest" else (42 if best.model == "logistic_regression" else None),
            "selection_split": "validation", "selection_metric": "balanced_accuracy_mean",
            "selection_score": float(best.balanced_accuracy_mean)}
