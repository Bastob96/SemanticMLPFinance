"""Metrics, the naive benchmark, feature-set evaluation and permutation importance."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config
from model_trainer import fit_model, predict


def regression_metrics(y_true, y_pred):
    """Primary: MAE. Secondary: RMSE, R2, directional accuracy, Spearman IC (all on log returns)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.allclose(y_pred, y_pred[0]):
        spearman = None   # undefined for a constant prediction (e.g. the naive benchmark)
    else:
        spearman = float(spearmanr(y_pred, y_true)[0])
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
        "spearman_ic": spearman,
    }


def naive_predictions(n):
    """Naive benchmark: tomorrow's close equals today's close, i.e. a predicted log return of 0."""
    return np.zeros(n, dtype=float)


def price_metrics(close_t, y_true, y_pred):
    """Supplementary USD errors on the implied next-day close."""
    close_t = np.asarray(close_t, dtype=float)
    actual = close_t * np.exp(np.asarray(y_true, dtype=float))
    predicted = close_t * np.exp(np.asarray(y_pred, dtype=float))
    return {"price_mae_usd": float(np.mean(np.abs(actual - predicted))),
            "price_rmse_usd": float(np.sqrt(np.mean((actual - predicted) ** 2)))}


class FeatureSetEvaluator:
    """Train on the training rows, evaluate on validation; results cached by column set.

    HGBR without early stopping and with a fixed random_state is deterministic,
    so each feature set needs to be fitted only once per run.
    """

    def __init__(self, features, target, train_index, validation_index):
        self.features = features
        self.target = target
        self.train_index = train_index
        self.validation_index = validation_index
        self._cache = {}
        self.fit_count = 0

    def add_feature(self, name, values):
        if name in self.features.columns:
            raise ValueError(f"Feature '{name}' already exists")
        self.features[name] = values.reindex(self.features.index)

    def evaluate(self, columns):
        key = tuple(columns)
        if key not in self._cache:
            model = fit_model(self.features, self.target, self.train_index, key)
            prediction = predict(model, self.features, self.validation_index, key)
            self.fit_count += 1
            self._cache[key] = {"columns": list(key), "model": model, "predictions": prediction,
                                "metrics": regression_metrics(self.target.loc[self.validation_index], prediction)}
        return self._cache[key]


def compute_permutation_importance(model, features, target, index, columns):
    """Increase in MAE (log return) when one factor is shuffled; evaluated on ``index`` only."""
    result = permutation_importance(model, features.loc[index, list(columns)], target.loc[index],
                                    scoring="neg_mean_absolute_error",
                                    n_repeats=config.PERMUTATION_REPEATS,
                                    random_state=config.RANDOM_SEED)
    return pd.DataFrame({"factor": list(columns), "importance_mean": result.importances_mean,
                         "importance_std": result.importances_std}).sort_values(
        "importance_mean", ascending=False).reset_index(drop=True)


def relative_improvement(reference_mae, candidate_mae):
    """Positive when the candidate MAE is lower; e.g. 0.01 means 1 % lower."""
    return (reference_mae - candidate_mae) / reference_mae
