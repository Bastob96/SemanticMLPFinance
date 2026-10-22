"""Fixed Phase 1 HGBR pipeline. Only the feature columns change between experiments."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import config


def make_model():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=config.IMPUTER_STRATEGY)),
        ("regressor", HistGradientBoostingRegressor(**config.HGBR_PARAMS)),
    ])


def fit_model(features, target, fit_index, columns):
    """Fit on ``fit_index`` rows only; the imputer is fitted on the same rows."""
    model = make_model()
    model.fit(features.loc[fit_index, list(columns)], target.loc[fit_index])
    return model


def predict(model, features, index, columns):
    return np.asarray(model.predict(features.loc[index, list(columns)]), dtype=float)


def returns_to_prices(close_t, predicted_log_return):
    """Convert predicted next-day log returns to predicted next-day closes (for display)."""
    return np.asarray(close_t, dtype=float) * np.exp(np.asarray(predicted_log_return, dtype=float))
