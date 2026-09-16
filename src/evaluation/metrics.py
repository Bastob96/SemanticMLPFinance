"""Binary direction metrics; AUC uses the UP score, never hard labels as fallback."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

METRICS = ("accuracy", "balanced_accuracy", "auc", "precision", "recall", "f1")


def _vector(values, name, binary=False):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional vector")
    if binary and not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return array


def calculate_classification_metrics(y_true, predicted, probability=None):
    """Return six metrics. Undefined AUC is NaN; undefined P/R/F1 is zero.

    Balanced accuracy averages recall over classes present in y_true, following
    sklearn. probability, when supplied, is a finite probability for class 1.
    Inputs are validated even when AUC is undefined for a single-class target.
    """
    actual = _vector(y_true, "actual", binary=True)
    predicted = _vector(predicted, "predicted", binary=True)
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted lengths differ")
    auc = np.nan
    if probability is not None:
        probability = _vector(probability, "probability_up")
        if len(probability) != len(actual) or ((probability < 0) | (probability > 1)).any():
            raise ValueError("probability_up must match labels and lie in [0, 1]")
        if np.unique(actual).size == 2:
            auc = roc_auc_score(actual, probability)
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "auc": float(auc),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
    }
