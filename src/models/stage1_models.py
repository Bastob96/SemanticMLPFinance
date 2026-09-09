"""Model factories fitted only on the supplied training rows."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_logistic_regression(x_train, y_train, seed: int = 42) -> Pipeline:
    model = Pipeline(
        [
            ("standardise", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
        ]
    )
    return model.fit(x_train, y_train)


def fit_random_forest(x_train, y_train, seed: int) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    return model.fit(x_train, y_train)
