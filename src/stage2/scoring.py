"""Score candidate factors against the 22 Stage 1 factors (validation quarters only).

For every year the same quarterly split as Stage 1 is used (Q1-Q2 train,
Q3 validation, Q4 test). A candidate is added to feature set C and the
Stage 1 models are refitted on the training quarters. Only the validation
quarter is scored, so the Q4 test quarters stay untouched for the final test.

Reported per candidate and year:
  auc_gain      validation ROC AUC with the candidate minus without it
                (mean of Logistic Regression and the 3 Random Forest seeds)
  bal_acc_gain  same for balanced accuracy
  importance    permutation importance of the candidate (drop in validation AUC)
  helped        auc_gain > 0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.data import quarterly_split
from src.features import FEATURE_SETS
from src.models import fit_logistic_regression, fit_random_forest

TARGET = "target_next_day_up"
BASE = FEATURE_SETS["C"]
YEARS = (2022, 2023, 2024, 2025)
LR_SEED, RF_SEEDS = 42, (0, 1, 2)


def _models(x, y):
    yield "logistic_regression", fit_logistic_regression(x, y, LR_SEED)
    for seed in RF_SEEDS:
        yield f"random_forest_{seed}", fit_random_forest(x, y, seed)


def _fit_score(train, val, columns, importance_of=None, repeats=20):
    rows = []
    for name, model in _models(train[columns], train[TARGET]):
        proba = model.predict_proba(val[columns])[:, 1]
        row = {"model": name,
               "auc": roc_auc_score(val[TARGET], proba),
               "bal_acc": balanced_accuracy_score(val[TARGET], (proba >= 0.5).astype(int))}
        if importance_of is not None:
            # permutation importance of this one column only (same idea as sklearn's, much faster)
            rng = np.random.default_rng(0)
            x = val[columns].copy()
            drops = []
            for _ in range(repeats):
                x[importance_of] = rng.permutation(val[importance_of].to_numpy())
                drops.append(row["auc"] - roc_auc_score(val[TARGET], model.predict_proba(x)[:, 1]))
            row["importance"] = float(np.mean(drops))
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class Scorer:
    """Caches the base-model scores so each new candidate costs one refit per year."""
    features: pd.DataFrame          # Stage 1 feature table (build_stage1_features output)
    years: tuple = YEARS

    def __post_init__(self):
        self.features = self.features.copy()
        self.features["Date"] = pd.to_datetime(self.features["Date"])
        self._base = {}          # (year, team names) -> scores of the models without the new factor(s)
        for year in self.years:
            s = quarterly_split(self.features, year)
            self._base[(year, ())] = _fit_score(s["train"], s["validation"], BASE).set_index("model")

    def score(self, name: str, values: pd.Series) -> pd.DataFrame:
        """values: factor indexed by Date. Returns one row per year (gain over the 22 factors)."""
        return self.gains({name: values}, team={}, importance_of=name)

    def gains(self, add: dict, team: dict, importance_of=None) -> pd.DataFrame:
        """Gain from adding the factor(s) in `add` on top of the 22 factors plus `team`.

        add, team: {name: values indexed by Date}. With an empty team this is the solo test;
        with the selected team it is the team trial; with two factors in `add` it is a pair trial.
        Returns one row per year: auc_gain (mean of LR and the 3 RF seeds), auc_gain_lr,
        auc_gain_rf, bal_acc_gain and, if requested, the permutation importance of one factor.
        """
        add_names, team_names = list(add), sorted(team)
        label = "+".join(add_names)
        data = self.features
        for name, values in {**team, **add}.items():
            values = pd.Series(values, name=name)
            values.index = pd.to_datetime(values.index)
            data = data.merge(values.rename(name), left_on="Date", right_index=True, how="left")
        extra = team_names + add_names
        rows = []
        for year in self.years:
            s = quarterly_split(data, year)
            train, val = s["train"], s["validation"]
            bad = [c for c in extra if train[c].isna().any() or val[c].isna().any()
                   or not np.isfinite(train[c]).all() or train[c].nunique() < 3]
            if bad:
                rows.append({"factor": label, "year": year, "valid": False})
                continue
            # tame extreme values using the training quarters' 1st/99th percentiles (no peeking at validation)
            train, val = train.copy(), val.copy()
            for c in extra:
                lo, hi = train[c].quantile([0.01, 0.99])
                train[c], val[c] = train[c].clip(lo, hi), val[c].clip(lo, hi)
            key = (year, tuple(team_names))
            if key not in self._base:
                self._base[key] = _fit_score(train, val, BASE + team_names).set_index("model")
            base = self._base[key]
            cand = _fit_score(train, val, BASE + team_names + add_names, importance_of=importance_of).set_index("model")
            gain = cand[["auc", "bal_acc"]] - base[["auc", "bal_acc"]]
            row = {"factor": label, "year": year, "valid": True,
                   "auc_gain": gain["auc"].mean(),
                   "auc_gain_lr": gain.loc["logistic_regression", "auc"],
                   "auc_gain_rf": gain.drop("logistic_regression")["auc"].mean(),
                   "bal_acc_gain": gain["bal_acc"].mean()}
            row["importance"] = cand["importance"].mean() if importance_of is not None else float("nan")
            rows.append(row)
        return pd.DataFrame(rows)


def summarise(per_year: pd.DataFrame) -> pd.DataFrame:
    """One row per factor: averages over years and how many years it helped."""
    ok = per_year[per_year["valid"].astype(bool)]
    out = ok.groupby("factor").agg(
        auc_gain=("auc_gain", "mean"),
        auc_gain_lr=("auc_gain_lr", "mean"),
        auc_gain_rf=("auc_gain_rf", "mean"),
        bal_acc_gain=("bal_acc_gain", "mean"),
        importance=("importance", "mean"),
        years_helped=("auc_gain", lambda x: int((x > 0).sum())),
        years_scored=("auc_gain", "size"),
    )
    return out.sort_values(["years_helped", "auc_gain"], ascending=False)
