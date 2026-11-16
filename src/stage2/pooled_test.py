"""Pooled test: all four sealed Q4 quarters together (agreed with Liu, 12 Nov).

Each Q4 quarter has only about 63 trading days, so one quarter's AUC can move by roughly 0.07 by chance alone.
Here the Q4 predictions of 2022, 2023, 2024 and 2025 are put together (about 250 days) and three questions
are answered for a selected factor set:

  1. How big is the gain?   Pooled AUC and balanced accuracy with the 22 factors, and with the 22 plus the
                            selected factors (each year trained on its own Q1-Q2, exactly as in final_test.py).
  2. How sure are we?       A 95% bootstrap range for the gain: the ~250 days are resampled with replacement
                            many times, the same days for both factor sets.
  3. Is it luck?            The same number of factors made of pure random numbers is added instead, many
                            times. p = the share of random sets whose pooled gain is at least as large.

Nothing here selects anything: the factor sets were chosen on Q3 before, and Q4 is only measured.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.data import quarterly_split
from src.models import fit_logistic_regression, fit_random_forest

from .formula import evaluate
from .scoring import BASE, LR_SEED, RF_MAX_FEATURES, RF_SEEDS, TARGET, YEARS

MODELS = ("logistic_regression", "random_forest")


def factor_values(formula: str, prices: pd.DataFrame) -> pd.Series:
    """Daily values of a kept factor. noise(<seed>) = the same random numbers the batch noise check used."""
    m = re.fullmatch(r"noise\((\d+)\)", formula.replace(" ", ""))
    if m:
        return pd.Series(np.random.default_rng(int(m.group(1))).standard_normal(len(prices)), index=prices.index)
    return evaluate(formula, prices)


def q4_predictions(features: pd.DataFrame, extra: dict, years=YEARS) -> pd.DataFrame:
    """Q4 probabilities for every year, with the 22 factors plus the factors in `extra` ({name: values by date}).
    The same settings as final_test.py: extra factors clipped to the training 1st/99th percentiles,
    LR seed 42, RF seeds 0-2 averaged with 4 inputs per split."""
    data = features.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    for name, values in extra.items():
        v = pd.Series(values, name=name)
        v.index = pd.to_datetime(v.index)
        data = data.merge(v.rename(name), left_on="Date", right_index=True, how="left")
    cols, names = BASE + list(extra), list(extra)
    rows = []
    for year in years:
        s = quarterly_split(data, year)
        train, test = s["train"].copy(), s["test"].copy()
        for c in names:
            lo, hi = train[c].quantile([0.01, 0.99])
            train[c], test[c] = train[c].clip(lo, hi), test[c].clip(lo, hi)
        probs = {
            "logistic_regression": fit_logistic_regression(train[cols], train[TARGET], LR_SEED)
            .predict_proba(test[cols])[:, 1],
            "random_forest": np.mean([fit_random_forest(train[cols], train[TARGET], seed, max_features=RF_MAX_FEATURES)
                                      .predict_proba(test[cols])[:, 1] for seed in RF_SEEDS], axis=0),
        }
        for model, p in probs.items():
            rows.append(pd.DataFrame({"Date": test["Date"].to_numpy(), "year": year, "model": model,
                                      "actual": test[TARGET].to_numpy(), "prob": p}))
    return pd.concat(rows, ignore_index=True)


def _scores(actual, prob):
    return roc_auc_score(actual, prob), balanced_accuracy_score(actual, (prob >= 0.5).astype(int))


def pooled_gain(base: pd.DataFrame, plus: pd.DataFrame) -> dict:
    """Pooled AUC and balanced accuracy, 22 factors vs 22 + extra, per model, on the same Q4 days."""
    out = {}
    for model in MODELS:
        b = base[base.model == model].set_index("Date")
        p = plus[plus.model == model].set_index("Date").reindex(b.index)
        auc_b, bal_b = _scores(b.actual, b.prob)
        auc_p, bal_p = _scores(b.actual, p.prob)
        out[model] = {"days": len(b), "auc_22": auc_b, "auc_plus": auc_p, "auc_gain": auc_p - auc_b,
                      "bal_22": bal_b, "bal_plus": bal_p, "bal_gain": bal_p - bal_b}
    return out


def bootstrap(base: pd.DataFrame, plus: pd.DataFrame, n: int = 2000, seed: int = 0) -> dict:
    """95% range of the pooled gains: resample the Q4 days with replacement (the same days for both sets)."""
    rng = np.random.default_rng(seed)
    out = {}
    for model in MODELS:
        b = base[base.model == model].set_index("Date")
        p = plus[plus.model == model].set_index("Date").reindex(b.index)
        y, pb, pp = b.actual.to_numpy(), b.prob.to_numpy(), p.prob.to_numpy()
        auc, bal = [], []
        for _ in range(n):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() == y[i].max():
                continue
            ab, bb = _scores(y[i], pb[i])
            ap, bp = _scores(y[i], pp[i])
            auc.append(ap - ab)
            bal.append(bp - bb)
        out[model] = {"auc_low": float(np.percentile(auc, 2.5)), "auc_high": float(np.percentile(auc, 97.5)),
                      "bal_low": float(np.percentile(bal, 2.5)), "bal_high": float(np.percentile(bal, 97.5))}
    return out


def noise_gains(features: pd.DataFrame, prices: pd.DataFrame, base: pd.DataFrame, k: int,
                draws: int = 100, seed: int = 10_000, n_jobs: int = 1) -> pd.DataFrame:
    """Pooled gains of `draws` sets of k random-number factors (the 'is it luck?' reference)."""
    def one(d):
        rng = np.random.default_rng(seed + d)
        extra = {f"noise_{j}": pd.Series(rng.standard_normal(len(prices)), index=prices.index) for j in range(k)}
        g = pooled_gain(base, q4_predictions(features, extra))
        return {"draw": d, "k": k, **{f"{m}_{key}": g[m][key] for m in MODELS for key in ("auc_gain", "bal_gain")}}
    if n_jobs == 1:
        rows = [one(d) for d in range(draws)]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs)(delayed(one)(d) for d in range(draws))
    return pd.DataFrame(rows)


def p_value(observed: float, null: pd.Series) -> float:
    """Share of random sets with a pooled gain at least as large (with the usual +1 so it is never 0)."""
    null = null.dropna()
    return float((1 + (null >= observed).sum()) / (1 + len(null)))
