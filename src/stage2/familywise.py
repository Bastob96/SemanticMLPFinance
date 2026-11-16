"""Family-wise ("best of N") bars, factor signs and recurrence: luck-proof judging of selected factors.

Why (results of 16 Nov): the keep rules were set so that about 1 random-number factor in 30 passes. That is
fine for one factor, but a run tests many (48 in an 8-round run, about 100 in a batch) and keeps the best,
so the right comparison is the BEST of N random-number factors, not one. In batch mode 11 random-number
factors got through the old bars in 3 runs. This is the winner's curse (multiple testing).

    bar(null, n)      gain a factor must reach so that the best of n random-number factors reaches it only
                      5% of the time (family-wise error rate 5%). Larger n -> higher bar.
    lr_signs(...)     the sign of a factor's Logistic Regression coefficient in each year (trained on Q1-Q2,
                      as in the search). A real effect should keep its direction; for a random factor all
                      4 years agree only 1 time in 8, and match a sign stated in advance only 1 time in 16.
    same_factor(...)  two kept factors count as the same idea if their formulas are equal or their values
                      are almost identical (|Spearman correlation| >= 0.9 on the Q1-Q3 days).
Nothing here uses the Q4 test quarters.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data import quarterly_split
from src.models import fit_logistic_regression

from .formula import evaluate
from .scoring import BASE, LR_SEED, TARGET, YEARS


def values_of(formula: str, prices: pd.DataFrame) -> pd.Series:
    """Daily values of a formula; noise(<seed>) = the random numbers used by the noise checks."""
    m = re.fullmatch(r"noise\((\d+)\)", formula.replace(" ", ""))
    if m:
        return pd.Series(np.random.default_rng(int(m.group(1))).standard_normal(len(prices)), index=prices.index)
    return evaluate(formula, prices)


def bar(null, n: int, level: float = 0.95) -> dict:
    """Gain that the best of n random-number factors exceeds only (1 - level) of the time.

    The best of n draws stays below x with probability F(x)^n, so the bar is the single-factor quantile
    at level ** (1 / n). With enough random factors in the tail the empirical quantile is used; otherwise a
    normal curve fitted to the random factors (both are returned so the choice can be checked)."""
    null = np.asarray(pd.Series(null).dropna(), dtype=float)
    q = level ** (1 / max(int(n), 1))
    normal = float(null.mean() + null.std(ddof=1) * norm.ppf(q))
    tail = len(null) * (1 - q)                         # expected number of random factors above the bar
    empirical = float(np.quantile(null, q)) if tail >= 5 else np.nan
    return {"n": int(n), "quantile": q, "bar": empirical if tail >= 5 else normal,
            "bar_empirical": empirical, "bar_normal": normal, "method": "empirical" if tail >= 5 else "normal"}


def lr_signs(values: pd.Series, features: pd.DataFrame, years=YEARS) -> list:
    """+1 / -1: sign of the factor's LR coefficient in each year (22 factors + this one, trained on Q1-Q2,
    the factor clipped to the training 1st/99th percentiles, as in the search). Inputs are standardised
    inside the model, so the sign is comparable across years."""
    data = features.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    v = pd.Series(values, name="_f")
    v.index = pd.to_datetime(v.index)
    data = data.merge(v.rename("_f"), left_on="Date", right_index=True, how="left")
    out = []
    for year in years:
        train = quarterly_split(data, year)["train"].copy()
        lo, hi = train["_f"].quantile([0.01, 0.99])
        train["_f"] = train["_f"].clip(lo, hi)
        model = fit_logistic_regression(train[BASE + ["_f"]], train[TARGET], LR_SEED)
        out.append(int(np.sign(model.named_steps["classifier"].coef_[0][-1])))
    return out


def sign_summary(signs: list, expected: str | None = None) -> dict:
    """How consistent the direction is, and (if stated in advance) how often it matches the prediction."""
    s = np.asarray(signs)
    out = {"signs": "".join("+" if x > 0 else "-" for x in s), "same_sign_years": int(max((s > 0).sum(), (s < 0).sum()))}
    if expected in ("higher_means_up", "higher_means_down"):
        want = 1 if expected == "higher_means_up" else -1
        out["matches_prediction"] = int((s == want).sum())
    return out


def check_days(features: pd.DataFrame) -> pd.DatetimeIndex:
    d = pd.to_datetime(features["Date"])
    return pd.DatetimeIndex(d[(d.dt.year >= 2022) & (d.dt.month <= 9)])


def normalise(formula: str) -> str:
    return re.sub(r"\s+", "", str(formula))


def same_factor(a: pd.Series, b: pd.Series, days, threshold: float = 0.9) -> bool:
    x, y = a.reindex(days), b.reindex(days)
    return bool(abs(x.corr(y, method="spearman")) >= threshold)
