"""Direction test: is the story behind a factor right about which way the factor points? (16 Nov)

Liu (12 Nov): with the right instruction the LLM's factors should be interpretable, "something that makes
sense in 5 minutes". A story that makes sense makes a prediction: a HIGHER value of the factor should make an
up day tomorrow more likely, or less likely. Here the LLM states that direction in advance for every factor
(expected_sign), and the data checks it. Nothing is selected on results, so every day can be used and no
model is trained: this test does not depend on the 22-factor model, on scikit-learn or on random seeds.

    edge        direct edge of a factor: ROC AUC of the raw factor against "AAPL closes higher tomorrow",
                minus 0.5, computed inside each year and averaged (so a year's overall up/down rate cannot
                create an edge). Positive = higher values go with up days.
    match rate  share of factors whose edge has the sign the LLM stated. Pure guessing gives 50%.
    p-value     the up/down labels are shuffled inside each year many times, which breaks any link between
                factors and tomorrow but keeps the factors (and how similar they are to each other) as they are.
    persistence correlation, across factors, between the edge on the search days (Q1-Q3) and the edge on the
                sealed Q4 days: does a direction seen in the data carry over to days the search never used?
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.data import quarterly_split

from .formula import evaluate
from .scoring import TARGET, YEARS

UP, DOWN = "higher_means_up", "higher_means_down"


def day_blocks(features: pd.DataFrame, which: str, years=YEARS) -> list:
    """[(dates, labels)] per year. 'search' = Q1-Q3 (training + validation days), 'q4' = the sealed test
    days, 'all' = both. The same days as the search and the final test use (src.data.quarterly_split)."""
    data = features.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    out = []
    for year in years:
        s = quarterly_split(data, year)
        parts = {"search": [s["train"], s["validation"]], "q4": [s["test"]],
                 "all": [s["train"], s["validation"], s["test"]]}[which]
        d = pd.concat(parts)
        out.append((pd.DatetimeIndex(d["Date"]), d[TARGET].to_numpy(dtype=float)))
    return out


def factor_table(formulas: dict, prices: pd.DataFrame) -> pd.DataFrame:
    """Daily values of every formula ({name: formula}); noise(<seed>) = random numbers."""
    cols = {}
    for name, formula in formulas.items():
        m = re.fullmatch(r"noise\((\d+)\)", formula.replace(" ", ""))
        cols[name] = (pd.Series(np.random.default_rng(int(m.group(1))).standard_normal(len(prices)), index=prices.index)
                      if m else evaluate(formula, prices))
    table = pd.DataFrame(cols)
    table.index = pd.to_datetime(table.index)
    return table


def usable(table: pd.DataFrame, blocks: list) -> pd.Series:
    """A factor can be tested if it has values on at least 95% of the days and varies in every year."""
    ok = pd.Series(True, index=table.columns)
    for dates, _ in blocks:
        x = table.reindex(dates)
        ok &= (x.notna().mean() >= 0.95) & (x.nunique() > 1)
    return ok


def _ranks(table: pd.DataFrame, blocks: list) -> list:
    out = []
    for dates, _ in blocks:
        x = table.reindex(dates).to_numpy(dtype=float)
        med = np.nanmedian(x, axis=0)
        x = np.where(np.isnan(x), np.where(np.isfinite(med), med, 0.0), x)   # a missing day carries no information
        out.append(np.apply_along_axis(rankdata, 0, x))
    return out


def _edges(ranks: list, labels: list) -> np.ndarray:
    num, den = 0.0, 0
    for r, y in zip(ranks, labels):
        n1 = y.sum()
        n0 = len(y) - n1
        auc = (y @ r - n1 * (n1 + 1) / 2) / (n1 * n0)        # rank-sum form of the ROC AUC, all factors at once
        num, den = num + (auc - 0.5) * len(y), den + len(y)
    return num / den


def edges(table: pd.DataFrame, blocks: list) -> pd.Series:
    """Direct edge (AUC - 0.5, averaged over the years) of every factor on the given days."""
    return pd.Series(_edges(_ranks(table, blocks), [y for _, y in blocks]), index=table.columns)


def shuffled_edges(table: pd.DataFrame, blocks: list, n: int = 5000, seed: int = 0) -> np.ndarray:
    """Edges after shuffling the labels inside each year, n times: what 'no link at all' looks like."""
    rng = np.random.default_rng(seed)
    ranks, labels = _ranks(table, blocks), [y for _, y in blocks]
    return np.array([_edges(ranks, [rng.permutation(y) for y in labels]) for _ in range(n)])


def stated_signs(expected) -> np.ndarray:
    return np.where(pd.Series(expected).to_numpy() == UP, 1.0, -1.0)


def match_test(table: pd.DataFrame, expected, blocks: list, n: int = 5000, seed: int = 0) -> dict:
    """How often the stated direction is right, against shuffled labels (one-sided p for both sides)."""
    want = stated_signs(expected)
    e = edges(table, blocks).to_numpy()
    null = shuffled_edges(table, blocks, n, seed)
    match, signed = float((np.sign(e) == want).mean()), float((e * want).mean())
    null_match, null_signed = (np.sign(null) == want).mean(axis=1), (null * want).mean(axis=1)
    return {"factors": len(want), "days": int(sum(len(y) for _, y in blocks)),
            "match_rate": match, "p_better_than_chance": float((1 + (null_match >= match).sum()) / (n + 1)),
            "p_worse_than_chance": float((1 + (null_match <= match).sum()) / (n + 1)),
            "mean_signed_edge": signed, "p_signed_edge": float((1 + (null_signed >= signed).sum()) / (n + 1))}


def persistence(table: pd.DataFrame, search: list, q4: list, n: int = 5000, seed: int = 0) -> dict:
    """Does the direction seen on the search days carry over to the sealed Q4 days?"""
    es, eq = edges(table, search).to_numpy(), edges(table, q4).to_numpy()
    null = shuffled_edges(table, q4, n, seed)
    corr = float(np.corrcoef(es, eq)[0, 1])
    agree = float((np.sign(es) == np.sign(eq)).mean())
    null_corr = np.array([np.corrcoef(es, x)[0, 1] for x in null])
    null_agree = (np.sign(null) == np.sign(es)).mean(axis=1)
    return {"factors": table.shape[1], "correlation": corr, "p_correlation": float((1 + (null_corr >= corr).sum()) / (n + 1)),
            "same_direction": agree, "p_same_direction": float((1 + (null_agree >= agree).sum()) / (n + 1))}


def composite(table: pd.DataFrame, signs) -> pd.Series:
    """One signal from many factors with given directions: each factor is standardised with its own PAST
    values only (expanding window, at least 60 days), turned so that higher means 'up', and averaged."""
    z = (table - table.expanding(60).mean().shift(1)) / table.expanding(60).std().shift(1)
    return (z.clip(-3, 3) * np.asarray(signs, dtype=float)).mean(axis=1)


def composite_test(signal: pd.Series, blocks: list, n: int = 5000, seed: int = 0) -> dict:
    """ROC AUC of one signal on the given days, with a shuffled-label p-value."""
    t = signal.to_frame("signal")
    e = float(edges(t, blocks).iloc[0])
    null = shuffled_edges(t, blocks, n, seed)[:, 0]
    return {"auc": 0.5 + e, "p": float((1 + (null >= e).sum()) / (n + 1))}
