"""Grading on a curve: compare a factor with what a typical factor does in the same year.

Why (21 Oct): in each year the validation quarter (Q3) reacts to ANY new price-based factor in a similar way.
In 2024 about 90% of all factors made the score worse (Claude's, last week's and the template formulas with no
LLM alike); in 2023 about 80-90% made it better. So "helped in 3 of 4 years" mostly measured the years, not
the factors. Like a hard exam that almost everyone failed, the fair question is: did this factor do better
than a typical factor did in that year?

A fixed "class" of template formulas (no LLM, made by RandomProposer with a fixed seed) is tested the same
way as each factor, on top of the same team. A factor's curved gain in a year = its gain minus the median
gain of the class in that year (for the average, for Logistic Regression and for Random Forest separately).
The class is chosen before any real run and never changes.
"""

from __future__ import annotations

import pandas as pd

from .formula import FormulaError, evaluate, validate
from .llm import RandomProposer

PANEL_SIZE, PANEL_SEED = 24, 101


def template_formulas(n, seed, prices, exclude=()):
    """n different valid template formulas (no LLM) with their values: {name: (formula, values)}."""
    rng, out, seen, tries = RandomProposer(seed), {}, set(exclude), 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        f = rng._formula()
        if f in seen:
            continue
        seen.add(f)
        try:
            validate(f)
            v = evaluate(f, prices)
        except (FormulaError, KeyError, ValueError):
            continue
        if v.loc["2022":].isna().any() or v.loc["2022":].nunique() < 10:
            continue
        out[f"template_{seed}_{len(out):02d}"] = (f, v)
    return out


class Curve:
    """Median gain of the template class in each year, on top of a given team (cached per team)."""

    def __init__(self, scorer, prices, size=PANEL_SIZE, seed=PANEL_SEED):
        self.scorer = scorer
        self.panel = template_formulas(size, seed, prices)
        self._offsets = {}

    def offsets(self, team: dict) -> pd.DataFrame:
        key = tuple(sorted(team))
        if key not in self._offsets:
            rows = [self.scorer.gains({n: v}, team) for n, (_, v) in self.panel.items()]
            ok = [r for r in rows if r["valid"].astype(bool).all()]
            all_rows = pd.concat(ok, ignore_index=True)
            self._offsets[key] = all_rows.groupby("year")[["auc_gain", "auc_gain_lr", "auc_gain_rf"]].median()
        return self._offsets[key]

    def apply(self, per_year: pd.DataFrame, team: dict) -> pd.DataFrame:
        """The same table with every gain replaced by its curved gain (gain minus the class median)."""
        if per_year.empty or not per_year["valid"].astype(bool).all():
            return per_year
        off = self.offsets(team)
        out = per_year.copy()
        for c in ("auc_gain", "auc_gain_lr", "auc_gain_rf"):
            out[c] = out[c] - out["year"].map(off[c])
        return out
