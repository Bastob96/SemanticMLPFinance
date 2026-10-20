"""Keep rules and helpers for the interaction-aware search (team trial, pair trials, promising list).

Plain-language summary:
  - Solo test: the factor is added to the 22 Stage 1 factors (as in version 1).
  - Team trial: the factor is added to the 22 factors plus the selected team.
    It joins the team if the team gets clearly better, consistently across years.
  - Pair trial: two promising factors, preferably of different kinds, are added together.
    Both join the team if together they beat the better one alone, consistently.
  - Promising list: factors that are weak alone but show some sign of life are not thrown away.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

YEARS = (2022, 2023, 2024, 2025)


@dataclass
class Rules:
    min_years: int = 3            # must help in at least 3 of the 4 years
    team_gain: float = 0.01       # team trial: average gain needed (set from the random-number check)
    pair_gain: float = 0.015      # pair trial: average gain of the pair needed
    pair_synergy: float = 0.005   # pair must beat the better of its two members alone by this much
    max_corr: float = 0.9         # anything this similar to an existing or selected factor is a copy
    pair_max_corr: float = 0.7    # the two members of a pair should be different from each other
    team_size: int = 6            # the selected team stays small
    pairs_per_round: int = 6      # pair trials per round (keeps the number of "lottery tickets" low)

    def as_dict(self):
        return asdict(self)


def summarise_result(per_year: pd.DataFrame) -> dict:
    """Turn the per-year table from Scorer.gains into a few numbers."""
    if per_year.empty or not per_year["valid"].astype(bool).all():
        return {"valid": False}
    g = per_year.set_index("year")
    return {"valid": True,
            "gain": float(g["auc_gain"].mean()),
            "years": int((g["auc_gain"] > 0).sum()),
            "lr": float(g["auc_gain_lr"].mean()),
            "rf": float(g["auc_gain_rf"].mean()),
            "by_year": {int(y): float(v) for y, v in g["auc_gain"].items()}}


def group_of(formula: str, role: str | None = None) -> str:
    """Which kind of clue it is: 'market' (market or peers), 'relative' (Apple vs others) or 'apple' (Apple only).

    Uses Claude's role label when there is one, otherwise the stocks the formula uses.
    """
    if role in ("market_condition", "peer_lead"):
        return "market"
    if role == "relative_move":
        return "relative"
    if role == "apple_own":
        return "apple"
    uses_aapl = "AAPL_" in formula
    uses_other = any(f"{t}_" in formula for t in ("QQQ", "MSFT", "NVDA"))
    if uses_aapl and uses_other:
        return "relative"
    return "apple" if uses_aapl else "market"


def passes(result: dict, rules: Rules, gain_bar: float) -> bool:
    return bool(result.get("valid")) and result["years"] >= rules.min_years and result["gain"] >= gain_bar


def is_promising(rec: dict, rules: Rules) -> bool:
    """Weak alone but some sign of life: positive average and helped in 2+ years, alone or on the team, and not a copy."""
    if rec.get("status") == "invalid" or rec.get("max_corr", 1) >= rules.max_corr:
        return False
    best_gain = max(rec.get("auc_gain", -1), rec.get("team_gain", -1) if rec.get("team_size") else -1)
    best_years = max(rec.get("years_helped", 0), rec.get("team_years_helped", 0) if rec.get("team_size") else 0)
    return best_gain > 0 and best_years >= 2


def choose_pairs(new_names, pool_names, info, corr, rules: Rules, forced=()):
    """Pick which pairs to try this round.

    forced: (partner, parent) pairs Claude designed to work together; tried first.
    Then each new promising factor is paired with factors on the promising list, preferring
    different kinds (e.g. a market-condition factor with an Apple-relative one) and the
    strongest signs of life. info[name] = {"group", "score"}; corr(a, b) = similarity of two factors.
    """
    chosen, seen = [], set()

    def add(a, b):
        key = tuple(sorted((a, b)))
        if a == b or key in seen or len(chosen) >= rules.pairs_per_round:
            return
        if abs(corr(a, b)) >= rules.pair_max_corr:
            return
        seen.add(key)
        chosen.append((a, b))

    for a, b in forced:
        if a in info and b in info:
            add(a, b)
    candidates = []
    for a in new_names:
        for b in pool_names:
            if a == b or a not in info or b not in info:
                continue
            bonus = 0.005 if info[a]["group"] != info[b]["group"] else 0.0
            candidates.append((info[a]["score"] + info[b]["score"] + bonus, a, b))
    for _, a, b in sorted(candidates, reverse=True):
        add(a, b)
    return chosen
