"""Time-order check for the direction test, with the no-LLM controls. (free: no API calls, no model training)

The direction test showed the LLM the search days (January-September) of ALL four years before it proposed
its round-2 factors. The sealed quarters were never shown, but the 2022 sealed quarter was then predicted
with evidence that includes 2023-2025. A strict reader asks: does it still work using only the past?

For every pool of factors, each factor gets a direction and all factors are combined into one signal
(src.stage2.direction.composite), scored by ROC AUC on the sealed Q4 days. Four ways to choose directions:

    stated      the direction the LLM wrote down (LLM pools only)
    all years   measured on the search days of all four years (what round 2 was shown)
    past only   for the Q4 of year Y: measured only on days BEFORE that quarter (walk-forward)
    other years for the Q4 of year Y: measured on the search days of the other three years

Pools: each model's round-1 factors (proposed blind, so the pool itself cannot have been picked using the
data), its round-2 new ideas, the 22 existing factors, template formulas (no LLM), and random numbers.
The last column tests each pool against the 22 existing factors on the same days (bootstrap of the days).

Fresh days: the price file also holds January-August 2026 (164 days). No search, test or prompt evidence in
this project has used them. Directions are measured on 2022-2025 only and the signal is scored once on 2026.

Usage:  python experiments/check_direction_time_order.py
Writes results/stage2/direction/time_order.csv and time_order_by_year.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import build_stage1_features  # noqa: E402
from src.stage2.context import BASE_FORMULAS  # noqa: E402
from src.stage2.direction import (composite, day_blocks, edges, factor_table, shuffled_edges,  # noqa: E402
                                  stated_signs, usable)
from src.stage2.formula import wide_prices  # noqa: E402
from src.stage2.scoring import YEARS  # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "data/processed/aligned_daily_ohlcv_5y.csv"
OUT = ROOT / "results/stage2/direction"
RANDOM_POOLS, POOL_SIZE, SHUFFLES, BOOT = 200, 100, 5000, 2000
FRESH_YEAR = 2026


def signs_from(table, blocks):
    return np.sign(edges(table, blocks).to_numpy())


def stitched(table, signs_by_year, q4):
    """One signal over the Q4 days, where each year's quarter uses that year's own directions."""
    parts = [composite(table, s).reindex(dates) for s, (dates, _) in zip(signs_by_year, q4)]
    return pd.concat(parts)


def auc(x, y):
    r = rankdata(x)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def by_year(signal, q4):
    out = []
    for dates, y in q4:
        x = signal.reindex(dates).fillna(signal.reindex(dates).median()).to_numpy()
        out.append(auc(x, y))
    return out


def pooled(aucs, q4):
    w = np.array([len(y) for _, y in q4], dtype=float)
    return float(np.dot(aucs, w) / w.sum())


def p_value(signal, q4):
    t = signal.to_frame("s")
    e = float(edges(t, q4).iloc[0])
    null = shuffled_edges(t, q4, SHUFFLES, 0)[:, 0]
    return float((1 + (null >= e).sum()) / (SHUFFLES + 1))


def versus(signal, reference, q4, seed=0):
    """Bootstrap of the Q4 days (inside each year): difference in AUC against the reference signal."""
    rng = np.random.default_rng(seed)
    data = [(signal.reindex(d).fillna(signal.reindex(d).median()).to_numpy(),
             reference.reindex(d).fillna(reference.reindex(d).median()).to_numpy(), y) for d, y in q4]
    w = np.array([len(y) for _, y in q4], dtype=float)
    diffs = []
    for _ in range(BOOT):
        d = []
        for a, b, y in data:
            i = rng.integers(0, len(y), len(y))
            while y[i].min() == y[i].max():
                i = rng.integers(0, len(y), len(y))
            d.append(auc(a[i], y[i]) - auc(b[i], y[i]))
        diffs.append(np.dot(d, w) / w.sum())
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)), float((diffs <= 0).mean())


def direction_sets(table, search, q4):
    """{way: [signs for the Q4 of each year]} measured from the data."""
    n = len(q4)
    return {"all years": [signs_from(table, search)] * n,
            "past only": [signs_from(table, search[:i + 1] + q4[:i]) for i in range(n)],
            "other years": [signs_from(table, search[:i] + search[i + 1:]) for i in range(n)]}


if __name__ == "__main__":
    raw = pd.read_csv(DATA)
    features, prices = build_stage1_features(raw), wide_prices(raw)
    search, q4, every = (day_blocks(features, w) for w in ("search", "q4", "all"))

    pools = {}
    base = factor_table(BASE_FORMULAS, prices)
    pools["the 22 existing factors"] = (base.loc[:, usable(base, every)], None)
    for folder in sorted(OUT.glob("*/factors.csv")):
        f, tag = pd.read_csv(folder), folder.parent.name
        llm = not tag.startswith("template")
        for rnd, label in ((1, "round 1 (blind)"), (2, "round 2 (new ideas)")):
            g = f[f["round"] == rnd]
            if rnd == 2:
                g = g[g["new_idea"].astype(str) == "True"]
            if not llm and rnd == 2:
                continue
            t = factor_table(dict(zip(g["name"], g["formula"])), prices)
            name = f"{tag}, {label}" if llm else f"templates, no LLM ({tag[-1]})"
            pools[name] = (t, stated_signs(g["expected_sign"]) if llm else None)

    reference = {}
    rows, years = [], []
    for name, (table, stated) in pools.items():
        ways = direction_sets(table, search, q4)
        if stated is not None:
            ways = {"stated": [stated] * len(q4), **ways}
        for way, signs in ways.items():
            signal = stitched(table, signs, q4)
            a = by_year(signal, q4)
            row = {"pool": name, "factors": table.shape[1], "directions": way, "auc_q4": pooled(a, q4), "p": p_value(signal, q4)}
            if name.startswith("the 22"):
                reference[way] = signal
            else:
                ref = reference[way if way != "stated" else "all years"]
                row["vs_22"], row["vs_22_low"], row["vs_22_high"], row["p_vs_22"] = versus(signal, ref, q4)
            rows.append(row)
            years.append({"pool": name, "directions": way, **{str(y): v for y, v in zip(YEARS, a)}})

    # random numbers: what a pool of factors with no information gives, directions measured the same way
    rng_rows = {w: [] for w in ("all years", "past only", "other years")}
    for k in range(RANDOM_POOLS):
        t = factor_table({f"n{k}_{j}": f"noise({1000 * k + j})" for j in range(POOL_SIZE)}, prices)
        for way, signs in direction_sets(t, search, q4).items():
            rng_rows[way].append(pooled(by_year(stitched(t, signs, q4), q4), q4))
    for way, v in rng_rows.items():
        rows.append({"pool": f"random numbers ({RANDOM_POOLS} pools of {POOL_SIZE})", "factors": POOL_SIZE, "directions": way,
                     "auc_q4": float(np.mean(v)), "random_95th": float(np.quantile(v, 0.95))})

    # ---- fresh days: January-August 2026, never used by any search, test or prompt evidence
    f26 = features.assign(Date=pd.to_datetime(features["Date"]))
    f26 = f26[f26["Date"].dt.year == FRESH_YEAR]
    fresh = [(pd.DatetimeIndex(f26["Date"]), f26["target_next_day_up"].to_numpy(dtype=float))]
    fresh_rows, ref26 = [], None
    for name, (table, stated) in pools.items():
        ways = {"measured on 2022-2025": signs_from(table, every)}
        if stated is not None:
            ways = {"stated": stated, **ways}
        for way, signs in ways.items():
            signal = composite(table, signs).reindex(fresh[0][0])
            row = {"pool": name, "factors": table.shape[1], "directions": way, "auc_2026": by_year(signal, fresh)[0],
                   "p": p_value(signal, fresh)}
            if name.startswith("the 22"):
                ref26 = signal
            else:
                row["vs_22"], row["vs_22_low"], row["vs_22_high"], row["p_vs_22"] = versus(signal, ref26, fresh)
            fresh_rows.append(row)
    v = []
    for k in range(RANDOM_POOLS):
        t = factor_table({f"n{k}_{j}": f"noise({1000 * k + j})" for j in range(POOL_SIZE)}, prices)
        v.append(by_year(composite(t, signs_from(t, every)).reindex(fresh[0][0]), fresh)[0])
    fresh_rows.append({"pool": f"random numbers ({RANDOM_POOLS} pools of {POOL_SIZE})", "factors": POOL_SIZE,
                       "directions": "measured on 2022-2025", "auc_2026": float(np.mean(v)), "random_95th": float(np.quantile(v, 0.95))})
    fresh_table = pd.DataFrame(fresh_rows)
    fresh_table.to_csv(OUT / "fresh_2026.csv", index=False)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "time_order.csv", index=False)
    pd.DataFrame(years).to_csv(OUT / "time_order_by_year.csv", index=False)

    print("One signal per pool, ROC AUC on the sealed Q4 days (0.50 = no information)\n")
    for way in ("stated", "all years", "past only", "other years"):
        print(f"Directions: {way}")
        for _, r in table[table["directions"] == way].iterrows():
            line = f"  {r['pool']:38} {int(r['factors']):3d} factors | AUC {r['auc_q4']:.3f}"
            if pd.notna(r.get("p")):
                line += f" (p {r['p']:.3f})"
            if pd.notna(r.get("vs_22")):
                line += f" | vs the 22: {r['vs_22']:+.3f} (95% {r['vs_22_low']:+.3f} to {r['vs_22_high']:+.3f}, p {r['p_vs_22']:.3f})"
            if pd.notna(r.get("random_95th")):
                line += f" | 95% of random pools below {r['random_95th']:.3f}"
            print(line)
        print()
    print("By year (same signals):")
    print(pd.DataFrame(years).round(3).to_string(index=False))
    print(f"\nFRESH DAYS: January-August {FRESH_YEAR} ({len(fresh[0][1])} days, up on {fresh[0][1].mean():.0%}), never used before")
    for _, r in fresh_table.iterrows():
        line = f"  {r['pool']:38} {r['directions']:22} | AUC {r['auc_2026']:.3f}"
        if pd.notna(r.get("p")):
            line += f" (p {r['p']:.3f})"
        if pd.notna(r.get("vs_22")):
            line += f" | vs the 22: {r['vs_22']:+.3f} (95% {r['vs_22_low']:+.3f} to {r['vs_22_high']:+.3f}, p {r['p_vs_22']:.3f})"
        if pd.notna(r.get("random_95th")):
            line += f" | 95% of random pools below {r['random_95th']:.3f}"
        print(line)
    print(f"\nWrote {OUT / 'time_order.csv'}, time_order_by_year.csv and fresh_2026.csv")
