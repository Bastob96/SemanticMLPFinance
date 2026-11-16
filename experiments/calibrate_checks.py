"""A longer measuring stick: check every factor on Q2 (trained on Q1) as well as Q3 (trained on Q1-Q2),
so each factor gets 8 checks (4 years x 2) instead of 4. Sets the bars with random numbers first.

Null factors (should not pass): 48 random-number factors and 48 template formulas (no LLM, seed 202).
Every factor is tested alone (added to the 22 factors). Pairs: 32 pairs of null factors that look promising.
The bars are chosen from the null factors only, then written to results/stage2/rules_v2_8checks.json.
The real factors (the 48 of 15 Oct and the 48 of 21 Oct) are scored in the same run and saved to
results/stage2/checks8_real.csv for later use, but play no part in choosing the bars.

Usage: python experiments/calibrate_checks.py
"""

from __future__ import annotations

import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import build_stage1_features  # noqa: E402
from src.stage2.curve import template_formulas  # noqa: E402
from src.stage2.formula import evaluate, wide_prices  # noqa: E402
from src.stage2.scoring import Scorer  # noqa: E402
from src.stage2.selection import summarise_result  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "results/stage2"
TARGET_RATE = 1 / 30
REAL_RUNS = [OUT / "runs/20261015_1635_claude", OUT / "runs_v2/20261021_1105_claude"]


def row_of(kind, name, res):
    return {"kind": kind, "factor": name, "gain": res["gain"], "checks": res["years"], "lr": res["lr"],
            "rf": res["rf"], **{f"g_{k}": v for k, v in res["by_check"].items()}}


def rate(d, k, bar):
    return float(((d.checks >= k) & (d.gain >= bar)).mean())


if __name__ == "__main__":
    raw = pd.read_csv(ROOT / "data/processed/aligned_daily_ohlcv_5y.csv")
    features, prices = build_stage1_features(raw), wide_prices(raw)
    scorer, dates = Scorer(features, checks=("q2", "q3")), pd.to_datetime(features["Date"])
    null = {f"noise{i}": ("noise", pd.Series(np.random.default_rng(1000 + i).normal(size=len(dates)), index=dates.values))
            for i in range(48)}
    null.update({n: ("template", v) for n, (_, v) in template_formulas(48, 202, prices).items()})

    rows, values = [], {}
    for name, (kind, v) in null.items():
        res = summarise_result(scorer.gains({name: v}, {}))
        if res["valid"]:
            rows.append(row_of(kind, name, res))
            values[name] = v
            print(f"  {kind} {name}: {res['gain']:+.4f} ({res['years']}/8)", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "checks8_null.csv", index=False)

    print("\nALONE: share of null factors passing 'helped in >= k of 8 checks and average gain >= bar'")
    for kind in ("noise", "template"):
        x = d[d.kind == kind]
        for k in (5, 6, 7):
            print(f"  {kind} ({len(x)}) k={k}: " + " | ".join(f">= {b}: {rate(x, k, b):.0%}" for b in (0, 0.005, 0.0075, 0.01)))
    k_bar = None
    for k in (6, 7, 5):
        for b in (0.0, 0.005, 0.0075, 0.01):
            if all(rate(d[d.kind == kind], k, b) <= TARGET_RATE for kind in ("noise", "template")):
                k_bar = (k, b)
                break
        if k_bar:
            break
    k_bar = k_bar or (7, 0.01)

    # pairs of promising null factors (helped in at least half the checks, positive average)
    promising = list(d[(d.gain > 0) & (d.checks >= 4)].factor)
    rng, pair_rows = random.Random(0), []
    combos = list(dict.fromkeys(tuple(sorted(rng.sample(promising, 2))) for _ in range(200)))[:32]
    single = d.set_index("factor")
    for a, b in combos:
        res = summarise_result(scorer.gains({a: values[a], b: values[b]}, {}))
        extra = res["gain"] - single.loc[a, "gain"] - single.loc[b, "gain"]
        pair_rows.append({"a": a, "b": b, "gain": res["gain"], "checks": res["years"], "extra": extra})
        print(f"  pair {a}+{b}: {res['gain']:+.4f} ({res['years']}/8), extra over the sum {extra:+.4f}", flush=True)
    p = pd.DataFrame(pair_rows)
    p.to_csv(OUT / "checks8_null_pairs.csv", index=False)
    print(f"\nPAIRS ({len(p)} promising null pairs): extra over the sum mean {p.extra.mean():+.4f}, std {p.extra.std():.4f}")
    pair_bar = None
    for s in (0.005, 0.0075, 0.01, 0.0125, 0.015):
        share = float(((p.checks >= k_bar[0]) & (p.gain >= 0.01) & (p.extra >= s)).mean())
        print(f"  helped >= {k_bar[0]}/8, gain >= 0.01, extra >= {s}: {share:.0%}")
        if pair_bar is None and share <= TARGET_RATE:
            pair_bar = s
    pair_bar = pair_bar or 0.015

    rules = json.loads((OUT / "rules_v2.json").read_text())
    rules.update(checks=["q2", "q3"], min_years=k_bar[0], team_gain=max(k_bar[1], 0.005), pair_gain=0.01,
                 pair_synergy=pair_bar)
    (OUT / "rules_v2_8checks.json").write_text(json.dumps(rules, indent=2) + "\n")
    print(f"\nBars for 8 checks: {rules}")

    # the real factors, scored the same way (saved only; not used for the bars)
    real = []
    for run in REAL_RUNS:
        board = pd.read_csv(run / "scoreboard.csv")
        for r in board.itertuples():
            if r.status == "invalid":
                continue
            v = evaluate(r.formula, prices)
            res = summarise_result(scorer.gains({r.name: v}, {}))
            if res["valid"]:
                real.append({**row_of(run.name, r.name, res), "formula": r.formula})
    pd.DataFrame(real).to_csv(OUT / "checks8_real.csv", index=False)
    print(f"Scored {len(real)} real factors with 8 checks -> results/stage2/checks8_real.csv")
