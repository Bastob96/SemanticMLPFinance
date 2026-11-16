"""Set the pass bar for 'grading on a curve' (src/stage2/curve.py) before looking at any real factor again.

Two kinds of factor that should NOT pass:
  - 48 new template formulas (no LLM; a different seed from the template class)
  - 48 factors made of random numbers (same seeds as calibrate_noise_v2.py)
Each is tested alone (added to the 22 factors), its gains are graded on the curve, and we count how often it
would pass "beats the typical factor in at least 3 of 4 years, by at least <bar> on average".
The lowest bar at which at most about 1 in 30 of either kind passes is written to results/stage2/rules_v2_curve.json
(the other rules are copied from rules_v2.json).

Usage: python experiments/calibrate_curve.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import build_stage1_features  # noqa: E402
from src.stage2.curve import Curve, template_formulas  # noqa: E402
from src.stage2.formula import wide_prices  # noqa: E402
from src.stage2.scoring import Scorer  # noqa: E402
from src.stage2.selection import summarise_result  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "results/stage2"
TARGET_RATE = 1 / 30

if __name__ == "__main__":
    raw = pd.read_csv(ROOT / "data/processed/aligned_daily_ohlcv_5y.csv")
    features, prices = build_stage1_features(raw), wide_prices(raw)
    scorer, dates = Scorer(features), pd.to_datetime(features["Date"])
    curve = Curve(scorer, prices)
    off = curve.offsets({})
    print("Typical (median) gain of the template class in each year:")
    print(off.round(4).to_string())

    held_out = template_formulas(48, 202, prices, exclude=[f for f, _ in curve.panel.values()])
    noise = {f"noise{i}": pd.Series(np.random.default_rng(1000 + i).normal(size=len(dates)), index=dates.values)
             for i in range(48)}
    rows = []
    for kind, items in (("template", {n: v for n, (_, v) in held_out.items()}), ("noise", noise)):
        for name, values in items.items():
            per_year = scorer.gains({name: values}, {})
            r, c = summarise_result(per_year), summarise_result(curve.apply(per_year, {}))
            if not r["valid"]:
                continue
            rows.append({"kind": kind, "factor": name, "raw_gain": r["gain"], "raw_years": r["years"],
                         "curved_gain": c["gain"], "curved_years": c["years"],
                         **{f"curved_{y}": g for y, g in c["by_year"].items()}})
            print(f"  {kind} {name}: raw {r['gain']:+.4f} ({r['years']}/4) | curved {c['gain']:+.4f} ({c['years']}/4)",
                  flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "curve_null.csv", index=False)
    pd.DataFrame([{"name": n, "formula": f} for n, (f, _) in curve.panel.items()]).to_csv(OUT / "curve_class.csv", index=False)

    print("\nShare that would pass (3+ of 4 years and average gain >= bar)")
    for kind in ("template", "noise"):
        k = d[d.kind == kind]
        raw_txt = " | ".join(f">= {b}: {((k.raw_years >= 3) & (k.raw_gain >= b)).mean():.0%}" for b in (0, 0.005, 0.01))
        cur_txt = " | ".join(f">= {b}: {((k.curved_years >= 3) & (k.curved_gain >= b)).mean():.0%}"
                             for b in (0, 0.005, 0.01, 0.015))
        print(f"  {kind} ({len(k)}): without the curve {raw_txt}\n      with the curve {cur_txt}")

    bar = next((b for b in (0.0, 0.005, 0.01, 0.015, 0.02)
                if all(((d[d.kind == k].curved_years >= 3) & (d[d.kind == k].curved_gain >= b)).mean() <= TARGET_RATE
                       for k in ("template", "noise"))), 0.02)
    rules = json.loads((OUT / "rules_v2.json").read_text())
    rules.update(curve=True, team_gain=max(bar, 0.005))
    (OUT / "rules_v2_curve.json").write_text(json.dumps(rules, indent=2) + "\n")
    print(f"\nCurved bar: {rules['team_gain']} -> written to results/stage2/rules_v2_curve.json")
