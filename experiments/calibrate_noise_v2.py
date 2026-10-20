"""How often does pure noise pass the new team-trial and pair-trial bars? Sets the bars for the version 2 loop.

Same idea as experiments/calibrate_noise.py (version 1), now for the two new tests:
  - team trial: a noise factor added on top of the 22 factors plus a real selected team
  - pair trial: two noise factors added together on top of the same team; "extra" = pair gain minus the
    better of the two alone on the team
The pairs are tried twice: any two noise factors, and only noise factors that look "promising"
(the loop only pairs promising factors, which makes lucky pairs more likely).

Usage: python experiments/calibrate_noise_v2.py --n 48
Writes results/stage2/noise_null_v2_team.csv, noise_null_v2_pairs.csv and rules_v2.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import build_stage1_features  # noqa: E402
from src.stage2.formula import evaluate, wide_prices  # noqa: E402
from src.stage2.scoring import Scorer  # noqa: E402
from src.stage2.selection import Rules, summarise_result  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "results/stage2"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--team-run", type=Path, default=OUT / "runs/20261015_1635_claude",
                    help="run whose kept factors form the team used here")
    args = ap.parse_args()

    raw = pd.read_csv(ROOT / "data/processed/aligned_daily_ohlcv_5y.csv")
    features, prices = build_stage1_features(raw), wide_prices(raw)
    kept = json.loads((args.team_run / "kept_factors.json").read_text())
    team = {k["name"]: evaluate(k["formula"], prices) for k in kept}
    print(f"Team used for the check: {list(team)}")
    scorer, dates = Scorer(features), pd.to_datetime(features["Date"])
    noise = {f"noise{i}": pd.Series(np.random.default_rng(1000 + i).normal(size=len(dates)), index=dates.values)
             for i in range(args.n)}

    team_rows = []
    for name, values in noise.items():
        res = summarise_result(scorer.gains({name: values}, team))
        team_rows.append({"factor": name, **{k: v for k, v in res.items() if k != "by_year"},
                          **{f"gain_{y}": g for y, g in res["by_year"].items()}})
        print(f"  team trial {name}: {res['gain']:+.4f} ({res['years']}/4)", flush=True)
    team_df = pd.DataFrame(team_rows).set_index("factor")
    team_df.to_csv(OUT / "noise_null_v2_team.csv")

    promising = [n for n in team_df.index if team_df.loc[n, "gain"] > 0 and team_df.loc[n, "years"] >= 2]
    names = list(noise)
    pairs = [(names[i], names[(i + 1) % len(names)], "any") for i in range(0, len(names), 2)]
    pairs += [(promising[i], promising[i + 1], "promising") for i in range(0, len(promising) - 1)]
    pair_rows = []
    for a, b, kind in pairs:
        res = summarise_result(scorer.gains({a: noise[a], b: noise[b]}, team))
        extra = res["gain"] - max(team_df.loc[a, "gain"], team_df.loc[b, "gain"])
        pair_rows.append({"a": a, "b": b, "kind": kind, "gain": res["gain"], "years": res["years"],
                          "lr": res["lr"], "rf": res["rf"], "extra": extra})
        print(f"  pair {a}+{b} ({kind}): {res['gain']:+.4f} ({res['years']}/4), extra {extra:+.4f}", flush=True)
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(OUT / "noise_null_v2_pairs.csv", index=False)

    print("\nTEAM TRIAL: share of noise factors that would join the team")
    for bar in (0.0, 0.005, 0.01, 0.015):
        share = ((team_df.years >= 3) & (team_df.gain >= bar)).mean()
        print(f"  helped 3+/4 years and gain >= {bar:.3f}: {share:.0%}")
    print(f"  noise team-trial gain: mean {team_df.gain.mean():+.4f}, std {team_df.gain.std():.4f}")

    print("\nPAIR TRIAL: share of noise pairs that would join the team")
    for kind in ("any", "promising"):
        d = pair_df[pair_df.kind == kind]
        print(f"  {kind} pairs ({len(d)}): gain mean {d.gain.mean():+.4f}, extra mean {d.extra.mean():+.4f}")
        for bar in (0.01, 0.015, 0.02):
            for syn in (0.0, 0.005, 0.01):
                share = ((d.years >= 3) & (d.gain >= bar) & (d.extra >= syn)).mean()
                print(f"    gain >= {bar:.3f} and extra >= {syn:.3f}: {share:.0%}")

    # choose the lowest bars at which at most ~1 in 30 noise cases pass (the version 1 standard)
    rules = Rules()
    for bar in (0.0, 0.005, 0.01, 0.015, 0.02):
        if ((team_df.years >= 3) & (team_df.gain >= bar)).mean() <= 1 / 30:
            rules.team_gain = bar
            break
    d = pair_df[pair_df.kind == "promising"] if (pair_df.kind == "promising").sum() >= 10 else pair_df
    done = False
    for bar in (0.01, 0.015, 0.02, 0.025):
        for syn in (0.0, 0.005, 0.01):
            if ((d.years >= 3) & (d.gain >= bar) & (d.extra >= syn)).mean() <= 1 / 30:
                rules.pair_gain, rules.pair_synergy, done = bar, syn, True
                break
        if done:
            break
    rules.team_gain = max(rules.team_gain, 0.01)      # never below the version 1 bar
    (OUT / "rules_v2.json").write_text(json.dumps(rules.as_dict(), indent=2) + "\n")
    print(f"\nBars written to results/stage2/rules_v2.json: {rules.as_dict()}")
