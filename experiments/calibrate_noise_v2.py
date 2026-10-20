"""How often does pure noise pass the new team-trial and pair-trial bars? Sets the bars for the version 2 loop.

Same idea as experiments/calibrate_noise.py (version 1), now for the two new tests:
  - team trial: a noise factor added on top of the 22 factors plus a real selected team
  - pair trial: two noise factors added together on top of the same team
The pairs are tried twice: random pairs of noise factors, and pairs of noise factors that look "promising"
(the loop only pairs promising factors, so the members were already lucky once).

What the first check showed (20 Oct): two lucky noise factors added together simply add up their luck,
so "the pair beats the better one alone" passed for about half of the promising noise pairs. The pair rule
therefore compares the pair with the two gains ADDED UP ("extra" = pair gain - gain of a alone - gain of b
alone, all on top of the same team). Luck adds up; real teamwork gives more than the sum.

Usage: python experiments/calibrate_noise_v2.py --n 48 --pairs 64
Writes results/stage2/noise_null_v2_team.csv, noise_null_v2_pairs.csv and rules_v2.json.
"""

from __future__ import annotations

import argparse
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
from src.stage2.formula import evaluate, wide_prices  # noqa: E402
from src.stage2.scoring import Scorer  # noqa: E402
from src.stage2.selection import Rules, summarise_result  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "results/stage2"
TARGET_RATE = 1 / 30          # the version 1 standard: about 1 noise factor in 30 gets through


def pass_rate(d, gain_bar, extra_bar, min_years=3):
    return float(((d.years >= min_years) & (d.gain >= gain_bar) & (d.extra >= extra_bar)).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48, help="noise factors for the team trial")
    ap.add_argument("--pairs", type=int, default=64, help="pairs of each kind (random / promising)")
    ap.add_argument("--team-run", type=Path, default=OUT / "runs/20261015_1635_claude",
                    help="run whose kept factors form the team used here")
    ap.add_argument("--fresh", action="store_true", help="redo the team trials even if they are saved")
    args = ap.parse_args()

    raw = pd.read_csv(ROOT / "data/processed/aligned_daily_ohlcv_5y.csv")
    features, prices = build_stage1_features(raw), wide_prices(raw)
    kept = json.loads((args.team_run / "kept_factors.json").read_text())
    team = {k["name"]: evaluate(k["formula"], prices) for k in kept}
    print(f"Team used for the check: {list(team)}")
    scorer, dates = Scorer(features), pd.to_datetime(features["Date"])
    noise = {f"noise{i}": pd.Series(np.random.default_rng(1000 + i).normal(size=len(dates)), index=dates.values)
             for i in range(args.n)}

    team_file = OUT / "noise_null_v2_team.csv"
    if team_file.exists() and not args.fresh and len(pd.read_csv(team_file)) == args.n:
        team_df = pd.read_csv(team_file, index_col=0)
        print(f"Using the saved team trials of {args.n} noise factors")
    else:
        rows = []
        for name, values in noise.items():
            res = summarise_result(scorer.gains({name: values}, team))
            rows.append({"factor": name, **{k: v for k, v in res.items() if k != "by_year"},
                         **{f"gain_{y}": g for y, g in res["by_year"].items()}})
            print(f"  team trial {name}: {res['gain']:+.4f} ({res['years']}/4)", flush=True)
        team_df = pd.DataFrame(rows).set_index("factor")
        team_df.to_csv(team_file)

    rng = random.Random(0)
    names = list(noise)
    promising = [n for n in names if team_df.loc[n, "gain"] > 0 and team_df.loc[n, "years"] >= 2]
    combos = {"random": [tuple(sorted(rng.sample(names, 2))) for _ in range(4 * args.pairs)],
              "promising": [tuple(sorted(rng.sample(promising, 2))) for _ in range(4 * args.pairs)]}
    pair_rows = []
    for kind, pairs in combos.items():
        for a, b in list(dict.fromkeys(pairs))[:args.pairs]:         # distinct pairs, in a fixed random order
            res = summarise_result(scorer.gains({a: noise[a], b: noise[b]}, team))
            ga, gb = team_df.loc[a, "gain"], team_df.loc[b, "gain"]
            pair_rows.append({"a": a, "b": b, "kind": kind, "gain": res["gain"], "years": res["years"],
                              "lr": res["lr"], "rf": res["rf"], "a_alone": ga, "b_alone": gb,
                              "extra": res["gain"] - ga - gb, "extra_vs_better": res["gain"] - max(ga, gb)})
            print(f"  pair {a}+{b} ({kind}): {res['gain']:+.4f} ({res['years']}/4), "
                  f"extra over the sum {res['gain'] - ga - gb:+.4f}", flush=True)
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(OUT / "noise_null_v2_pairs.csv", index=False)

    print("\nTEAM TRIAL: share of noise factors that would join the team")
    for bar in (0.0, 0.005, 0.01, 0.015):
        share = ((team_df.years >= 3) & (team_df.gain >= bar)).mean()
        print(f"  helped 3+/4 years and gain >= {bar:.3f}: {share:.0%}")
    print(f"  noise team-trial gain: mean {team_df.gain.mean():+.4f}, std {team_df.gain.std():.4f}; "
          f"{len(promising)} of {len(team_df)} look promising")

    print("\nPAIR TRIAL: share of noise pairs that would join the team")
    for kind in ("random", "promising"):
        d = pair_df[pair_df.kind == kind]
        print(f"  {kind} pairs ({len(d)}): pair gain mean {d.gain.mean():+.4f}; "
              f"extra over the better one mean {d.extra_vs_better.mean():+.4f}; extra over the sum mean "
              f"{d.extra.mean():+.4f} (std {d.extra.std():.4f})")
        old = ((d.years >= 3) & (d.gain >= 0.015) & (d.extra_vs_better >= 0.005)).mean()
        print(f"    first rule (gain >= 0.015, beats the better one by 0.005): {old:.0%}")
        for bar in (0.01, 0.015, 0.02):
            print(f"    gain >= {bar:.3f}: " + " | ".join(
                f"extra >= {s}: {pass_rate(d, bar, s):.0%}" for s in (0.0, 0.005, 0.0075, 0.01, 0.0125)))

    # lowest bars at which at most ~1 in 30 noise cases pass, for both kinds of pairs
    rules = Rules()
    team_bar = next((b for b in (0.0, 0.005, 0.01, 0.015, 0.02)
                     if ((team_df.years >= 3) & (team_df.gain >= b)).mean() <= TARGET_RATE), 0.02)
    rules.team_gain = max(team_bar, 0.01)            # never below the version 1 bar
    chosen = None
    for bar in (0.01, 0.015, 0.02, 0.025):
        for s in (0.0, 0.005, 0.0075, 0.01, 0.0125, 0.015):
            if all(pass_rate(pair_df[pair_df.kind == k], bar, s) <= TARGET_RATE for k in ("random", "promising")):
                chosen = (bar, s)
                break
        if chosen:
            break
    if chosen is None:
        chosen = (0.025, 0.015)
        print("\nNo tried bar kept noise pairs below 1 in 30; using the strictest one tried.")
    rules.pair_gain, rules.pair_synergy = chosen
    (OUT / "rules_v2.json").write_text(json.dumps(rules.as_dict(), indent=2) + "\n")
    print(f"\nBars written to results/stage2/rules_v2.json: {rules.as_dict()}")
    for k in ("random", "promising"):
        print(f"  {k} noise pairs passing these bars: "
              f"{pass_rate(pair_df[pair_df.kind == k], rules.pair_gain, rules.pair_synergy):.1%}")
