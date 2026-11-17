"""Random-number factors for the family-wise ("best of N") bars (src/stage2/familywise.py).

Each random-number factor goes through exactly the solo test of the search (added to the 22 factors,
trained on Q1-Q2 and scored on Q3 in each year); each random pair through the pair test (two added together).
Q4 is never used. Safe to stop and run again: it continues where it stopped.

Usage (same Python as the runs, scikit-learn 1.7.2):
    python experiments/calibrate_familywise.py                  # 2000 single factors and 1000 pairs
    python experiments/calibrate_familywise.py --solo 300 --pairs 100    # quick look
Writes results/stage2/familywise/noise_solo.csv and noise_pairs.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import build_stage1_features  # noqa: E402
from src.stage2.formula import wide_prices  # noqa: E402
from src.stage2.scoring import Scorer  # noqa: E402
from src.stage2.selection import summarise_result  # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "data/processed/aligned_daily_ohlcv_5y.csv"
OUT = ROOT / "results/stage2/familywise"
SOLO_SEED, PAIR_SEED = 500_000, 700_000      # far from the seeds used anywhere else


def noise(seed, index):
    return pd.Series(np.random.default_rng(seed).standard_normal(len(index)), index=index)


def one(scorer, index, seeds):
    add = {f"noise_{s}": noise(s, index) for s in seeds}
    res = summarise_result(scorer.gains(add, {}))
    if not res.get("valid"):
        return None
    return {"seeds": "+".join(map(str, seeds)), "gain": res["gain"], "years": res["years"],
            "lr": res["lr"], "rf": res["rf"], **{f"gain_{y}": g for y, g in res["by_year"].items()}}


def run(kind, target, scorer, index, jobs, path):
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    start = len(done)
    if start >= target:
        print(f"{kind}: {start} already done")
        return done
    print(f"{kind}: {start} done, running {target - start} more", flush=True)
    scorer.gains({"_warm": noise(1, index)}, {})            # cache the 22-factor scores before copying the scorer
    for lo in range(start, target, 200):                    # in chunks of 200, saved after each chunk
        hi, t0 = min(lo + 200, target), time.time()
        if kind == "solo":
            seeds = [(SOLO_SEED + i,) for i in range(lo, hi)]
        else:
            seeds = [(PAIR_SEED + 2 * i, PAIR_SEED + 2 * i + 1) for i in range(lo, hi)]
        rows = Parallel(n_jobs=jobs)(delayed(one)(scorer, index, s) for s in seeds)
        done = pd.concat([done, pd.DataFrame([r for r in rows if r])], ignore_index=True)
        done.to_csv(path, index=False)
        print(f"  {kind} {hi}/{target} ({time.time() - t0:.0f}s)", flush=True)
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", type=int, default=2000)
    ap.add_argument("--pairs", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA)
    scorer, index = Scorer(build_stage1_features(raw)), wide_prices(raw).index
    solo = run("solo", args.solo, scorer, index, args.jobs, OUT / "noise_solo.csv")
    pairs = run("pairs", args.pairs, scorer, index, args.jobs, OUT / "noise_pairs.csv")
    for name, d in (("single factors", solo), ("pairs", pairs)):
        print(f"{name}: {len(d)} | gain mean {d.gain.mean():+.4f}, sd {d.gain.std():.4f}, "
              f"99th pct {d.gain.quantile(0.99):+.4f} | helped 3+ years: {(d.years >= 3).mean():.1%}")
