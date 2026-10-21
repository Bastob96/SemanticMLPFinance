"""How do pure-noise 'factors' score? Sets the selection threshold for the loop.

Usage: python experiments/calibrate_noise.py --n 32
Writes results/stage2/noise_null.csv.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import build_stage1_features  # noqa: E402
from src.stage2 import Scorer, summarise  # noqa: E402

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()
    features = build_stage1_features(pd.read_csv(ROOT / "data/processed/aligned_daily_ohlcv_5y.csv"))
    scorer, dates = Scorer(features), pd.to_datetime(features["Date"])
    rows = []
    for i in range(args.n):
        noise = pd.Series(np.random.default_rng(i).normal(size=len(dates)), index=dates.values)
        rows.append(summarise(scorer.score(f"noise{i}", noise)).reset_index())
    null = pd.concat(rows, ignore_index=True)
    null.to_csv(ROOT / "results/stage2/noise_null.csv", index=False)
    print(null[["auc_gain", "importance"]].describe().round(4))
    for gain in (0, 0.005, 0.01):
        share = ((null.years_helped >= 3) & (null.auc_gain >= gain)).mean()
        print(f"noise passing years>=3 and gain>={gain}: {share:.0%}")
