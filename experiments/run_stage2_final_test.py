"""Run the one-time Q4 test for the factors kept by a loop run.

Usage: python experiments/run_stage2_final_test.py --run results/stage2/runs/<run>
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import build_stage1_features  # noqa: E402
from src.stage2.final_test import run_final_test  # noqa: E402
from src.stage2.formula import wide_prices  # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "data/processed/aligned_daily_ohlcv_5y.csv"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    kept = json.loads((args.run / "kept_factors.json").read_text())
    if not kept:
        sys.exit("No kept factors in this run.")
    raw = pd.read_csv(DATA)
    metrics, equity = run_final_test(build_stage1_features(raw), wide_prices(raw), raw, kept)
    metrics.to_csv(args.run / "final_test_metrics.csv", index=False)
    equity.to_csv(args.run / "final_test_equity.csv", index=False)
    print(f"{len(kept)} kept factors:", ", ".join(k["name"] for k in kept))
    print(metrics.pivot_table(index=["model", "features"], columns="year",
                              values=["balanced_accuracy", "strategy_return"]).round(3).to_string())
