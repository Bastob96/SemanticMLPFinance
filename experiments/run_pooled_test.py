"""Pooled test of selected factor sets: all four sealed Q4 quarters together (src/stage2/pooled_test.py).

Usage (no API key needed; use the same Python as the runs, scikit-learn 1.7.2):
    python experiments/run_pooled_test.py                       # every run with kept factors in the default list
    python experiments/run_pooled_test.py full_B=results/stage2/runs_v2/<run> batch_1=results/stage2/runs_v2/<run>
    python experiments/run_pooled_test.py --draws 50            # fewer random sets (faster, rougher p-values)

Writes results/stage2/pooled/summary.csv (one row per run and model) and
results/stage2/pooled/noise_k<k>.csv (the random-number sets, reused next time).
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
from src.stage2.formula import wide_prices  # noqa: E402
from src.stage2.pooled_test import (MODELS, bootstrap, factor_values, noise_gains, p_value,  # noqa: E402
                                    pooled_gain, q4_predictions)

warnings.filterwarnings("ignore")
DATA = ROOT / "data/processed/aligned_daily_ohlcv_5y.csv"
R = ROOT / "results/stage2/runs_v2"
OUT = ROOT / "results/stage2/pooled"

# The runs of the ablation study (results/stage2/ablation/summary.csv), plus every batch-mode run found.
DEFAULT = {
    "one_at_a_time_A": ROOT / "results/stage2/runs/20261015_1635_claude",
    "full_B": R / "20261021_1110_replay_20261015_1635_claude",
    "full_C": R / "20261021_1105_claude",
    "C_repeat2": R / "abl_claude_repeat2",
    "C_repeat3": R / "abl_claude_repeat3",
    "no_feedback": R / "abl_claude_nofeedback",
    "haiku": R / "abl_claude_haiku",
    "rounds_16": R / "abl_claude_16rounds",
    "B_no_pairs": R / "abl_B_nopairs",
    "template_0": R / "abl_template_seed0",
    "template_1": R / "abl_template_seed1",
    "template_2": R / "abl_template_seed2",
}


def runs_from_args(pairs):
    if pairs:
        return {k: Path(v) for k, v in (p.split("=", 1) for p in pairs)}
    runs = dict(DEFAULT)
    for d in sorted(R.glob("*batch*")):
        runs[d.name] = d
    return runs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="name=run_folder (default: the ablation runs and all batch runs)")
    ap.add_argument("--draws", type=int, default=200, help="random-number sets per factor count")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel processes for the random-number sets")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--name", default="summary.csv", help="file name of the summary (the random-number sets are shared)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA)
    features, prices = build_stage1_features(raw), wide_prices(raw)
    print("Q4 predictions with the 22 factors", flush=True)
    base = q4_predictions(features, {})
    rows, nulls = [], {}
    for name, folder in runs_from_args(args.runs).items():
        kept_file = folder / "kept_factors.json"
        if not kept_file.exists():
            print(f"skip {name}: {kept_file} not found", flush=True)
            continue
        kept = json.loads(kept_file.read_text())
        if not kept:
            print(f"skip {name}: no factors kept", flush=True)
            continue
        k = len(kept)
        if k not in nulls:
            path = args.out / f"noise_k{k}.csv"
            old = pd.read_csv(path) if path.exists() else pd.DataFrame()
            if len(old) < args.draws:
                print(f"random-number sets with {k} factors: {args.draws} draws", flush=True)
                old = noise_gains(features, prices, base, k, draws=args.draws, n_jobs=args.jobs)
                old.to_csv(path, index=False)
            nulls[k] = old
        plus = q4_predictions(features, {f["name"]: factor_values(f["formula"], prices) for f in kept})
        gain, ci = pooled_gain(base, plus), bootstrap(base, plus, n=args.boot)
        for m in MODELS:
            null = nulls[k]
            rows.append({"run": name, "folder": str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else str(folder),
                         "model": m, "kept": k, **gain[m], **ci[m],
                         "noise_auc_gain_mean": null[f"{m}_auc_gain"].mean(),
                         "noise_auc_gain_95th": null[f"{m}_auc_gain"].quantile(0.95),
                         "p_auc": p_value(gain[m]["auc_gain"], null[f"{m}_auc_gain"]),
                         "noise_bal_gain_95th": null[f"{m}_bal_gain"].quantile(0.95),
                         "p_bal": p_value(gain[m]["bal_gain"], null[f"{m}_bal_gain"])})
            r = rows[-1]
            print(f"{name:24} {m:20} k={k} AUC {r['auc_22']:.3f} -> {r['auc_plus']:.3f} "
                  f"({r['auc_gain']:+.3f}, 95% {r['auc_low']:+.3f} to {r['auc_high']:+.3f}, p={r['p_auc']:.2f}) | "
                  f"bal. acc. {r['bal_gain']:+.3f} (p={r['p_bal']:.2f})", flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / args.name, index=False)
    print(f"\nWrote {args.out / args.name} ({len(summary)} rows)")
