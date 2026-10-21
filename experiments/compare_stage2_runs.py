"""Compare Stage 2 runs side by side (Liu, 20 Oct: show the new search next to the single-factor one).

For each run:
  - how many factors were tried and how many were selected
  - validation (Q3): the selected set added together on top of the 22 factors (average gain, years helped)
  - the one-time Q4 test: the 22 factors versus the 22 plus the selected set, for both models
    (ROC AUC, balanced accuracy, long/flat strategy return; buy-and-hold for reference)
Nothing here is used to choose factors: the runs are fixed before this script is run.

Usage:
    python experiments/compare_stage2_runs.py          # A = 15 Oct run, B = latest replay, C = latest v2 Claude run
    python experiments/compare_stage2_runs.py A=results/stage2/runs/20261015_1635_claude \
        B=results/stage2/runs_v2/<replay run> C=results/stage2/runs_v2/<claude run>
Writes results/stage2/comparison/summary.csv and by_year.csv.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import build_stage1_features  # noqa: E402
from src.stage2.final_test import run_final_test  # noqa: E402
from src.stage2.formula import evaluate, wide_prices  # noqa: E402
from src.stage2.scoring import Scorer  # noqa: E402
from src.stage2.selection import summarise_result  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "results/stage2/comparison"

if __name__ == "__main__":
    runs = [a.split("=", 1) for a in sys.argv[1:]]
    if not runs:
        v2 = sorted((ROOT / "results/stage2/runs_v2").glob("*"))
        replay = [r for r in v2 if "_replay_" in r.name]
        claude = [r for r in v2 if r.name.endswith("_claude") and "_replay_" not in r.name]
        runs = [("A", ROOT / "results/stage2/runs/20261015_1635_claude")]
        runs += [("B", replay[-1])] if replay else []
        runs += [("C", claude[-1])] if claude else []
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "data/processed/aligned_daily_ohlcv_5y.csv")
    features, prices = build_stage1_features(raw), wide_prices(raw)
    scorer = Scorer(features)
    summary, by_year = [], []
    for label, run in runs:
        run = Path(run)
        board = pd.read_csv(run / "scoreboard.csv")
        kept = json.loads((run / "kept_factors.json").read_text())
        row = {"run": label, "folder": run.name, "tried": len(board),
               "invalid": int((board.status == "invalid").sum()), "selected": len(kept),
               "selected_names": " | ".join(k["name"] for k in kept)}
        if kept:
            values = {k["name"]: evaluate(k["formula"], prices) for k in kept}
            val = summarise_result(scorer.gains(values, {}))
            row.update(val_gain=val["gain"], val_years=val["years"], val_lr=val["lr"], val_rf=val["rf"])
            metrics, _ = run_final_test(features, prices, raw, kept)
            metrics.insert(0, "run", label)
            by_year.append(metrics)
            for model, m in metrics.groupby("model"):
                short = "lr" if model == "logistic_regression" else "rf"
                base, plus = m[m.features == "stage1_22"].set_index("year"), m[m.features == "stage1_plus_llm"].set_index("year")
                row[f"q4_auc_change_{short}"] = (plus.auc - base.auc).mean()
                row[f"q4_auc_years_better_{short}"] = int((plus.auc > base.auc).sum())
                row[f"q4_bal_acc_change_{short}"] = (plus.balanced_accuracy - base.balanced_accuracy).mean()
                row[f"q4_return_22_{short}"] = base.strategy_return.mean()
                row[f"q4_return_plus_{short}"] = plus.strategy_return.mean()
            row["q4_buy_hold"] = metrics.buy_hold_return.groupby(metrics.year).first().mean()
        summary.append(row)
        print(f"{label}: {row['selected']} selected of {row['tried']} tried ({run.name})", flush=True)

    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary.csv", index=False)
    if by_year:
        pd.concat(by_year, ignore_index=True).to_csv(OUT / "by_year.csv", index=False)

    show = summary.set_index("run").drop(columns=["folder", "selected_names"]).T
    print("\n" + show.to_string(float_format=lambda x: f"{x:+.4f}"))
    print("\nSelected factors:")
    for r in summary.itertuples():
        print(f"  {r.run}: {r.selected_names or '(none)'}")
