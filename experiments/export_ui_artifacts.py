"""Build the small, frozen data bundle consumed by the Streamlit dashboard.

The exporter deliberately reads committed prediction and discovery artifacts.
It does not tune a model or select anything using Q4. Stage 1 backtests reuse
the validation-only selection rule, while Stage 2 final results reproduce the
one-time Q4 evaluation for the two pinned discovery runs.

Usage:
    python experiments/export_ui_artifacts.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtesting import backtest_best
from src.evaluation import comparison_table
from src.features import build_stage1_features
from src.stage2.final_test import run_final_test
from src.stage2.formula import wide_prices

YEARS = (2022, 2023, 2024, 2025)
DATA = ROOT / "data/processed/aligned_daily_ohlcv_5y.csv"
STAGE1_DIR = ROOT / "results/multi_year"
OUTPUT = ROOT / "results/ui"
RUNS = {
    "claude": ROOT / "results/stage2/runs/20261015_1635_claude",
    "random": ROOT / "results/stage2/runs/20261015_1645_random",
}


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def export_stage1(raw_prices: pd.DataFrame) -> dict[str, object]:
    comparisons, summaries, dailies, selections = [], [], [], []
    for year in YEARS:
        predictions_path = STAGE1_DIR / str(year) / "stage1_predictions.csv"
        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing Stage 1 predictions: {predictions_path}")
        predictions = pd.read_csv(predictions_path)
        comparisons.append(comparison_table(predictions).assign(year=year))
        selection, daily, summary = backtest_best(predictions, raw_prices)
        dailies.append(daily.assign(year=year))
        summaries.append(summary.assign(year=year))
        selections.append({"year": year, **selection})

    pd.concat(comparisons, ignore_index=True).to_csv(
        OUTPUT / "stage1_comparison.csv", index=False
    )
    pd.concat(summaries, ignore_index=True).to_csv(
        OUTPUT / "stage1_backtest_summary.csv", index=False
    )
    pd.concat(dailies, ignore_index=True).to_csv(
        OUTPUT / "stage1_backtest_daily.csv", index=False
    )
    (OUTPUT / "stage1_backtest_selection.json").write_text(
        json.dumps(selections, indent=2) + "\n"
    )
    return {
        "years": list(YEARS),
        "comparison": "stage1_comparison.csv",
        "backtest_summary": "stage1_backtest_summary.csv",
        "backtest_daily": "stage1_backtest_daily.csv",
        "backtest_selection": "stage1_backtest_selection.json",
        "importance": "../multi_year/permutation_importance_summary.csv",
    }


def export_stage2(raw_prices: pd.DataFrame) -> dict[str, object]:
    features = build_stage1_features(raw_prices)
    wide = wide_prices(raw_prices)
    scoreboards, by_year, final_metrics, final_equity = [], [], [], []

    for proposer, run_dir in RUNS.items():
        run_id = run_dir.name
        kept = json.loads((run_dir / "kept_factors.json").read_text())
        scoreboards.append(
            pd.read_csv(run_dir / "scoreboard.csv").assign(
                proposer=proposer, run_id=run_id
            )
        )
        by_year.append(
            pd.read_csv(run_dir / "scores_by_year.csv").assign(
                proposer=proposer, run_id=run_id
            )
        )
        metrics, equity = run_final_test(features, wide, raw_prices, kept)
        final_metrics.append(metrics.assign(proposer=proposer, run_id=run_id))
        final_equity.append(equity.assign(proposer=proposer, run_id=run_id))

    pd.concat(scoreboards, ignore_index=True).to_csv(
        OUTPUT / "stage2_scoreboard.csv", index=False
    )
    pd.concat(by_year, ignore_index=True).to_csv(
        OUTPUT / "stage2_scores_by_year.csv", index=False
    )
    pd.concat(final_metrics, ignore_index=True).to_csv(
        OUTPUT / "stage2_final_metrics.csv", index=False
    )
    pd.concat(final_equity, ignore_index=True).to_csv(
        OUTPUT / "stage2_final_equity.csv", index=False
    )
    pd.read_csv(ROOT / "results/stage2/noise_null.csv").to_csv(
        OUTPUT / "stage2_noise_null.csv", index=False
    )

    return {
        "scoreboard": "stage2_scoreboard.csv",
        "scores_by_year": "stage2_scores_by_year.csv",
        "final_metrics": "stage2_final_metrics.csv",
        "final_equity": "stage2_final_equity.csv",
        "noise_null": "stage2_noise_null.csv",
        "runs": {key: path.name for key, path in RUNS.items()},
        "selection_rule": {
            "minimum_years_helped": 3,
            "minimum_mean_auc_gain": 0.01,
            "maximum_absolute_correlation": 0.9,
            "selection_split": "Q3 validation only",
        },
        "test_status": "Frozen one-time Q4 reproduction; not used for tuning",
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_prices = pd.read_csv(DATA)
    manifest = {
        "schema_version": 1,
        "source_commit": _source_commit(),
        "canonical_data": str(DATA.relative_to(ROOT)),
        "canonical_data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "export_environment": {
            package: version(package)
            for package in ("numpy", "pandas", "scikit-learn")
        },
        "prediction_target": "AAPL next-trading-day direction",
        "stage1": export_stage1(raw_prices),
        "stage2": export_stage2(raw_prices),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Dashboard artifacts written to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
