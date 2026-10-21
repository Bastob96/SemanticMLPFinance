"""Validated loading of the frozen dashboard artifact bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "results/ui"


@dataclass(frozen=True)
class DashboardData:
    manifest: dict
    stage1: pd.DataFrame
    stage1_backtest_summary: pd.DataFrame
    stage1_backtest_daily: pd.DataFrame
    stage1_selections: list[dict]
    importance: pd.DataFrame
    stage2_scoreboard: pd.DataFrame
    stage2_scores_by_year: pd.DataFrame
    stage2_final_metrics: pd.DataFrame
    stage2_final_equity: pd.DataFrame
    noise_null: pd.DataFrame


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if frame.empty or missing:
        raise ValueError(f"{name} is empty or missing columns: {sorted(missing)}")


def _csv(path: Path, required: set[str], name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dashboard artifact: {path}")
    frame = pd.read_csv(path)
    _require(frame, required, name)
    return frame


def load_dashboard_data(bundle_dir: str | Path = DEFAULT_BUNDLE) -> DashboardData:
    """Load and validate the files used by the app; never train a model."""
    bundle = Path(bundle_dir)
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run: python experiments/export_ui_artifacts.py"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dashboard manifest schema")

    stage1 = _csv(
        bundle / manifest["stage1"]["comparison"],
        {"year", "split", "experiment", "model", "balanced_accuracy_mean", "auc_mean"},
        "Stage 1 comparison",
    )
    if set(stage1["year"]) != {2022, 2023, 2024, 2025}:
        raise ValueError("Stage 1 dashboard bundle must contain 2022-2025")
    if not set(stage1["split"]).issubset({"validation", "test"}):
        raise ValueError("Stage 1 contains an unexpected split")

    backtest_summary = _csv(
        bundle / manifest["stage1"]["backtest_summary"],
        {"year", "strategy", "cumulative_return", "max_drawdown", "sharpe"},
        "Stage 1 backtest summary",
    )
    backtest_daily = _csv(
        bundle / manifest["stage1"]["backtest_daily"],
        {"year", "Date", "target_date", "strategy_equity", "buy_hold_equity"},
        "Stage 1 backtest daily",
    )
    for column in ("Date", "target_date"):
        backtest_daily[column] = pd.to_datetime(backtest_daily[column], errors="raise")

    selection_path = bundle / manifest["stage1"]["backtest_selection"]
    selections = json.loads(selection_path.read_text())
    if {int(row["year"]) for row in selections} != {2022, 2023, 2024, 2025}:
        raise ValueError("Backtest selections must cover 2022-2025")

    importance_path = (bundle / manifest["stage1"]["importance"]).resolve()
    importance = _csv(
        importance_path,
        {"model", "factor", "mean_importance", "positive_years"},
        "Stage 1 permutation importance",
    )

    scoreboard = _csv(
        bundle / manifest["stage2"]["scoreboard"],
        {"proposer", "round", "name", "formula", "status", "auc_gain", "years_helped"},
        "Stage 2 scoreboard",
    )
    scores_by_year = _csv(
        bundle / manifest["stage2"]["scores_by_year"],
        {"proposer", "factor", "year", "auc_gain", "bal_acc_gain", "valid"},
        "Stage 2 per-year scores",
    )
    final_metrics = _csv(
        bundle / manifest["stage2"]["final_metrics"],
        {
            "proposer", "year", "features", "model", "n_features",
            "balanced_accuracy", "auc", "strategy_return", "buy_hold_return",
        },
        "Stage 2 final metrics",
    )
    final_equity = _csv(
        bundle / manifest["stage2"]["final_equity"],
        {"proposer", "year", "features", "model", "Date", "strategy_equity", "buy_hold_equity"},
        "Stage 2 final equity",
    )
    for column in ("Date", "target_date"):
        final_equity[column] = pd.to_datetime(final_equity[column], errors="raise")
    noise = _csv(
        bundle / manifest["stage2"]["noise_null"],
        {"factor", "auc_gain", "years_helped"},
        "Stage 2 noise calibration",
    )

    if set(scoreboard["proposer"]) != {"claude", "random"}:
        raise ValueError("Stage 2 must contain the pinned Claude and random-control runs")
    if not set(scoreboard["status"]).issubset({"kept", "rejected", "invalid"}):
        raise ValueError("Stage 2 contains an unexpected candidate status")

    return DashboardData(
        manifest=manifest,
        stage1=stage1,
        stage1_backtest_summary=backtest_summary,
        stage1_backtest_daily=backtest_daily,
        stage1_selections=selections,
        importance=importance,
        stage2_scoreboard=scoreboard,
        stage2_scores_by_year=scores_by_year,
        stage2_final_metrics=final_metrics,
        stage2_final_equity=final_equity,
        noise_null=noise,
    )
