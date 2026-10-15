"""Run the quarterly Stage 2 comparison with validation-first model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtesting import backtest_best
from src.data import verify_canonical_data
from src.evaluation import comparison_table, evaluate_predictions
from src.stage2 import (
    PRIMARY_EXPERIMENTS,
    build_experiment_specs,
    build_feature_explanation,
    build_predictions,
    incremental_comparison,
    load_integrated_splits,
    select_primary_configuration,
)


DEFAULT_DATA = ROOT / "data/processed/aligned_daily_ohlcv_5y.csv"
DEFAULT_STAGE2 = ROOT / "Stage2_Factor_Files/data/processed/quarterly"
DEFAULT_FEATURE_SETS = ROOT / "Stage2_Factor_Files/configs/feature_sets.json"
DEFAULT_MANIFEST = ROOT / "configs/canonical_data_2025.json"
DEFAULT_OUTPUT = ROOT / "results/stage2_2025"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes(data: Path, stage2: Path, feature_sets: Path) -> dict[str, str]:
    paths = [
        data,
        feature_sets,
        stage2 / "stage2_train_2025.csv",
        stage2 / "stage2_validation_2025.csv",
    ]
    return {str(path.resolve().relative_to(ROOT.resolve())): _sha256(path) for path in paths}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.12g")


def _selection_identity(selection: dict) -> tuple:
    return selection["experiment"], selection["model"], selection["seed"]


def _format_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("" if np.isnan(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def _direct_c_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    return comparison.loc[
        comparison["experiment"].isin(["C_stage1", "C_stage2"]),
        ["split", "experiment", "model", "accuracy_mean", "balanced_accuracy_mean",
         "auc_mean", "f1_mean", "balanced_accuracy_std"],
    ].sort_values(["split", "model", "experiment"])


def _write_validation_report(
    output: Path,
    comparison: pd.DataFrame,
    increments: pd.DataFrame,
    explanation: pd.DataFrame,
    selection: dict,
) -> None:
    primary = comparison.loc[comparison.experiment.isin(["baseline", *PRIMARY_EXPERIMENTS])]
    result_columns = [
        "experiment", "model", "balanced_accuracy_mean", "auc_mean", "f1_mean",
        "balanced_accuracy_std",
    ]
    delta_columns = ["ablation", "model", "delta_balanced_accuracy", "delta_auc", "delta_f1"]
    factor_columns = [
        "feature", "logistic_standardized_coefficient", "rf_importance_mean", "rf_importance_std",
    ]
    llm = explanation.loc[explanation.source.eq("llm_candidate")]
    text = f"""# Stage 2 validation — 2025 quarterly profile

Q1-Q2 was used for training and Q3 for model selection. Q4 was not loaded by
this phase. The only controlled change is the addition of the LLM factors.

## Validation results

{_format_table(primary, result_columns)}

## Effect of adding the factors

{_format_table(increments, delta_columns)}

## LLM-factor explanation fitted on training data only

{_format_table(llm, factor_columns)}

Logistic coefficients are directional effects on UP log-odds after
standardisation. Random Forest importance is non-directional and is averaged
over seeds 0, 1 and 2. Neither measure proves causality.

## Locked selection

- Experiment: `{selection['experiment']}`
- Model: `{selection['model']}`
- Seed: `{selection['seed']}`
- Validation balanced accuracy: `{selection['selection_score']:.6f}`
- Test status: **locked and not evaluated**
"""
    (output / "STAGE2_VALIDATION_2025.md").write_text(text, encoding="utf-8")


def _write_final_report(
    output: Path,
    comparison: pd.DataFrame,
    increments: pd.DataFrame,
    selection: dict,
    backtest_summary: pd.DataFrame,
) -> None:
    direct = _direct_c_comparison(comparison)
    direct_columns = [
        "split", "experiment", "model", "accuracy_mean", "balanced_accuracy_mean",
        "auc_mean", "f1_mean", "balanced_accuracy_std",
    ]
    delta_c = increments.loc[increments.ablation.eq("C")]
    delta_columns = ["split", "model", "delta_balanced_accuracy", "delta_auc", "delta_f1"]
    backtest_columns = ["strategy", "cumulative_return", "max_drawdown", "sharpe"]
    test = comparison.loc[comparison.split.eq("test")]
    previous = float(test.loc[
        test.experiment.eq("baseline") & test.model.eq("previous_direction"),
        "balanced_accuracy_mean",
    ].iloc[0])
    best_augmented = float(test.loc[
        test.experiment.isin(["B_stage2", "C_stage2"]), "balanced_accuracy_mean"
    ].max())
    conclusion = (
        "The factors did not establish a reliable improvement: both C models "
        "lost balanced accuracy on validation, and the best augmented Q4 result "
        f"across B/C ({best_augmented:.4f}) remained below the previous-direction baseline "
        f"({previous:.4f})."
    )
    text = f"""# Stage 2 final results — 2025 quarterly profile

The feature sets and selection rule were locked using Q3 before Q4 was
evaluated. Q4 is reported once and is not a tuning input.

## Supervisor-requested comparison: 22 features versus 22 + 8 factors

`C_stage1` contains the existing 22 conventional features. `C_stage2` contains
the same 22 features plus all eight LLM-generated candidates.

{_format_table(direct, direct_columns)}

## Increment caused by the eight factors

{_format_table(delta_c, delta_columns)}

## Validation-selected backtest

- Experiment: `{selection['experiment']}`
- Model: `{selection['model']}`
- Seed: `{selection['seed']}`
- Validation balanced accuracy: `{selection['selection_score']:.6f}`

{_format_table(backtest_summary, backtest_columns)}

The backtest is an optimistic frictionless same-close to next-close diagnostic,
not an executable trading claim.

## Conclusion

{conclusion}
"""
    (output / "STAGE2_RESULTS_2025.md").write_text(text, encoding="utf-8")


def run_validation(
    data: Path,
    stage2: Path,
    feature_sets: Path,
    manifest: Path,
    output: Path,
) -> dict:
    canonical = verify_canonical_data(ROOT, manifest)
    splits, stage2_sets = load_integrated_splits(
        data, stage2, feature_sets, include_test=False
    )
    specs = build_experiment_specs(stage2_sets)
    predictions = build_predictions(splits, specs, ("validation",))
    detailed = evaluate_predictions(predictions)
    comparison = comparison_table(predictions)
    increments = incremental_comparison(comparison)
    explanation = build_feature_explanation(
        splits["train"], specs["C_stage2"], stage2_sets["C"]
    )
    selection = select_primary_configuration(comparison)

    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "stage2_validation_predictions.csv"
    _write_csv(predictions, predictions_path)
    _write_csv(detailed, output / "stage2_validation_metrics_detailed.csv")
    _write_csv(comparison, output / "stage2_validation_comparison.csv")
    _write_csv(increments, output / "stage2_validation_incremental.csv")
    _write_csv(explanation, output / "stage2_feature_explanation.csv")

    lock = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "year": 2025,
        "profile": "quarterly",
        "split_policy": "Q1-Q2 train, Q3 validation, Q4 test; boundary labels purged",
        "selection": selection,
        "primary_experiments": specs,
        "model_configuration": {
            "logistic_regression": {"seed": 42, "C": 1.0, "max_iter": 2000, "threshold": 0.5},
            "random_forest": {"seeds": [0, 1, 2], "n_estimators": 300,
                              "max_depth": 5, "min_samples_leaf": 5,
                              "class_weight": "balanced", "threshold": 0.5},
        },
        "canonical_data": canonical,
        "source_sha256": _source_hashes(data, stage2, feature_sets),
        "validation_predictions_sha256": _sha256(predictions_path),
        "test_status": "locked_not_evaluated",
    }
    (output / "stage2_selection_lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )
    run_info = {
        "phase": "validation", "test_evaluated": False,
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "python": platform.python_version(), "pandas": pd.__version__,
        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
    }
    (output / "stage2_validation_run_info.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )
    _write_validation_report(output, comparison, increments, explanation, selection)
    return lock


def run_final(
    data: Path,
    stage2: Path,
    feature_sets: Path,
    manifest: Path,
    output: Path,
    confirm_locked_test: bool,
) -> dict:
    if not confirm_locked_test:
        raise ValueError("Final testing requires --confirm-locked-test")
    lock_path = output / "stage2_selection_lock.json"
    if not lock_path.is_file():
        raise ValueError("Run the validation phase first; selection lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("test_status") != "locked_not_evaluated":
        raise ValueError("The locked test has already been evaluated")
    verify_canonical_data(ROOT, manifest)
    if lock.get("source_sha256") != _source_hashes(data, stage2, feature_sets):
        raise ValueError("Training or validation inputs changed after the lock")

    splits, stage2_sets = load_integrated_splits(
        data, stage2, feature_sets, include_test=True
    )
    specs = build_experiment_specs(stage2_sets)
    if specs != lock.get("primary_experiments"):
        raise ValueError("Experiment features differ from the locked specification")
    predictions = build_predictions(splits, specs, ("validation", "test"))
    detailed = evaluate_predictions(predictions)
    comparison = comparison_table(predictions)
    increments = incremental_comparison(comparison)
    selection = select_primary_configuration(comparison)
    if _selection_identity(selection) != _selection_identity(lock["selection"]):
        raise ValueError("Validation selection no longer matches the lock")

    _write_csv(predictions, output / "stage2_final_predictions.csv")
    _write_csv(detailed, output / "stage2_final_metrics_detailed.csv")
    _write_csv(comparison, output / "stage2_final_comparison.csv")
    _write_csv(increments, output / "stage2_final_incremental.csv")

    backtest_selection, daily, backtest_summary = backtest_best(predictions, data)
    if _selection_identity(backtest_selection) != _selection_identity(lock["selection"]):
        raise ValueError("Backtest selection differs from the lock")
    (output / "stage2_backtest_selection.json").write_text(
        json.dumps(backtest_selection, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(daily, output / "stage2_backtest_daily.csv")
    _write_csv(backtest_summary, output / "stage2_backtest_summary.csv")

    completed = datetime.now(timezone.utc).isoformat()
    final_info = {
        "phase": "final", "completed_at_utc": completed,
        "test_evaluated": True, "test_evaluation_count": 1,
        "selection_locked_before_test": True, "selection": lock["selection"],
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "python": platform.python_version(), "pandas": pd.__version__,
        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
    }
    (output / "stage2_final_run_info.json").write_text(
        json.dumps(final_info, indent=2) + "\n", encoding="utf-8"
    )
    lock["test_status"] = "evaluated_once_after_lock"
    lock["final_completed_at_utc"] = completed
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    _write_final_report(output, comparison, increments, selection, backtest_summary)
    return final_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["validation", "final"], default="validation")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--feature-sets", type=Path, default=DEFAULT_FEATURE_SETS)
    parser.add_argument("--canonical-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confirm-locked-test", action="store_true")
    args = parser.parse_args()

    if args.phase == "validation":
        lock = run_validation(
            args.data, args.stage2_dir, args.feature_sets,
            args.canonical_manifest, args.output,
        )
        print("Validation complete; Q4 was not evaluated.")
        print(json.dumps(lock["selection"], indent=2))
    else:
        info = run_final(
            args.data, args.stage2_dir, args.feature_sets,
            args.canonical_manifest, args.output, args.confirm_locked_test,
        )
        print("Final locked-test evaluation complete.")
        print(json.dumps(info["selection"], indent=2))


if __name__ == "__main__":
    main()
