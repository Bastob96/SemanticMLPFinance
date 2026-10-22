"""Deterministic data context and round feedback passed to Claude.

Only training-period statistics and validation results are used here. The
final test rows are never passed to these functions, and ``assert_no_test_dates``
checks every prompt payload before it is sent.
"""

import json
import math

import numpy as np
import pandas as pd

import config
from baseline_factors import BASELINE_FACTOR_DEFINITIONS
from factor_engine import LANGUAGE_REFERENCE


def round_floats(value, digits=5):
    """Round floats to ``digits`` significant figures so prompts stay compact and readable."""
    if isinstance(value, dict):
        return {k: round_floats(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(v, digits) for v in value]
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            return None
        return float(f"{float(value):.{digits}g}")
    if isinstance(value, np.integer):
        return int(value)
    return value


def training_statistics(dev_wide, target, train_index):
    """Descriptive statistics from the training period only (including its warm-up rows)."""
    rows = dev_wide.loc[dev_wide.index <= train_index.max()]
    per_asset = {}
    for ticker in config.TICKERS:
        p = ticker.lower()
        ret = rows[f"{p}_ret1"].dropna()
        per_asset[ticker] = {
            "mean_daily_log_return": ret.mean(),
            "std_daily_log_return": ret.std(),
            "annualised_volatility": ret.std() * np.sqrt(252),
            "skewness": ret.skew(),
            "lag1_autocorrelation": ret.autocorr(1),
            "largest_daily_drop": ret.min(),
            "largest_daily_gain": ret.max(),
            "close_min": rows[f"{p}_close"].min(),
            "close_max": rows[f"{p}_close"].max(),
            "mean_intraday_range_pct": ((rows[f"{p}_high"] - rows[f"{p}_low"]) / rows[f"{p}_close"]).mean() * 100,
            "mean_volume": rows[f"{p}_volume"].mean(),
            "std_daily_log_volume_change": rows[f"{p}_logvol"].diff().std(),
        }
    returns = rows[[f"{t.lower()}_ret1" for t in config.TICKERS]].dropna()
    returns.columns = list(config.TICKERS)
    beta = returns["AAPL"].cov(returns["QQQ"]) / returns["QQQ"].var()
    y = target.loc[train_index]
    return {
        "period": {"first_date": str(rows.index.min().date()), "last_date": str(rows.index.max().date()),
                   "rows": int(len(rows))},
        "per_asset": per_asset,
        "daily_return_correlation": returns.corr().round(3).to_dict(),
        "aapl_beta_to_qqq": beta,
        "target_on_training_rows": {"mean": y.mean(), "std": y.std(), "share_positive": (y > 0).mean(),
                                    "mean_absolute": y.abs().mean()},
    }


def selection_policy_text():
    return (f"Primary metric: validation MAE of the next-day log return (lower is better). A change to the "
            f"factor set is accepted only if it lowers validation MAE by at least "
            f"{config.MIN_RELATIVE_MAE_IMPROVEMENT:.1%} relative to the current set and does not raise "
            f"validation RMSE. A candidate may be ADDED (while fewer than {config.MAX_RETAINED_LLM_FACTORS} "
            f"LLM factors are retained) or may REPLACE one retained LLM factor. At most "
            f"{config.MAX_ACCEPTED_PER_ROUND} changes are accepted per round. The 10 baseline factors and the "
            f"HGBR settings never change. Candidates must pass validity checks first: allowed syntax, "
            f"lookback <= {config.MAX_LOOKBACK} days, no future data, <= {config.MAX_MISSING_FRACTION:.0%} missing, "
            f"not near-constant, |Spearman| < {config.REDUNDANCY_THRESHOLD} with existing factors on training rows.")


def build_data_context(dev_wide, target, splits, baseline_metrics, naive_metrics, baseline_importance):
    context = {
        "task": {
            "target": config.TARGET_DEFINITION,
            "prediction_time": "After the close of trading day t; predict AAPL's log return from t to t+1.",
            "model": "Fixed sklearn HistGradientBoostingRegressor (Phase 1 settings) on baseline + selected factors.",
            "primary_metric": "Validation MAE of next-day log return",
            "secondary_diagnostics": ["RMSE", "R2", "directional accuracy", "Spearman IC",
                                      "permutation importance"],
        },
        "assets": {"AAPL": "Apple (prediction target)", "MSFT": "Microsoft (peer tech stock)",
                   "NVDA": "NVIDIA (peer tech stock)", "QQQ": "Nasdaq-100 ETF (tech market proxy)"},
        "data": {"frequency": "daily, common trading dates of all four assets, year 2025",
                 "price_adjustment": config.PRICE_ADJUSTMENT_NOTE},
        "factor_language": LANGUAGE_REFERENCE,
        "baseline_factors": [{k: v for k, v in spec.items() if k != "lookback"}
                             for spec in BASELINE_FACTOR_DEFINITIONS],
        "periods": {"training": {"first_feature_date": str(splits.train.min().date()),
                                 "last_feature_date": str(splits.train.max().date()),
                                 "rows": int(len(splits.train))},
                    "validation": {"first_feature_date": str(splits.validation.min().date()),
                                   "last_feature_date": str(splits.validation.max().date()),
                                   "rows": int(len(splits.validation))}},
        "training_statistics": training_statistics(dev_wide, target, splits.train),
        "baseline_validation_metrics": baseline_metrics,
        "naive_zero_return_validation_metrics": naive_metrics,
        "baseline_permutation_importance_validation": baseline_importance,
        "selection_policy": selection_policy_text(),
    }
    return round_floats(context)


def build_round_feedback(record):
    """Compact, LLM-facing summary of one round, built only from computed results."""
    candidates = []
    for item in record["candidates"]:
        entry = {
            "name": item["name"],
            "expression": item.get("expression"),
            "category": item.get("category"),
            "relation_to_previous": item.get("relation_to_previous"),
            "validation_check": item["validation"]["status"],
        }
        if item["validation"]["status"] == "invalid":
            entry["rejection_code"] = item["validation"]["code"]
            entry["rejection_reason"] = item["validation"]["message"]
        else:
            entry.update({
                "mae_baseline_plus_candidate": item["mae_baseline_plus"],
                "relative_mae_improvement_vs_baseline": item["rel_improvement_vs_baseline"],
                "best_move_vs_current_set": item.get("best_move"),
                "relative_mae_improvement_of_best_move": item.get("best_move_rel_improvement"),
                "rmse_of_best_move": item.get("best_move_rmse"),
                "decision": item["decision"],
                "decision_reason": item["decision_reason"],
                "max_abs_spearman_with_existing_train": item["validation"].get("max_abs_spearman_train"),
                "most_correlated_with": item["validation"].get("most_correlated_with"),
                "spearman_with_target_validation": item.get("target_spearman_validation"),
            })
        candidates.append(entry)
    feedback = {
        "round": record["round"],
        "candidates_requested": record["candidates_requested"],
        "candidates_received": record["candidates_received"],
        "llm_status": record["llm_status"],
        "candidates": candidates,
        "changes_accepted": record["changes"],
        "retained_llm_factors_before": [f["name"] for f in record["retained_before"]],
        "retained_llm_factors_after": [{"name": f["name"], "expression": f["expression"],
                                        "added_in_round": f["round_added"]} for f in record["retained_after"]],
        "validation_metrics": {"baseline": record["baseline_metrics"],
                               "current_set_before_round": record["metrics_before"],
                               "current_set_after_round": record["metrics_after"],
                               "all_valid_candidates_added": record.get("round_evaluated_metrics")},
        "permutation_importance_current_set_validation": record.get("importance_after"),
        "high_correlation_pairs_train": record.get("high_correlation_pairs"),
        "observations": record["observations"],
        "capacity": {"retained_llm_factors": len(record["retained_after"]),
                     "max_retained_llm_factors": config.MAX_RETAINED_LLM_FACTORS,
                     "additions_possible": len(record["retained_after"]) < config.MAX_RETAINED_LLM_FACTORS,
                     "replacement_allowed": config.ALLOW_REPLACEMENT},
    }
    return round_floats(feedback)


def observations(record):
    """Rule-based, factual notes derived from the round's numbers (no LLM interpretation)."""
    notes = []
    invalid = [c for c in record["candidates"] if c["validation"]["status"] == "invalid"]
    if invalid:
        codes = sorted({c["validation"]["code"] for c in invalid})
        notes.append(f"{len(invalid)} of {len(record['candidates'])} candidates failed validity checks ({', '.join(codes)}).")
    valid = [c for c in record["candidates"] if c["validation"]["status"] == "valid"]
    better = [c["name"] for c in valid if c["rel_improvement_vs_baseline"] > 0]
    if valid:
        notes.append(f"{len(better)} of {len(valid)} valid candidates lowered validation MAE when added alone to the baseline.")
    redundant_like = [c["name"] for c in valid if (c["validation"].get("max_abs_spearman_train") or 0) >= 0.8]
    if redundant_like:
        notes.append(f"Highly correlated (|rho| >= 0.8) with an existing factor: {', '.join(redundant_like)}.")
    if record["changes"]:
        notes.append("Accepted changes: " + "; ".join(
            f"{c['type']} {c['candidate']}" + (f" (replacing {c['replaced']})" if c.get("replaced") else "")
            for c in record["changes"]) + ".")
    else:
        notes.append("No change met the acceptance rule; the factor set is unchanged.")
    importance = record.get("importance_after") or []
    negative = [row["factor"] for row in importance if row["importance_mean"] < 0]
    if negative:
        notes.append("Negative validation permutation importance (shuffling lowered MAE): " + ", ".join(negative) + ".")
    return notes


def tried_expressions(round_records):
    return [{"round": r["round"], "name": c["name"], "expression": c.get("expression"),
             "outcome": (c["validation"]["code"] if c["validation"]["status"] == "invalid" else c["decision"])}
            for r in round_records for c in r["candidates"]]


def assert_no_test_dates(payload, test_index):
    """Fail loudly if any final-test feature date appears in an LLM prompt payload."""
    text = json.dumps(payload, default=str)
    leaked = [str(d.date()) for d in test_index if str(d.date()) in text]
    if leaked:
        raise AssertionError(f"Final test dates found in LLM prompt: {leaked[:5]}")
