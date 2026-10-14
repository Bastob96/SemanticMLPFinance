"""Final, one-time test of the selected factors on the locked Q4 test quarters.

For each year: train on Q1-Q2 with (a) the 22 Stage 1 factors and (b) the 22 plus
the kept LLM factors, predict Q4, and compare balanced accuracy, ROC AUC and a
long/flat backtest (src.backtesting.simulate_long_flat) against buy-and-hold.
Random Forest predictions are the average probability of seeds 0, 1 and 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.backtesting.report import performance_metrics
from src.backtesting.simulate import simulate_long_flat
from src.data import quarterly_split
from src.models import fit_logistic_regression, fit_random_forest

from .formula import evaluate
from .scoring import BASE, LR_SEED, RF_SEEDS, TARGET, YEARS


def _clip(train, test, cols):
    train, test = train.copy(), test.copy()
    for c in cols:
        lo, hi = train[c].quantile([0.01, 0.99])
        train[c], test[c] = train[c].clip(lo, hi), test[c].clip(lo, hi)
    return train, test


def run_final_test(features, prices_wide, raw_prices, kept, years=YEARS):
    data = features.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    names = [k["name"] for k in kept]
    for k in kept:
        data = data.merge(evaluate(k["formula"], prices_wide).rename(k["name"]),
                          left_on="Date", right_index=True, how="left")
    sets = {"stage1_22": BASE, "stage1_plus_llm": BASE + names}
    metrics, equity = [], []
    for year in years:
        s = quarterly_split(data, year)
        for set_name, cols in sets.items():
            train, test = _clip(s["train"], s["test"], [c for c in cols if c in names])
            probs = {
                "logistic_regression": fit_logistic_regression(train[cols], train[TARGET], LR_SEED)
                .predict_proba(test[cols])[:, 1],
                "random_forest": np.mean([fit_random_forest(train[cols], train[TARGET], seed)
                                          .predict_proba(test[cols])[:, 1] for seed in RF_SEEDS], axis=0),
            }
            for model, p in probs.items():
                pred = (p >= 0.5).astype(int)
                frame = pd.DataFrame({"Date": test["Date"], "target_date": test["target_date"],
                                      "actual": test[TARGET], "predicted": pred, "probability_up": p,
                                      # labels only, to satisfy the Stage 1 prediction-file checks
                                      "split": "test", "experiment": "C", "model": model,
                                      "seed": LR_SEED if model == "logistic_regression" else 0})
                daily = simulate_long_flat(frame, raw_prices)
                strat = performance_metrics(daily.strategy_return)
                hold = performance_metrics(daily.aapl_return)
                metrics.append({"year": year, "features": set_name, "model": model, "n_features": len(cols),
                                "balanced_accuracy": balanced_accuracy_score(test[TARGET], pred),
                                "auc": roc_auc_score(test[TARGET], p),
                                "strategy_return": strat["cumulative_return"],
                                "buy_hold_return": hold["cumulative_return"],
                                "days_long": int(pred.sum()), "days": len(pred)})
                equity.append(daily.assign(year=year, features=set_name, model=model))
    return pd.DataFrame(metrics), pd.concat(equity, ignore_index=True)
