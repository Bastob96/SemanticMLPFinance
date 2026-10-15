# Stage 1 — Evaluation & Backtesting: Supervisor Summary
**PG-S2-36 · Abu Wakkas Bastob · Prepared 16 September 2026, from a fresh end-to-end run**

---

## What was built

The Stage 1 pipeline (data → features → models) was already built by the team. The two pieces that were missing — an honest evaluation framework and a backtesting check — are now complete, tested, merged into `main`, and independently re-verified tonight on a freshly rebuilt environment. Not just re-checked from old logs: actually re-run from scratch, start to finish.

## Results — Logistic Regression vs Random Forest vs naive baselines (test split, year 2025 Q4)

| Model | Accuracy | Balanced Accuracy | AUC |
|---|---|---|---|
| Majority-class baseline | 0.556 | 0.500 | 0.500 |
| Previous-direction baseline | 0.540 | 0.532 | 0.532 |
| Logistic Regression — A (AAPL only) | 0.413 | 0.461 | 0.558 |
| Random Forest — A (mean of 3 seeds) | 0.460 | 0.504 | 0.572 |
| Random Forest — B (+QQQ) | 0.413 | 0.449 | 0.554 |
| Random Forest — C (+MSFT, +NVDA) | 0.402 | 0.438 | 0.470 |

**Honest finding:** no model configuration beat the naive previous-direction baseline on the untouched test set. Adding more tickers (B, C) made results worse, not better, than AAPL alone (A). This matches what was found in the original committed results and is reported as-is, not adjusted to look better.

## Backtest — does it make money?

Strategy selected on **validation** performance only (Logistic Regression, experiment B, seed 42) — never on test performance, so this isn't cherry-picked:

| Strategy | Return (Q4 2025) | Max Drawdown | Sharpe |
|---|---|---|---|
| Model-driven long/flat | **-0.72%** | -2.20% | -0.53 |
| Buy-and-hold AAPL | **+6.53%** | -5.32% | 1.53 |

The strategy loses money and underperforms simply holding AAPL over the same window. Reported honestly, not softened.

## A note on rigor (worth saying out loud)

While re-running tonight, a genuine environment issue surfaced and was resolved properly: a newer numpy version caused a low-level crash in pandas' datetime handling, unrelated to the project's own code. Diagnosed via Python's fault handler down to the exact line, fixed by pinning compatible library versions, and re-verified. Separately, it's documented that Random Forest results shift slightly depending on the exact scikit-learn version installed — proven by reproducing the original archived numbers exactly on scikit-learn 1.7.1, and by tonight's run on 1.9.1 landing close but not identical, as expected.

**All 22 automated tests pass** on tonight's fresh run (0 failures), including tests that independently recompute results by hand and cross-check them against the pipeline's own output.

## What's next

- Taiyu's parallel regression experiment (predicting next-day *return* rather than direction, using HistGradientBoosting) is a separate, well-executed track — not yet folded into this evaluation framework, pending a team decision on scope
- A small methodological difference between Kevin's production split (excludes quarter-boundary labels) and Taiyu's notebook split (doesn't) is worth a short team conversation, not a rewrite
- Stage 2 LLM-based factor generation work is already underway (Leyao)
- Given Stage 1's naive baselines aren't beaten yet, the team's next real opportunity is investigating richer features or a different modeling approach — not just re-running what exists
