# Stage 1 baseline results — 2025

## Experimental setup

- Target: AAPL direction on the next trading day
- Training: Q1-Q2, 121 labelled observations
- Validation: Q3, 63 labelled observations
- Untouched test: Q4, 63 labelled observations
- Logistic Regression seed: 42
- Random Forest seeds: 0, 1 and 2
- Threshold: 0.5

The final trading day of each partition is excluded when its label would depend
on a price from the following partition. Rolling features use earlier history,
including late-2024 data for the beginning of 2025, but those earlier dates are
not training observations.

## Untouched Q4 results

| Experiment | Model | Accuracy | Balanced accuracy | AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | Majority class | 0.556 | 0.500 | 0.500 | 0.556 | 1.000 | 0.714 |
| Baseline | Previous direction | 0.540 | 0.532 | 0.532 | 0.583 | 0.600 | 0.592 |
| A: AAPL | Logistic Regression | 0.413 | 0.461 | 0.558 | 0.250 | 0.029 | 0.051 |
| A: AAPL | Random Forest | 0.471 ± 0.009 | 0.513 ± 0.008 | 0.563 ± 0.026 | 0.607 ± 0.031 | 0.133 ± 0.016 | 0.219 ± 0.024 |
| B: AAPL+QQQ | Logistic Regression | 0.413 | 0.454 | 0.534 | 0.375 | 0.086 | 0.140 |
| B: AAPL+QQQ | Random Forest | 0.402 ± 0.009 | 0.438 ± 0.010 | 0.547 ± 0.008 | 0.376 ± 0.021 | 0.114 ± 0.000 | 0.175 ± 0.002 |
| C: all stocks | Logistic Regression | 0.381 | 0.411 | 0.478 | 0.357 | 0.143 | 0.204 |
| C: all stocks | Random Forest | 0.407 ± 0.018 | 0.443 ± 0.016 | 0.488 ± 0.013 | 0.386 ± 0.067 | 0.124 ± 0.044 | 0.187 ± 0.058 |

Random Forest entries are mean ± sample standard deviation across its three
seeds. Logistic Regression and baseline entries are single deterministic runs.

## Initial interpretation

The best ML configuration on Q4 was AAPL-only Random Forest, with mean balanced
accuracy of 0.513 and AUC of 0.563. It did not outperform the previous-direction
baseline's balanced accuracy of 0.532 or the majority baseline's accuracy of
0.556. Adding QQQ, MSFT and NVDA did not improve this test-period performance.
This close-to-random outcome is acceptable for Stage 1 and should be reported
honestly rather than tuned against Q4.

## Implementation issues

1. Abu's shared evaluation module was not present in the latest pull. The CSV
   predictions are ready for his evaluator; current metrics use scikit-learn
   directly and are explicitly marked provisional.
2. The one-year split provides only 121 training labels, limiting model
   stability and making quarter-to-quarter regime changes influential.
3. Three boundary rows were deliberately excluded because their target prices
   came from a different partition. This reduced the supplied 122/64/64 dated
   rows to 121/63/63 labelled model rows.
4. No feature, parameter, or threshold tuning used Q4 results. The initial
   hyperparameters were fixed before the test run.
