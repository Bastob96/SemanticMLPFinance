# Abu — Stage 1 evaluation and backtesting, 2025

## Outcome

Completed and verified locally on `codex/abu-evaluation-backtesting`, based on `8b6574167a048a72a88791412e8c7193d6d8a737`. Nothing pushed or merged. The local checkout originally had clean `main` at `8bb8eb3`, seven commits behind the fetched remote; the new branch starts from current `origin/main`. Local `main` was not advanced.

The original pipeline passed, all 3 original tests passed, and the final full Stage 1 suite passes all 22 tests (19 added). Shared evaluation matches original fresh results within 1e-14. Protected files are unchanged against the fetched base. The only existing source-file edit removes the temporary metric function/import and replaces its four call sites with the shared evaluator.

Current classification results do not establish an ML advantage over the naive baselines on Q4. RF A has the best ML test balanced accuracy, 0.503571429, below previous-direction's 0.532142857. Majority-class accuracy is 0.555555556.

## Backtest and selection policy

Selection uses **validation balanced accuracy**, including the baselines and RF seed means. Ties resolve by experiment then model name. If RF wins, seed 0 is fixed in advance, not selected on test returns. Only the chosen configuration's test rows are simulated. This resolves the handoff's unspecified meaning of “best” without selecting a strategy on test performance.

Selected **Logistic Regression B**, seed 42: validation balanced accuracy **0.575757575758**. Its test balanced accuracy is 0.453571429. The descriptive best ML test configuration (RF A) is not the backtest selection.

Window: **2025-10-01 close through 2025-12-31 close**; **63 contiguous next-trading-day intervals**, long on **8** and flat on **55**.

| strategy | cumulative_return | max_drawdown | sharpe |
| --- | --- | --- | --- |
| long_flat | -0.007190339664 | -0.022000048618 | -0.529881728176 |
| buy_and_hold_aapl | 0.065271217935 | -0.053181439621 | 1.529809497858 |

Long/flat return: **-0.719033966%**. Buy-and-hold: **6.527121793%**. Strategy minus buy-and-hold: **-7.246155760 percentage points**. Thus the chosen strategy underperforms buy-and-hold over this window.

Assumptions: idealized execution at source-day close using that day's close-based features; earn the exact next-close return when predicted=1; zero transaction costs, slippage, interest and risk-free return; no shorts or leverage. **Same-close execution is optimistic and is not evidence of an executable trading strategy.** A more realistic next-open rule would measure a different return interval and is outside this simple diagnostic. Prices are the repository's existing Close series; no market data were downloaded or adjusted. Buy-and-hold uses exactly the same start and end closes. Maximum drawdown is reported as a negative return from the running peak, including initial wealth 1. Sharpe is sqrt(252) × mean daily return / sample SD, counting flat days; undefined Sharpe is NaN.

## Setup and exact feature lists

Target is AAPL next-trading-day direction, positive=1; ties=0. Year 2025, training Q1–Q2, validation Q3, test Q4, with cross-partition labels excluded. The March-to-April boundary remains within training; the handoff's description of excluding every quarter boundary was imprecise. The implementation excludes partition boundaries.

| Split | Rows | First source | Last source | Last target |
| --- | --- | --- | --- | --- |
| train | 121 | 2025-01-02 | 2025-06-27 | 2025-06-30 |
| validation | 63 | 2025-07-01 | 2025-09-29 | 2025-09-30 |
| test | 63 | 2025-10-01 | 2025-12-30 | 2025-12-31 |

LR uses seed 42, StandardScaler fit on training only, C=1, max_iter=2000. RF uses seeds 0/1/2, 300 trees, depth 5, leaf size 5, balanced class weights. Prediction threshold remains 0.5. Existing models, features and split implementation are unchanged.

### Experiment A (14 features)

`AAPL_return_1d`, `AAPL_mean_return_3d`, `AAPL_mean_return_5d`, `AAPL_mean_return_10d`, `AAPL_mean_return_20d`, `AAPL_cum_return_5d`, `AAPL_cum_return_10d`, `AAPL_cum_return_20d`, `AAPL_volatility_5d`, `AAPL_volatility_10d`, `AAPL_volatility_20d`, `AAPL_close_to_ma20`, `AAPL_volume_to_ma20`, `AAPL_daily_range`

### Experiment B (17 features)

`AAPL_return_1d`, `AAPL_mean_return_3d`, `AAPL_mean_return_5d`, `AAPL_mean_return_10d`, `AAPL_mean_return_20d`, `AAPL_cum_return_5d`, `AAPL_cum_return_10d`, `AAPL_cum_return_20d`, `AAPL_volatility_5d`, `AAPL_volatility_10d`, `AAPL_volatility_20d`, `AAPL_close_to_ma20`, `AAPL_volume_to_ma20`, `AAPL_daily_range`, `QQQ_return_1d`, `QQQ_cum_return_5d`, `AAPL_minus_QQQ_1d`

### Experiment C (22 features)

`AAPL_return_1d`, `AAPL_mean_return_3d`, `AAPL_mean_return_5d`, `AAPL_mean_return_10d`, `AAPL_mean_return_20d`, `AAPL_cum_return_5d`, `AAPL_cum_return_10d`, `AAPL_cum_return_20d`, `AAPL_volatility_5d`, `AAPL_volatility_10d`, `AAPL_volatility_20d`, `AAPL_close_to_ma20`, `AAPL_volume_to_ma20`, `AAPL_daily_range`, `QQQ_return_1d`, `QQQ_cum_return_5d`, `MSFT_return_1d`, `MSFT_cum_return_5d`, `NVDA_return_1d`, `NVDA_cum_return_5d`, `AAPL_minus_QQQ_1d`, `AAPL_minus_peer_mean_1d`

## Validation comparison

Baselines are reported once per split because their predictions do not depend on A/B/C. All model configurations use the same 63 dates and labels. RF entries use mean ± sample SD over seeds 0/1/2, not pooled predictions or standard error. Single-run SD is undefined (blank in CSV); displayed values below have nine decimal places. CSVs retain full precision.

| Experiment | Model | accuracy | balanced_accuracy | auc | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | logistic_regression | 0.492063492 | 0.515151515 | 0.581818182 | 1.000000000 | 0.030303030 | 0.058823529 |
| A | random_forest | 0.492063492 ± 0.015873016 | 0.512626263 ± 0.016689608 | 0.475420875 ± 0.014189433 | 0.638888889 ± 0.127293769 | 0.080808081 ± 0.017495463 | 0.142373142 ± 0.027391147 |
| B | logistic_regression | 0.555555556 | 0.575757576 | 0.593939394 | 1.000000000 | 0.151515152 | 0.263157895 |
| B | random_forest | 0.492063492 ± 0.041996053 | 0.508585859 ± 0.043237161 | 0.469023569 ± 0.025420318 | 0.568181818 ± 0.159090909 | 0.161616162 ± 0.017495463 | 0.250837931 ± 0.036335061 |
| C | logistic_regression | 0.555555556 | 0.574242424 | 0.612121212 | 0.857142857 | 0.181818182 | 0.300000000 |
| C | random_forest | 0.518518519 ± 0.018328580 | 0.535353535 ± 0.017948844 | 0.465993266 ± 0.011965896 | 0.641666667 ± 0.052041650 | 0.181818182 ± 0.030303030 | 0.282851201 ± 0.040970568 |
| baseline | majority_class | 0.523809524 | 0.500000000 | 0.500000000 | 0.523809524 | 1.000000000 | 0.687500000 |
| baseline | previous_direction | 0.555555556 | 0.554545455 | 0.554545455 | 0.575757576 | 0.575757576 | 0.575757576 |

## Test comparison

| Experiment | Model | accuracy | balanced_accuracy | auc | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | logistic_regression | 0.412698413 | 0.460714286 | 0.558163265 | 0.250000000 | 0.028571429 | 0.051282051 |
| A | random_forest | 0.460317460 ± 0.000000000 | 0.503571429 ± 0.000000000 | 0.571768707 ± 0.030275930 | 0.571428571 ± 0.000000000 | 0.114285714 ± 0.000000000 | 0.190476190 ± 0.000000000 |
| B | logistic_regression | 0.412698413 | 0.453571429 | 0.533673469 | 0.375000000 | 0.085714286 | 0.139534884 |
| B | random_forest | 0.412698413 ± 0.031746032 | 0.448809524 ± 0.030374645 | 0.553741497 ± 0.013854022 | 0.400000000 ± 0.100000000 | 0.123809524 ± 0.043643578 | 0.188810087 ± 0.061736688 |
| C | logistic_regression | 0.380952381 | 0.410714286 | 0.477551020 | 0.357142857 | 0.142857143 | 0.204081633 |
| C | random_forest | 0.402116402 ± 0.024246432 | 0.438095238 ± 0.021527549 | 0.470068027 ± 0.003863203 | 0.364957265 ± 0.085286127 | 0.114285714 ± 0.049487166 | 0.173232323 ± 0.066500021 |
| baseline | majority_class | 0.555555556 | 0.500000000 | 0.500000000 | 0.555555556 | 1.000000000 | 0.714285714 |
| baseline | previous_direction | 0.539682540 | 0.532142857 | 0.532142857 | 0.583333333 | 0.600000000 | 0.591549296 |

## Differences from STAGE1_RESULTS_2025.md

All baseline and LR metrics reproduce the committed CSVs exactly to floating-point precision and match the markdown rounding. Current RF values differ **before and after** integration. Project requirements permit scikit-learn >=1.4,<2, and installation resolved to **1.9.1**. A separate controlled rerun changing only scikit-learn to **1.7.1** reproduced every committed metric, including all RF runs, within 1e-14. This demonstrates version sensitivity; it does not establish the undocumented historical environment. No dependency pin or model behavior was changed to force agreement. Current deliverables use 1.9.1.

| Experiment | Metric | Archived mean | Current mean | Delta | Archived SD | Current SD |
| --- | --- | --- | --- | --- | --- | --- |
| A | accuracy | 0.470899471 | 0.460317460 | -0.010582011 | 0.009164290 | 0.000000000 |
| A | balanced_accuracy | 0.513095238 | 0.503571429 | -0.009523810 | 0.008247861 | 0.000000000 |
| A | auc | 0.562244898 | 0.571768707 | 0.009523810 | 0.025077971 | 0.030275930 |
| A | precision | 0.607142857 | 0.571428571 | -0.035714286 | 0.030929479 | 0.000000000 |
| A | recall | 0.133333333 | 0.114285714 | -0.019047619 | 0.016495722 | 0.000000000 |
| A | f1 | 0.218530823 | 0.190476190 | -0.028054633 | 0.024296025 | 0.000000000 |
| B | accuracy | 0.402116402 | 0.412698413 | 0.010582011 | 0.009164290 | 0.031746032 |
| B | balanced_accuracy | 0.438095238 | 0.448809524 | 0.010714286 | 0.010309826 | 0.030374645 |
| B | auc | 0.544557823 | 0.553741497 | 0.009183673 | 0.008310743 | 0.013854022 |
| B | precision | 0.375757576 | 0.400000000 | 0.024242424 | 0.020994555 | 0.100000000 |
| B | recall | 0.114285714 | 0.123809524 | 0.009523810 | 0.000000000 | 0.043643578 |
| B | f1 | 0.175201288 | 0.188810087 | 0.013608798 | 0.002231305 | 0.061736688 |
| C | accuracy | 0.396825397 | 0.402116402 | 0.005291005 | 0.015873016 | 0.024246432 |
| C | balanced_accuracy | 0.432142857 | 0.438095238 | 0.005952381 | 0.014285714 | 0.021527549 |
| C | auc | 0.487414966 | 0.470068027 | -0.017346939 | 0.012594224 | 0.003863203 |
| C | precision | 0.360101010 | 0.364957265 | 0.004856255 | 0.058413627 | 0.085286127 |
| C | recall | 0.114285714 | 0.114285714 | 0.000000000 | 0.028571429 | 0.049487166 |
| C | f1 | 0.173337445 | 0.173232323 | -0.000105122 | 0.039719440 | 0.066500021 |

`comparison_vs_committed.csv` includes all six means and SDs for both validation and test, with current values, archived values and deltas. Original fresh metrics/predictions are retained in `original_run/`; the 1.7.1 reproduction metrics are in `sklearn_1_7_1_check/`.

## Implementation and verification

- Reusable `src.evaluation`: CSV/DataFrame validation; six sklearn metrics; strict complete RF seed sets; grouped means and sample SD; structured comparison; validation-only configuration selection; command-line output.
- Reusable `src.backtesting`: one test configuration; checked next-trading-day price joins, actual-label agreement and contiguous dates; long/flat returns, buy-and-hold, equity, drawdown and Sharpe; command-line output.
- Inputs reject missing/nonfinite values, invalid binary labels/probabilities, duplicate rows, inconsistent dates/labels across runs, missing RF seeds, and validation backtesting. AUC is NaN if unavailable or only one actual class; P/R/F1 use zero_division=0. Single-class balanced accuracy follows sklearn, which may emit a diagnostic warning.
- Tests independently compute confusion counts and pairwise AUC, sample SD, compound wealth, initial-wealth drawdown and sample-volatility Sharpe. Real committed prediction files are checked against existing metrics; fresh predictions are independently cross-checked in the verification audit.
- Original, integrated and final training outputs match within 1e-14 across predictions, detailed metrics, summaries and features. Final suite: 22/22. Dependency consistency check passed. No Stage 2 code or notebooks were run.
- `verification.json` records base commit, data SHA-256, dimensions and checks. `environment_versions.txt` records Python and installed packages. `final_tests.txt` and `final_pipeline.txt` capture final runs.

## Issues and deliberately preserved legacy behavior

1. Initial Python 3.13 required a pandas 2.2.2 source build; this was stopped, and installed Python 3.11.4 was used in `.venv-stage1`. Requirements were not edited. The environment is locally ignored via Git's exclude file.
2. RF results depend on the permitted scikit-learn version, as demonstrated above. Reproducing a run requires the recorded environment versions.
3. Contrary to the handoff, the current training script already computes grouped means and SDs. That existing code stays intact; the reusable evaluator independently reproduces it.
4. Under the strict minimal-edit boundary, old “provisional” filenames, module description and run-info note remain in the training script. The new `evaluation_*` outputs and this report are authoritative; the legacy note does not mean shared evaluation was skipped.
5. The 63-day test sample and idealized execution limit interpretation. No hyperparameters, threshold, data or features were tuned to improve these results.

## Reproduce

From the repository root, with `.venv-stage1` or an equivalent Python 3.11 environment installed from project requirements:

```sh
.venv-stage1/bin/python experiments/run_stage1_training.py --year 2025 --output results/abu_stage1_2025
.venv-stage1/bin/python -m src.evaluation --predictions results/abu_stage1_2025/stage1_predictions.csv --output results/abu_stage1_2025
.venv-stage1/bin/python -m src.backtesting --predictions results/abu_stage1_2025/stage1_predictions.csv --prices data/processed/aligned_daily_ohlcv_5y.csv --output results/abu_stage1_2025
.venv-stage1/bin/python -m unittest discover -s tests -v
```

Keep `environment_versions.txt` with the results. The original tracked results and all protected sources/data/notebooks remain intact.
