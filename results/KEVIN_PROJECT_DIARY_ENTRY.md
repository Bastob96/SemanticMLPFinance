Date: 2026-09-09
Type: Result
Contributor: Kevin Varghese
Title: Completed Stage 1 baseline model training and ablation experiments
Message:
Implemented the Stage 1 training pipeline using Taiyu's cleaned and aligned
AAPL, QQQ, MSFT and NVDA daily OHLCV data. The code constructs the agreed small,
interpretable feature set and next-trading-day AAPL direction target without
using future observations as features. The 2025 data was divided
chronologically into Q1-Q2 training, Q3 validation and untouched Q4 testing.
Rows whose target crossed a partition boundary were excluded, resulting in
121/63/63 labelled observations.

Logistic Regression was run with seed 42 and training-only standardisation.
Random Forest was run with seeds 0, 1 and 2, with mean and standard deviation
reported. Both models were evaluated under AAPL-only, AAPL+QQQ, and
AAPL+QQQ+MSFT+NVDA ablations, alongside majority-class and previous-direction
baselines. The strongest ML test setting was AAPL-only Random Forest (balanced
accuracy 0.513 ± 0.008; AUC 0.563 ± 0.026), but it did not outperform the
previous-direction baseline's balanced accuracy of 0.532. Adding market and peer
features did not improve Q4 performance. This close-to-random result is being
reported as expected for the initial pipeline.

The latest pull did not yet contain Abu's shared evaluation module, so model
labels and probabilities were saved for integration and the current
scikit-learn metrics are marked provisional. Automated checks confirm the
target direction, chronological boundary handling and ablation isolation.
