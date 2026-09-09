# SemanticMLPFinance

Interpretable financial factor discovery for next-trading-day AAPL prediction
using machine learning and LLMs.

## Stage 1 baseline training

Kevin's Stage 1 pipeline builds leakage-safe features from the cleaned daily
OHLCV file and predicts whether AAPL rises on the next trading day. For 2025,
Q1-Q2 is training, Q3 is validation, and Q4 is the untouched test set. Labels
that cross a partition boundary are excluded.

Models and seeds:

- Logistic Regression: seed 42, with scaling fitted on training data only
- Random Forest: seeds 0, 1 and 2; results report mean and standard deviation
- Majority-class and previous-direction baselines
- Ablations A (AAPL), B (AAPL+QQQ), and C (AAPL+QQQ+MSFT+NVDA)

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python experiments/run_stage1_training.py --year 2025
python -m unittest discover -s tests -v
```

Predictions, provisional metrics, the summary table, exact seeds and split sizes
are written to `results/`. The provisional metrics should be replaced or
confirmed with Abu's shared evaluation framework after it is merged.

## Stage 2 candidate factors

The separate `Stage2_Factor_Files` directory is preserved unchanged. See its
own README for factor-generation instructions and limitations.
