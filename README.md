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

Predictions, metrics, the summary table, exact seeds and split sizes are written
to `results/`. The merged `src/evaluation` and `src/backtesting` modules validate
the saved prediction schema, aggregate Random Forest seeds and apply
validation-only selection.

## Stage 2 quarterly factor comparison

Stage 2 compares the existing features against the same features plus the eight
accepted LLM-generated factors. Both stages are checked against
`configs/canonical_data_2025.json`, which fixes the shared data snapshot and the
121/63/63 quarterly profile.

Run validation first. This does not load Q4:

```bash
python experiments/run_stage2_training.py --phase validation
```

Review `results/stage2_2025/STAGE2_VALIDATION_2025.md` and the generated lock.
Then run the preregistered final comparison once:

```bash
python experiments/run_stage2_training.py --phase final --confirm-locked-test
```

The final prediction CSV uses Abu's existing schema and goes directly through
the shared evaluator and backtester. Taiyu's separate 70/15/15 return-regression
experiment remains under its own notebook, result files and Stage 2 data profile;
it is not mixed with this quarterly direction-classification comparison.

Run all tests from the repository root:

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s Stage2_Factor_Files/tests -v
```
