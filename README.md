# SemanticMLPFinance

Interpretable financial factor discovery for next-trading-day AAPL prediction
using machine learning and LLMs.

## Interactive research dashboard

The Streamlit dashboard is the presentation layer for the committed Stage 1
and Stage 2 outputs. Its result pages are intentionally read-only: opening them does not train a
model, call Claude, download market data, or use Q4 to make a decision. The one
exception is the **Live prediction** page, which downloads current prices and
refreshes on a timer. It applies the Stage 1 recipe (2025 Q1 to Q2 training rows)
to the newest finished US session and never reads Q4 results. It needs internet
access and `yfinance`, and falls back to the saved data if the download fails.

```bash
python -m pip install -r requirements.txt
python experiments/export_ui_artifacts.py  # only needed if results/ui is absent or intentionally refreshed
streamlit run streamlit_app.py
```

The interface includes:

- Stage 1 multi-year metrics and A/B/C feature ablations
- validation-selected Q4 long/flat backtests
- Stage 1 permutation importance
- the pinned Claude and random-control Stage 2 candidate scoreboards
- per-year validation gains and the frozen Stage 2 Q4 comparison
- downloadable result tables and a source-artifact manifest
- a Live prediction page with auto refreshing prices and Stage 1 model calls

Dashboard data is pinned by `results/ui/manifest.json`. Do not refresh this
bundle after inspecting Q4 unless the experimental protocol itself has been
changed and documented.

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

Predictions, shared evaluation metrics, the summary table, exact seeds and
split sizes are written to `results/`. Abu's evaluation and backtesting modules
validate and aggregate the saved prediction format used by this pipeline.

## Stage 2 candidate factors

The canonical Stage 2 implementation is in `src/stage2/`. It proposes readable
factor formulas, validates and computes them, scores them on the 2022–2025 Q3
validation quarters, and applies fixed consistency, effect-size and
correlation gates. The final Q4 runner is separate from discovery.

```bash
python experiments/run_stage2_loop.py --help
python experiments/run_stage2_final_test.py --help
```

The committed dashboard pins the successful Claude run
`20261015_1635_claude` and the matched random-control run
`20261015_1645_random`. The older `Stage2_Factor_Files/` directory remains as
legacy/reference material and is not the dashboard's data source.

## Tests

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s Stage2_Factor_Files/tests -v
```
