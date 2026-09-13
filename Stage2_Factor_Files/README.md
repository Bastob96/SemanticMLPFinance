# Stage 2 candidate factors

This directory contains a structured candidate-generation runner, local validation,
eight reference factor implementations, a 2025 factor table and tests. Generation
runs are recorded separately from model training so that the final training pipeline
can change without changing the factor proposal process.

## Run

Keep this folder at the root of `SemanticMLPFinance`, beside the existing `data`,
`src` and `configs` folders.
Use Python 3.10 or newer. In a project environment with pandas and numpy installed,
run these commands from the repository root (Windows PowerShell, Git Bash or Linux):

```text
python Stage2_Factor_Files/run.py
python -m unittest discover -s Stage2_Factor_Files/tests -v
```

If dependencies are missing, install `Stage2_Factor_Files/requirements.txt` in a
separate virtual environment. Data-builder versions are recorded in
`data/processed/build_report.json`; no yfinance download or API key is needed
for rebuilding the reference factor table.

## Generate a candidate batch

Set `OPENAI_API_KEY` in the environment and choose a model available to the
project account. Then run from the repository root:

```text
python Stage2_Factor_Files/generate_factors.py --model MODEL_NAME
```

The command reads `configs/llm_factor_prompt_v1.txt` and
`configs/llm_factor_schema_v1.json`. It requests exactly eight structured
candidates, then checks required fields, unique IDs and names, allowed inputs and
operators, formula syntax, formula/input agreement, lookbacks and duplicate
formulas. It does not read validation or test-period performance.

Each run is saved under `data/generated/run_TIMESTAMP/`:

| File                      | Purpose                                                           |
| ------------------------- | ----------------------------------------------------------------- |
| raw_output.json           | Original structured response                                      |
| generation_metadata.json  | Model, time, response ID, hashes, token use and validation status |
| validated_candidates.json | Accepted candidate batch; absent when validation fails            |

A rejected run keeps its original response and error record for review. It does
not create `validated_candidates.json`. An existing JSON file can be checked
without making a request:

```text
python Stage2_Factor_Files/generate_factors.py --validate path/to/candidates.json
```

The builder reads the existing `aligned_daily_ohlcv_5y.csv` and
`aligned_daily_ohlcv_2025_split.csv`. For each file it checks the repository's
`data/processed/` and `data/processed_5y_yahoo/` directories. It stops if neither
exists, or if both contain the same filename, rather than picking an unknown copy.
Source files are not changed. The output files inside this folder are regenerated.

If sources are elsewhere, use explicit paths (quote any path containing spaces):

```text
python Stage2_Factor_Files/run.py --five-year "path/to/aligned_daily_ohlcv_5y.csv" --split-2025 "path/to/aligned_daily_ohlcv_2025_split.csv"
```

The default test command uses the two documented data directories. For a custom
layout, `STAGE2_FIVE_YEAR` and `STAGE2_SPLIT_2025` can specify the test input files.

## Output

All outputs are under this folder's `data/processed/`:

| File                                   | Rows | Use                                                   |
| -------------------------------------- | ---: | ----------------------------------------------------- |
| stage2_candidate_factor_table_2025.csv |  250 | Complete dated candidate table                        |
| stage2_train_2025.csv                  |  121 | Training rows with within-period targets              |
| stage2_validation_2025.csv             |   63 | Validation rows with within-period targets            |
| stage2_test_2025.csv                   |   63 | Held-out test rows with within-period targets         |
| build_report.json                      |    - | Counts, boundary exclusions, input hashes and runtime |

Each CSV has 33 columns: Date, Quarter, Split, 20 OHLCV columns, 8 candidate
factors, target_next_day_up and target_date. Raw OHLCV columns are retained for
traceability; they are not automatically selected as model inputs.

The complete table retains the supplied date split: 122 train, 64 validation,
64 test dates. The three smaller files exclude June 30, September 30 and
December 31 because their next-day labels use a price outside their own period.
The December 31 label remains in the complete table and uses January 2, 2026.
Use the smaller files for modelling. Apply the same boundary policy to the
baseline when comparing it with these factors.

## Timing and model inputs

FCT-08 uses a three-day QQQ return lagged by two trading dates. Its calculation therefore needs QQQ closing prices back to t-5 (six close observations including day t).
The lookback_days value of 3 records the largest single operator parameter, not the total history required by this nested formula.

The signal is formed after day-t OHLCV is available. The target is 1 if AAPL's
next trading-day adjusted close is higher than today's close, otherwise 0.
This is a classification target, not a claim that a strategy can trade at the
already-known closing price. Backtesting needs its own executable entry timing.

All rolling windows end at t and include t. Previous-year data provides rolling
history, not extra training rows. No scaling or imputation is fitted here.
For model training, fit transformations on train only and retain the Q4 holdout.

`configs/feature_sets.json` contains the exact candidate columns for A, B and C.
Use those lists explicitly. Never pass every numeric column to the model:
target_next_day_up is y; Date, Quarter, Split and target_date are metadata.
This accepted generated batch contains no AAPL-only Stage 2 candidate, so A has no new Stage 2 factors. B contains the AAPL/QQQ candidates, while C adds the MSFT/NVDA candidates.
These are candidate additions, not a replacement for the Stage 1 baseline set.
Do not count the same formula twice when merging feature sets.

## Checks and limits

The tests cover all eight formulas at four dates, all targets, input alignment,
split boundaries, output validity, repeatability, input-path handling, and ablation
input isolation. They also cover the generation prompt split, schema checks,
formula restrictions, metadata records and rejected-run handling. Prefix
truncation and future-data mutation tests check that the implemented feature
calculations do not depend on later rows.

The supplied prices were downloaded with auto_adjust=True. These tests check
calculation timing on that fixed snapshot; they do not establish that the data
is a point-in-time archive. No factor has been accepted or rejected using Q4
prediction results. LLM-generated formulas must be reviewed before implementation;
do not execute arbitrary model-generated code or expressions with eval.
