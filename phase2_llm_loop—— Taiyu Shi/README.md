# Capstone Phase 2: LLM-Driven Factor Generation with an Automated Feedback Loop

This project tests whether financial factors proposed by Claude, and refined over several rounds using
experimental feedback, improve the fixed Phase 1 HGBR baseline for predicting **AAPL's next-trading-day
log return**. It is an academic research prototype, not a trading system. A negative result is an
acceptable outcome.

```
Generate factors (Claude) → Validate → Train HGBR → Evaluate on validation → Retention policy
        ↑                                                                        │
        └──────────── deterministic structured feedback (no test data) ──────────┘
After the last round: freeze → select by validation rule → one final test vs baseline and naive
```

## 1. What is fixed (from Phase 1, Experiment A)

| Item | Setting |
|---|---|
| Data | `data/aligned_daily_ohlcv_2025_split.csv`: AAPL, MSFT, NVDA, QQQ, 250 common 2025 trading days, Yahoo Finance `auto_adjust=True` (split- and dividend-adjusted) |
| Target | `y(t) = ln(Close_AAPL(t+1) / Close_AAPL(t))`; features use data up to the close of day t |
| Baseline factors | `log_return_1d/5d/20d`, `close_to_ema_10/20`, `rsi_14_scaled` (Wilder RSI, (RSI−50)/50), `realized_vol_5/20`, `volume_change_1d`, `volume_z_20` |
| Model | `SimpleImputer(median)` + `HistGradientBoostingRegressor(learning_rate=0.04, max_iter=300, max_leaf_nodes=15, min_samples_leaf=15, l2_regularization=2.0, early_stopping=False, random_state=42)` |
| Split | Phase 1 chronological 70/15/15 over rows with complete baseline factors: train 2025-02-03→09-22 (160), validation 09-23→11-07 (34), test 11-10→12-30 (35) |
| Primary metric | Validation MAE of the log return. Secondary: RMSE, R², directional accuracy, Spearman IC, permutation importance |
| Naive benchmark | Predicted return = 0, i.e. tomorrow's close = today's close |

Every run checks that the baseline reproduces Phase 1 (validation MAE 0.011370, RMSE 0.013050).

## 2. Installation (Windows, VS Code)

```powershell
cd capstone_phase2
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
```
In VS Code choose **Python: Select Interpreter → .venv**. On macOS/Linux, use `python3 -m venv .venv` and
`source .venv/bin/activate`.

## 3. Data input

The default input is `data/aligned_daily_ohlcv_2025_split.csv`, in long format with the columns
`Ticker, Date, Open, High, Low, Close, Volume` (extra `Quarter`/`Split` columns are ignored).
You can pass a different file with `--data path.csv`. The loader:

- removes invalid bars and never imputes them;
- keeps only dates on which all four tickers trade;
- never forward- or backward-fills;
- flags any daily move above 25% as a possible unadjusted split.

## 4. Claude API configuration

```powershell
copy .env.example .env      # then edit .env and set ANTHROPIC_API_KEY
```
`.env` is git-ignored. The key is read only from the environment and is never written to logs or results.
Optional settings in `.env`: `CLAUDE_MODEL` (default `claude-opus-5-5`), `CLAUDE_EFFORT` (default `medium`),
`CLAUDE_USE_FALLBACKS` (server-side refusal fallback, default on) and `LLM_CACHE` (default on).

## 5. Running

| Command | What it does | API calls |
|---|---|---|
| `python main.py --baseline-only` | Stage 0 only: reproduces the baseline on validation. Does not touch the test set. | none |
| `python main.py --dry-run` | Full pipeline using fixed offline candidates, including deliberately invalid ones | none |
| `python main.py` | **Live experiment**: 3 rounds × 5 candidates with Claude | 3 (+≤1 retry per round) |
| `python main.py --rounds 4 --candidates 5` | Change the number of rounds or candidates | `rounds` |
| `python main.py --no-cache` | Force fresh Claude responses | |
| `python main.py --model claude-sonnet-5-5 --effort low` | Different model or effort | |
| `python -m pytest -q` | Offline test suite | none |

> **A dry run does not demonstrate LLM feedback.** Its candidates are fixed in `llm_generator.py`
> (`MOCK_LIBRARY`) and ignore the feedback. It checks only that data loading, the baseline, factor
> computation, validation, the loop, logging and the figures work together. Every dry-run output is
> labelled as such.

**Cost control:**
- One Claude call per round; there is no separate reflection call.
- The SDK timeout is 180 s, with at most 2 SDK retries.
- At most 1 extra request per round if the output is refused, truncated or not valid JSON.
- Identical prompts are served from `results/cache/llm/`.
- Token use and estimated cost are written to `llm_usage.csv` for every call.
- Configuration errors (bad key, unknown model) stop the run *before* the test set is evaluated.

## 6. How the loop works

**Each round:**
1. Builds a prompt containing:
   - the data context (assets, column definitions, the factor language, the 10 baseline formulas, and training-period statistics only);
   - baseline and naive validation metrics;
   - the retained LLM factors;
   - **the structured feedback from every earlier round**;
   - the list of expressions already tried.
2. Calls Claude with a JSON schema (structured outputs). Claude must also explain, for each candidate, how the earlier results motivated it (`relation_to_previous`).
3. Validates each candidate (see 6.1). Invalid candidates are rejected with a code and a reason; the others continue.
4. Evaluates valid candidates with the fixed HGBR (fit on train, score on validation), applies the retention policy (6.2) and computes diagnostics.
5. Builds the feedback *deterministically* from the computed numbers:
   - per-candidate status and rejection reasons;
   - MAE vs the baseline and vs the current set;
   - the best ADD/REPLACE move;
   - correlations;
   - Spearman correlation with the target on validation;
   - permutation importance;
   - rule-based observations;
   - remaining capacity.

**Test protection:** prompts are built only from training statistics and validation results, and an
assertion stops the run if any test date appears in a prompt.

### 6.1 Factor language and validation
Factors are expressions such as `rolling_mean(aapl_ret1 - qqq_ret1, 10)` over the inputs
`{aapl,msft,nvda,qqq}_{open,high,low,close,volume,ret1,logvol}`. The full reference is in
`factor_engine.py` and is saved to `factor_language.txt` in each run.

Expressions are interpreted from Python's syntax tree using an allowlist. **Nothing generated by the LLM
is executed as code**, so imports, attributes, indexing, file access and similar are impossible.

**Checks, in order:**
1. Schema
2. Name
3. Declared inputs exist
4. Syntax / allowlist
5. Lookback ≤ 20 trading days (nested windows add up)
6. Duplicate expression
7. **Truncation leakage test** (values up to date T must not change when later data is removed)
8. ≤ 2% missing on modelling rows
9. Not near-constant on training rows
10. |Spearman| < 0.95 against the baseline, retained factors and earlier valid candidates in the same round, on training rows

### 6.2 Retention policy (fixed before any run, identical in every round)
- **Moves:** each valid candidate may be ADDED (if fewer than 5 LLM factors are retained) or may REPLACE a retained LLM factor. Its best move is the one with the lowest validation MAE.
- **Order:** candidates are tried in order of their best-move MAE. Each is re-evaluated against the current set.
- **Acceptance:** a move is accepted only if validation MAE falls by **≥ 0.5% (relative)** and validation **RMSE does not rise**.
- **Limits:** at most **2** accepted changes per round and at most **5** retained LLM factors.
- **Fixed parts:** baseline factors and HGBR settings never change. A replaced factor is retired permanently.
- **No change:** if nothing qualifies, the set stays the same, Claude is told so, and the loop continues.

**Final selection:** the end-of-round set with the lowest validation MAE that contains at least one LLM
factor. Ties go to fewer factors, then the earlier round. If no such set exists, the selected model is the
baseline and the run is reported as a **negative result**.

**Final test:** after freezing the definitions, the baseline and the selected model are refit on
train + validation (194 rows) and evaluated **once** on the test rows, together with the naive benchmark.
A train-only refit (the Phase 1 protocol) is also reported as a secondary result.

## 7. Results layout

```
results/runs/<run_id>/
  run_config.json            settings, fixed config, library versions, data hash
  data_manifest.json         coverage, integrity checks, adjustment note
  split_boundaries.json      feature and target dates of each split
  baseline/                  factor definitions, validation metrics, Phase 1 reproduction check, importance
  llm_data_context.json      exact data context sent to Claude
  rounds/round_k/
    prompt.json              exact system + user prompt
    llm_response.json        raw API response(s), usage, cost, feedback_analysis
    candidates.json          parsed candidate specifications
    validation_report.csv    status, rejection code and reason per candidate
    candidate_evaluations.csv  MAE vs baseline, best move, decision and reason
    move_evaluations.csv     every ADD/REPLACE move that was evaluated
    retention_decision.json  changes, retained/retired factors, metrics before/after
    feedback.json            feedback passed to the next round
    diagnostics/             permutation importance (validation), correlations (train)
  iteration_summary.csv      baseline and each round's validation metrics
  llm_usage.csv              tokens and estimated cost per call
  final/selected_model.json, frozen_factor_definitions.json, test_metrics.json, test_predictions.csv
  run_summary.json, run.log
results/figures/<run_id>/    fig1..fig6 PNG + the CSV each was drawn from
```

## 8. Figures

| Figure | Content | Data period |
|---|---|---|
| 1 | Test MAE and RMSE: naive, HGBR baseline, LLM-enhanced HGBR | test |
| 2 | Validation MAE and RMSE for the baseline and each round. Solid line = best retained set; hollow markers = that round's model with all valid candidates added | validation |
| 3 | Permutation importance of the selected model: increase in validation MAE when a factor is shuffled (30 repeats, seed 42), model fit on train. Negative values mean shuffling *lowered* MAE (noise or redundancy) | validation |
| 4 | Actual vs predicted AAPL close (`Close_t × exp(predicted return)`) for the baseline and LLM model, with naive | test |
| 5 | Spearman correlations of the baseline and selected LLM factors | train |
| 6 | Every candidate's outcome by round (accepted / not accepted / invalid with code) | validation |

## 9. Interpreting the results
- **Compare against the naive benchmark as well as the baseline.** On this data the naive zero-return forecast has lower validation MAE (0.00841) than the HGBR baseline (0.01137). This is common for next-day returns.
- **Validation gains are optimistic.** The loop uses the same 34 validation days repeatedly, so a lower validation MAE may not carry over to the test period; the dry run shows exactly that. The 0.5% threshold, the RMSE guard, the per-round and total caps, and the absence of combination search limit this, but cannot remove it. Report it as a limitation.
- With 34 validation and 35 test days, small differences are not statistically meaningful.
- The Phase 1 test period was already viewed during Phase 1, so it is not completely untouched.
- Claude's responses are not deterministic. Exact prompts and responses are saved, and the cache lets you replay a run.

## 10. Troubleshooting
| Problem | Fix |
|---|---|
| `ANTHROPIC_API_KEY is not set` | Create `.env` from `.env.example`, or use `--dry-run` |
| `AuthenticationError` / `NotFoundError` | Check the key, or the `CLAUDE_MODEL` spelling and your account's model access |
| `BadRequestError` mentioning `fallbacks` or `effort` | Set `CLAUDE_USE_FALLBACKS=false` (needed off the first-party Claude API) or `CLAUDE_EFFORT=none` for models without effort support |
| Rate-limit or overload errors | Wait and rerun; already-completed responses are cached |
| A round shows `llm_status: failed` | See `rounds/round_k/llm_response.json`; the loop continues with that round's failure in the feedback |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Figures missing | `run.log` names the failed figure and the reason; no figure is drawn from placeholder data |
