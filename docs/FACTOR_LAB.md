# Factor Lab: the project's interface

A Streamlit app that makes a live next-day call for AAPL from LLM-proposed, human-readable factors,
shows the reasons for the call, and shows the evidence for and against it.

It answers the supervisor's requests for the interface: live data, live predictions from the best model,
the LLM-discovered factors and their predictions, and results a person can understand.

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

Tested with Streamlit 1.51 and 1.67 (Python 3.13). If you already have the project's packages, only `yfinance` may
be missing: `pip install yfinance`.

The app opens at http://localhost:8501. With internet it downloads daily prices from Yahoo Finance.
Without internet (or if Yahoo is unavailable) it uses the saved price file and says so in the sidebar.

## Deploy it (Streamlit Community Cloud, free)

1. Push the branch that contains `app.py` to GitHub.
2. Go to https://share.streamlit.io and sign in with the GitHub account that owns the repository.
3. Create app, choose the repository and branch, and set the main file path to `app.py`.
4. In Advanced settings choose Python 3.12, then Deploy.

No secrets or API keys are needed: the app makes no LLM calls. The LLM factors are stored in `app_data/`.

## Pages

| Page | What it shows |
|---|---|
| Today | The call for the next trading day (UP or DOWN, chance of an up day, how it compares with a usual day), the factor groups and single factors behind it with their meanings, how similar signals did before, the classifiers' opinions, recent calls. Any earlier date can be replayed. |
| Track record | Every past call scored against what happened: ranking skill (AUC) with a luck test, the same recipe on the 22 existing factors, on template formulas, on random numbers, and the project's classifiers. Year by year, reliability of the stated chances, and whether acting on it would have paid. |
| Factors | All factors with formula, meaning, the direction the LLM stated, the direction the data shows, edge by year and by level. |
| Research findings | The five results of the project and their limits, with the numbers from the research runs. |
| Try a formula | Write a factor in the formula language and test it the same way. |
| How it works | The method, the safeguards, a glossary and credits. |

## How a call is made

1. **Factors.** Three LLMs (Opus 5.5, Fable 5.2, Haiku 5.5) each proposed 100 interpretable factors, twice
   (`experiments/run_direction_test.py`). The default set is the first, blind round: 186 different formulas.
2. **Values.** Formulas run in the safe formula language (`src/stage2/formula.py`): backward-looking operators only.
3. **Standardising.** Each factor is compared with its own past (expanding mean and standard deviation, shifted one day).
4. **Directions.** At the start of each quarter, each factor's direction is measured on all earlier days whose
   outcome is known (within-year AUC, as in `src/stage2/direction.py`).
5. **Signal.** The average of the standardised factors, each turned so that higher means up. No factor is selected.
6. **Chance.** A one-input logistic curve fitted on the signal's earlier out-of-sample days.

All of this is in `src/app/engine.py` (class `Lab`).

## Decisions and why

| Decision | Reason |
|---|---|
| The LLM proposes, the data sets the direction | The direction test showed the LLMs' stated directions are unreliable (Fable 5.2: right for 23% of blind factors on the sealed quarters), while the same factors work with measured directions. |
| No factor selection | Selecting factors on validation data kept random numbers as often as real factors. Averaging every proposed factor leaves nothing to overfit. |
| Directions re-measured quarterly, on earlier days only | Keeps every past call identical to what a live app would have shown, at low cost. |
| Default set is the blind round | Those factors were proposed before any result was seen, so the set itself cannot have been picked using the data. |
| AUC as the main score | It does not depend on a threshold, and it is the project's metric throughout. |
| Chance compared with "a usual day" | AAPL closes up on about 53% of days, so "UP" alone says little. The tilt against the usual day is the informative part. |
| Classifiers shown as second opinions, not as the call | Retrained each quarter on earlier days, logistic regression and random forest rank days no better than chance; the signal does better. |
| Full price history downloaded when live | Adjusted prices change after dividends; mixing a saved file with newer downloads would create small jumps. |
| Today's unfinished bar is dropped | A call must only use finished days (New York close plus a margin). |

## Assumptions and limits

- Daily adjusted prices from Yahoo Finance; four tickers (AAPL, QQQ, MSFT, NVDA).
- The edge is small, not steady between years, and mostly known structure (volatility, reversal).
  The app says so on the Today and Track record pages. It is a research demonstration, not financial advice.
- The LLMs' training data may include 2022 to 2025 market history. The app does not rely on their directions.
- NYSE holidays are listed for 2026 and 2027 (used only to name the next trading day).
- Numbers from the classifiers can differ slightly between scikit-learn versions; the signal uses no model and does not.

## Files

| File | Role |
|---|---|
| `app.py` | Entry point: page setup, sidebar, navigation |
| `src/app/engine.py` | Every calculation: data loading and checks, factor values, directions, signal, chance, scores |
| `src/app/state.py` | Sidebar choices and cached results shared by the pages |
| `src/app/theme.py` | Colours, styles, chart defaults |
| `src/app/views/*.py` | One file per page |
| `app_data/factors.csv`, `app_data/research.json` | Factor catalogue and research numbers, built by `experiments/build_app_data.py` |
| `tests/test_app_engine.py`, `tests/test_app_pages.py` | Tests |
| `.streamlit/config.toml` | Theme |

## Tests (`pytest tests/test_app_engine.py tests/test_app_pages.py`)

- Removing the future from the price file leaves every earlier factor value, direction, signal and chance unchanged.
- Directions use only days whose outcome was already known.
- The edge calculation equals the research code's.
- Factor contributions add up to the signal.
- A planted informative factor is found; random numbers score about 0.50.
- Live download: tidying, dropping an unfinished day, dropping days with a missing ticker, refusing bad data, falling back to the saved file.
- Calendar: next trading day across weekends and holidays.
- Every page runs without error for every factor set, in replay mode, on the first day of the record, with newer live prices, and with an invalid formula.

## Libraries

streamlit (interface), plotly (charts), pandas and numpy (data), scipy (ranks), scikit-learn (logistic curve and the
two classifiers), yfinance (prices).
