# Design and Supporting Documentation

**Project:** LLM-Generated Interpretable Financial Factors for Stock Trend Prediction
**Group:** PG-S2-36 (Adelaide University capstone)
**Client and supervisor:** A/Prof Lingqiao Liu, Australian Institute for Machine Learning
**Team:** Abu Wakkas Bastob (team lead), Kevin Varghese, Taiyu Shi, Leyao Chang, Nikhil Nakade
**Date:** 20 November 2026

| Where | Link |
|---|---|
| Code (public) | https://github.com/Bastob96/SemanticMLPFinance (branch `main`) |
| Live app (Factor Lab) | https://llm-factor-lab.streamlit.app |
| Project board | Jira project PS3C (board "PS3C board", Sprints 0 to 6) |
| Notebooks | `Stage 1_Demo.ipynb`, `Stage 2_Demo.ipynb`, `Stage 2_v2_Demo.ipynb`, `Stage 2_v3_Ablation.ipynb` |
| Older design notes | `docs/FACTOR_LAB.md` (app), `src/stage2/README.md` (LLM loop), `DATA_README.md` (data) |

---

## 1. What we built and why

The client asked one question: can an LLM write stock factors that a person can read, and that help predict
whether Apple (AAPL) closes higher on the next trading day?

A factor is a short formula on past prices, for example "Apple's 5-day return minus the market's 5-day return".
Most factor search tools produce formulas nobody can read. An LLM can write a formula and explain it in plain
words, so the idea was to let the LLM propose and let the data decide.

The client's requirements, as recorded in the supervisor diary and Jira:

- Use only the data we have: daily prices and volume for AAPL, with MSFT, NVDA and QQQ as context (scoping, Aug 2026).
- One time split for everyone: Q1 and Q2 to train, Q3 to validate, Q4 as a sealed test (2 Sep 2026).
- Target is next-day direction, up or down (8 Oct 2026).
- A fully automated loop through an LLM API, judged by relative improvement over the Stage 1 factors (Oct 2026).
- Test factors together, not only one at a time (20 Oct 2026).
- A simple interface with live data and live predictions from the best model (8 and 15 Oct 2026).
- Factors whose meaning "makes sense in 5 minutes" (12 Nov 2026).

What we delivered:

1. **Stage 1:** a clean data set, 22 hand-made factors, two classifiers, two naive baselines, an evaluator and a
   backtester.
2. **Stage 2:** an automated LLM factor search, plus the controls needed to tell a real result from luck.
3. **Factor Lab:** a deployed app that makes a live next-day call from LLM factors and shows its full track record.
4. **Team dashboard:** a second Streamlit app with the pinned Stage 1 and Stage 2 results, built by Kevin.

## 2. Who owns what

| Member | Part | Main files |
|---|---|---|
| Abu Wakkas Bastob | Team lead, supervisor diary, Jira. Stage 1 evaluation and backtesting. Stage 2 LLM loop (v1, v2), luck tests, direction test. Factor Lab app. | `src/evaluation/`, `src/backtesting/`, `src/stage2/`, `src/app/`, `app.py`, `experiments/` |
| Kevin Varghese | Stage 1 features, models and training. Stage 2 training with the accepted factors. Team dashboard. | `src/features/`, `src/models/`, `experiments/run_stage1_training.py`, `streamlit_app.py`, `src/ui/` |
| Taiyu Shi | Data download. A separate LLM loop on a return-regression model (HGBR). | `data/`, `phase2_llm_loop—— Taiyu Shi/` |
| Leyao Chang | Data cleaning. Structured candidate generation and eight reference factors. | `data/processed/`, `Stage2_Factor_Files/` |
| Nikhil Nakade | Risk register. | |

## 3. How the system fits together

```
 Yahoo Finance (yfinance)
        |
        v
 data/  cleaned daily OHLCV, 4 tickers ----------------------------------+
        |                                                                |
        v                                                                |
 Stage 1  src/features -> 22 factors -> src/models (LR, RF, baselines)   |
        |        -> src/evaluation (6 metrics) -> src/backtesting        |
        v                                                                |
 Stage 2  src/stage2:  LLM proposes formulas (llm.py, prompt_v2.py)      |
        |   -> safe formula language (formula.py)                        |
        |   -> score on Q3 of 2022-2025 (scoring.py)                     |
        |   -> select / team / pair trials (loop.py, loop_v2.py)         |
        |   -> report card back to the LLM (feedback.py)                 |
        |   -> controls: templates, random numbers, family-wise bar      |
        |   -> sealed Q4 test once (final_test.py, pooled_test.py)       |
        |   -> direction test (direction.py, run_direction_test.py)      |
        v                                                                |
 app_data/  factor catalogue + research numbers (build_app_data.py)      |
        |                                                                |
        v                                                                v
 Factor Lab  app.py + src/app (live call, track record)     Team dashboard  streamlit_app.py + src/ui
```

Every step writes its output to `results/`, so later steps and both apps read saved files and never retrain
or call an LLM when a page opens. The one exception is live prices, which are downloaded when available.

## 4. Data

- Source: Yahoo Finance through the `yfinance` package, daily bars, `auto_adjust=True` (split and dividend adjusted).
- Tickers: AAPL (target), QQQ (market), MSFT and NVDA (peers).
- Range: 31 Aug 2021 to 27 Aug 2026, 1,254 rows per ticker in the raw files.
- Checks: missing values, duplicate dates, OHLC consistency, date order, same trading days for all four tickers.
- Four incomplete rows dated 28 Aug 2026 were removed, not filled in, so no price was invented.
- Label: 1 if AAPL's next close is above today's close, else 0. Labels that cross a split boundary are dropped.

Details: `DATA_README.md`.

## 5. Stage 1: baseline

**Factors.** 22 factors in three groups: AAPL only (14), plus QQQ (3), plus MSFT and NVDA (5). They are
returns over 1 to 20 days, volatility, distance from the 20-day average, volume against its average, daily
range, and Apple against the market and its peers. Every factor uses only data up to today's close.

**Models.** Logistic regression (seed 42, scaler fitted on training rows only) and random forest
(300 trees, depth 5, seeds 0, 1 and 2, reported as mean and standard deviation). Two naive baselines:
always the majority class, and "same direction as today".

**Evaluation.** Six metrics: accuracy, balanced accuracy, AUC, precision, recall, F1. Accuracy alone misleads
because up days are more common than down days.

**Backtest.** Pick the configuration with the best validation balanced accuracy, then simulate it on Q4 once:
long when the model says up, flat otherwise, no costs. This is a check, not a trading strategy.

**Result (2025, sealed Q4).** The best model, random forest on AAPL only, had balanced accuracy 0.513 and AUC
0.563. The "same as today" baseline had 0.532. The selected strategy returned -0.72% against +6.53% for holding
AAPL. We reported this as it was. It set the bar Stage 2 had to beat.

## 6. Stage 2: the LLM factor search

### 6.1 The loop (version 1, Oct)

1. **Propose.** Claude gets the goal, the data, the 22 factors written as formulas, a market summary up to
   30 Jun 2025 and the full scoreboard. It returns new factors and variants of good ones as JSON.
2. **Build.** Each formula is checked against a small formula language: allowed inputs, allowed operators,
   allowed windows, and every operator looks backwards only. Invalid formulas go back with the reason.
3. **Score.** The factor is added to the 22, both models are refitted on Q1 and Q2 of each year 2022 to 2025,
   and scored on Q3. Q4 is never touched during the search.
4. **Select.** Keep a factor if it helped in at least 3 of 4 years, with a mean AUC gain of at least 0.01, and a
   rank correlation below 0.9 with every factor already in.
5. **Feed back.** Every result goes on the scoreboard for the next round.

### 6.2 Version 2: testing factors together (20 Oct)

The client pointed out that a factor that only helps next to another one is thrown away in version 1.
Version 2 adds a report card for every factor (gain per year and per model, similarity, the reason it was kept
or rejected), a team trial (add the factor on top of the 22 plus the current team), and a pair trial (test two
promising factors together). The bars for the team and pair trials were set with random-number factors before
any real run (`experiments/calibrate_noise_v2.py`).

### 6.3 Ablation study (Sprint 5)

We switched parts of the loop off one at a time: no feedback, no pair trial, a smaller model, 16 rounds instead
of 8, template formulas instead of the LLM, and repeats of the normal setting. Main points:

- The LLM improved random forest where templates did not (balanced accuracy +0.031 against -0.006).
- Testing factors together beat judging them alone on the same 48 factors (+0.034 against +0.010).
- 16 rounds looked best on Q3 and was the weakest on Q4, so more rounds overfit the validation checks.
- The same setting picked 0, 3 and 4 factors in three repeats, so single runs must not be over-read.

### 6.4 The luck test (Nov)

We ran the same search on two controls with the same budget: template formulas written by code, and columns of
pure random numbers. The usual validation test kept 0 LLM factors, 11 template formulas and 11 columns of random
numbers. A test that keeps random numbers cannot tell us anything about the LLM.

So we changed how factors are judged:

- **Family-wise bar.** A factor must beat the best of N random factors, not just zero.
- **Pooled sealed test.** All four Q4 quarters together (250 days), with a bootstrap range and a p-value
  against random-number sets of the same size.
- **No selection.** Take every factor a model proposed, set each one's direction from past data only, average
  them into one signal, and score it with AUC on the sealed quarters.

### 6.5 The direction test (16 Nov)

Three LLMs (Opus 5.5, Fable 5.2, Haiku 5.5) each proposed 100 readable factors and said in advance which way each
one points. Round 1 was blind. In round 2 they saw round 1's directions on the search days only.

| | Opus 5.5 | Fable 5.2 | Haiku 5.5 |
|---|---|---|---|
| Blind directions right on sealed Q4 | 58% | 23% | 45% |
| After feedback, new ideas, sealed Q4 | 75% | 72% | 58% |
| Blind pool, directions from past data, sealed Q4 AUC | 0.592 | 0.597 | 0.582 |

For comparison, past-only directions: the 22 hand-made factors 0.558, template pools 0.49 to 0.53, random-number
pools 0.502 (95% of them below 0.573). The Stage 1 classifiers scored 0.465 (LR) and 0.504 (RF) on the same days.

**What it means.** The LLMs propose useful factors, but their stated directions are not reliable. The same Fable
factors score 0.42 with Fable's own directions and 0.60 with directions set by the data. So in the final design
the LLM proposes and the data sets the direction.

**Limits.** On fresh data (Jan to Aug 2026, used once) the edge is smaller: 0.52 to 0.55, not significant on 163
days. The LLM pools were not shown to beat the 22 hand-made factors. Most of the edge is known structure:
volatile days are followed by more up days, and 20 to 60 day strength reverses.

## 7. Factor Lab (the deployed app)

Code: `app.py`, `src/app/`. Live at https://llm-factor-lab.streamlit.app.

| Page | What it does |
|---|---|
| Live | Builds tomorrow's call from today's 5-minute bars so far, every 30 seconds while the market is open. Outside market hours it replays a past day step by step. The last step always equals the official call. |
| Today's call | The call for the next trading day (up or down, chance of an up day, tilt against a usual day), the factors behind it with their meaning, and the classifiers' opinions. Any past date can be replayed. |
| Track record | Every call since July 2022 scored with AUC and a shuffle (luck) test, against the 22 factors, templates, random numbers and the classifiers. |
| Factors | 918 factors with formula, meaning, the LLM's stated direction and the measured direction. |
| Research findings, Try a formula, How it works | The results and their limits, a formula tester, and the method. |

**How a call is made** (`src/app/engine.py`, class `Lab`):

1. Compute the 186 blind-round LLM formulas in the safe formula language.
2. Standardise each factor against its own past only.
3. At the start of each quarter, measure each factor's direction on earlier days whose outcome is known.
4. Average the factors, each turned so that higher means up. No factor is selected.
5. Turn the signal into a chance with a one-input logistic curve fitted on earlier days.

Track record (saved prices to 27 Aug 2026, AUC): all days since Jul 2022 0.541 (p 0.013), sealed quarters 0.588
(p 0.007), fresh 2026 0.532 (p 0.24). As a plain up/down call it is right 54% of the time, the same as always
saying up. The app says this on the page: it shows that the factors hold some information. It is not a trading tool.

The app makes no LLM calls and needs no API key. Without internet it falls back to the saved price file.

## 8. Key decisions

| Decision | Why |
|---|---|
| One quarterly split (Q1-Q2 / Q3 / Q4) for all members | Client instruction, 2 Sep. Markets change over long periods, so a short window is easier to read, and repeating it over 2022 to 2025 gives four tests. |
| Q4 sealed and used once per experiment | Any look at Q4 during the search would leak the answer into the choice. |
| Direction (up/down) as the target, AUC as the main score | Client decision, 8 Oct. AUC does not depend on a threshold or on how many up days there are. |
| A small formula language instead of free Python | The LLM cannot read future data or run unsafe code, and every formula stays readable. |
| Random numbers and templates run through the same pipeline | The only fair way to see how often luck passes our rules. |
| Random forest fixed at 4 inputs per split in Stage 2 | With the default, the forest changed at 25 factors and random columns looked useful (see debugging notes). |
| No factor selection in the app | Selection on short validation windows kept random numbers as often as real factors. |
| The LLM proposes, the data sets the direction | Blind stated directions were 23% to 58% right on sealed days. |
| Classifiers shown as a second opinion, not as the call | Retrained each quarter, LR and RF rank days no better than chance; the averaged signal does better. |
| Factor Lab built as its own app | It needs past-only calculations for every date; the team dashboard shows pinned Stage 1 and Stage 2 results. Both are linked from the README. |

## 9. Assumptions and limits

- Daily adjusted prices from Yahoo Finance, four tickers, US market only. Results are not claimed for other markets.
- The backtest uses same-close execution with no costs. That is optimistic and only a check.
- The LLMs' training data may include 2022 to 2025 prices. This is why we do not rely on their directions and why
  we tested on fresh 2026 data.
- Sealed Q4 data has now been used in many analyses. Fresh 2026 has been used once.
- Test sizes are small: about 63 days per Q4. The smallest AUC gain we can reliably see on four Q4 quarters is
  about 0.035 to 0.055.

## 10. Ethics and security

- **Not financial advice.** Every result page in Factor Lab says the edge is small and is not a trading tool.
- **Honest reporting.** Negative results (Stage 1 below the naive baseline, the usual test keeping random numbers)
  are reported, not hidden.
- **LLM output is a hypothesis.** An LLM can give a convincing explanation the data does not support, which the
  direction test showed. Every LLM factor is checked by the same data tests as a random one.
- **Keys and secrets.** The Anthropic API key is read from an environment variable and never committed. The full
  Git history was checked for keys and secret files before the final merge on 20 Nov (none found). The app needs
  no key.
- **Public data only.** No personal data is used.

## 11. Testing

Run all tests from the repository root:

```
python -m pip install -r requirements.txt
python -m pytest -q
```

Result on 20 Nov 2026, on `main` after the final merge: **98 passed** (Python 3.13, scikit-learn 1.9.1). The research numbers were produced with scikit-learn 1.7.2, which is why it is pinned; the tests pass on both.

| Test file | What it checks |
|---|---|
| `test_canonical_data.py` | The cleaned data matches the raw files; no duplicates or broken OHLC rows. |
| `test_stage1_training.py` | The split, labels at boundaries, seeds and model outputs. |
| `test_evaluation.py` | The six metrics match scikit-learn; aggregation over seeds. |
| `test_backtesting.py` | Long/flat returns, drawdown, Sharpe, the selection rule. |
| `test_stage2_formula.py` | The formula language rebuilds the 22 Stage 1 factors exactly; removing future data never changes a value; unsafe formulas are refused. |
| `test_stage2_training.py`, `test_stage2_v2.py` | Scoring, the selection rules, team and pair trials, report cards. |
| `test_stage2_batch.py`, `test_stage2_direction.py` | Batch mode, the pooled test, direction measurement on past days only. |
| `test_app_engine.py` | Factor Lab: cutting the future off the price file changes nothing for earlier dates; directions use known outcomes only; a planted factor is found; random numbers score about 0.50. |
| `test_app_live.py` | Market clock and holidays; the live preview at the close equals the official call; no later days are used. |
| `test_app_pages.py` | Every Factor Lab page runs for every factor set, in replay mode and with an invalid formula. |
| `test_ui_app.py`, `test_ui_data.py` | Team dashboard pages and data loading. |

Each Stage 2 run folder in `results/stage2/` keeps every prompt, raw LLM reply, scoreboard and kept factor, so
any run can be checked or replayed without calling the LLM again (`--proposer replay`).

## 12. Debugging notes

| Problem | How we found it | Fix |
|---|---|---|
| Random forest results changed between computers | Replaying run B through new code did not pick the same 4 factors. scikit-learn 1.9.1 gave different forests from 1.7.x. | Pinned `scikit-learn==1.7.2` in `requirements.txt` and ran every experiment in the same setup (Anaconda Python, same as runs B and C). The replay then picked the same 4 factors. |
| A crash inside pandas date handling | A newer NumPy broke pandas' datetime code; traced with Python's fault handler to the exact line. | Pinned compatible NumPy and pandas versions and re-ran. |
| LLM replies failed to decode | Some Anaconda installs ship an old `brotli` package that breaks compressed HTTP replies. | The client asks for gzip instead (`src/stage2/llm.py`). |
| The comparison picked the wrong run | Run C and the replay run both had folder names ending in `_claude`. | Folder matching now keeps them apart (commit 8796cac). |
| Random columns passed the team trial | Random forest looks at 4 inputs per split up to 24 factors and 5 from 25, so any extra column changed the forest itself. 8 of 18 random factors passed. | Fixed at 4 (`RF_MAX_FEATURES`), and the second-chance trial was turned off (commit dd76b00). |
| The usual validation test kept random numbers | Running the full pipeline on random numbers (Sec. 6.4). | Family-wise bar, pooled sealed test, and no selection in the app. |
| Live replay ended on 52.4% while the official call said 52% | Compared the replay's last step with the Today page. | The last step uses the official daily bar (commit 691e602). |
| An UP call with a negative signal confused users | Checking the deployed app before the review. | A sentence now explains the call against a usual day. |
| Track record page looked empty | The luck test ran on every visit with no loading message. | Cached each period's score and added a loading message (commit d51842b). |
| "LLM direction right" tile showed 63% | It counted round-2 factors, which had already seen data from the same years. | Counts blind-round factors only: 44% (commit 19a8062). |
| `git pull` from a bundle refused with "divergent branches" | Git's default pull setting on the Mac. | `git pull --no-rebase --no-edit <bundle> HEAD`. |
| Deployed app still showed old code after a push | Streamlit Community Cloud kept the old build. | Reboot the app from Manage app after each push. |

## 13. Version control and tracking

**Git.** One shared repository on GitHub, public. More than 90 commits on `main` from all five members between 27 Aug and
20 Nov 2026. Work happens on branches and reaches `main` through pull requests (#1 to #12). Branches: `abu/stage2-loop` (Stage 2 and Factor Lab, also what the live app deploys from),
`feature/stage1-baseline-training`, and others. 25 commits are authored by
Claude (Anthropic), used as a coding assistant on the Stage 2 and Factor Lab code; each of them carries a
co-author line and was pushed by Abu.

**Jira (project PS3C).** 7 sprints, 149 tasks and subtasks under 7 epics, all Done on 20 Nov 2026. One story point is
one hour.

| Sprint | Dates | Goal |
|---|---|---|
| 0 | 3 to 9 Aug | Onboarding and team communication |
| 1 | 10 to 20 Aug | Team charter, first supervisor meeting, project direction |
| 2 | 28 Aug to 10 Sep | Proposal and scoping document, risk plan, stakeholder presentation |
| 3 | 11 Sep to 8 Oct | Stage 1 pipeline and first Stage 2 experiments |
| 4 | 12 to 22 Oct | Automated LLM loop and first relative improvement, then v2 |
| 5 | 23 Oct to 5 Nov | Ablation study, repeat runs, best model to the UI |
| 6 | 6 to 20 Nov | Luck test, direction test, Factor Lab, final presentation and product review, handover |

## 14. How to run

```
git clone https://github.com/Bastob96/SemanticMLPFinance
cd SemanticMLPFinance
python -m pip install -r requirements.txt

streamlit run app.py                    # Factor Lab
streamlit run streamlit_app.py          # team dashboard

python experiments/run_stage1_training.py --year 2025
export ANTHROPIC_API_KEY=...            # only for new LLM runs; never commit it
python experiments/run_stage2_loop_v2.py --rounds 8
python experiments/run_direction_test.py --help
python -m pytest -q
```

## 15. Libraries and services

| Library or service | Version | Used for |
|---|---|---|
| Python | 3.11 to 3.13 (3.12 on Streamlit Cloud) | Everything |
| pandas | 2.2 | Data |
| NumPy | 1.26 to 2.x | Numbers |
| scikit-learn | 1.7.2 (pinned) | Logistic regression, random forest, metrics |
| SciPy | 1.11+ | Ranks and tests |
| yfinance | 1.7.0 | Daily and 5-minute prices from Yahoo Finance |
| anthropic (Python SDK) | 0.40+ | Calls to Claude for the factor proposals |
| Streamlit | 1.55+ | Both apps |
| Plotly | 5.24+ | Charts |
| pytest | 8+ | Tests |
| Anthropic API | Claude Sonnet, Opus 5.5, Fable 5.2, Haiku 5.5 | Factor proposals (every prompt and reply is saved in the run folder) |
| Streamlit Community Cloud | | Hosting Factor Lab |
| GitHub, Jira | | Version control, sprint tracking |

## 16. References

1. Kakushadze, Z. (2016). 101 Formulaic Alphas. *Wilmott*, 2016(84), 72-81. arXiv:1601.00991.
   (Formula-based factors; the style of our formula language.)
2. Wang, S., Yuan, H., Zhou, L., Ni, L. M., Shum, H.-Y., & Guo, J. (2023). Alpha-GPT: Human-AI Interactive Alpha
   Mining for Quantitative Investment. arXiv:2308.00016. (LLMs proposing readable factors.)
3. Harvey, C. R., Liu, Y., & Zhu, H. (2016). ...and the Cross-Section of Expected Returns. *Review of Financial
   Studies*, 29(1), 5-68. doi:10.1093/rfs/hhv059. (Many factors tested means a higher bar; our luck test.)
4. White, H. (2000). A Reality Check for Data Snooping. *Econometrica*, 68(5), 1097-1126. (Comparing the best of
   many tries against chance; our family-wise bar.)
5. Jegadeesh, N. (1990). Evidence of Predictable Behavior of Security Returns. *Journal of Finance*, 45(3), 881-898.
   (Short-term reversal, the main pattern our factors pick up.)
6. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
7. Fawcett, T. (2006). An Introduction to ROC Analysis. *Pattern Recognition Letters*, 27(8), 861-874. (AUC.)
8. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12,
   2825-2830.
9. Yang, X., Liu, W., Zhou, D., Bian, J., & Liu, T.-Y. (2020). Qlib: An AI-oriented Quantitative Investment
   Platform. arXiv:2009.11189. (Compared in Sprint 1 for the evaluation setup.)
10. Anthropic. Claude API documentation. https://docs.claude.com
11. Yahoo Finance data through yfinance. https://github.com/ranaroussi/yfinance
12. Streamlit documentation. https://docs.streamlit.io
