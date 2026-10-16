"""Builds 'Stage 2_Demo.ipynb'. Run: python experiments/build_stage2_notebook.py

All the Stage 2 code is written into the notebook cells (like the Stage 1 demo).
The code cells are taken from src/stage2/*.py when the notebook is built, so the
notebook and the Terminal command always run exactly the same code.
"""

import ast
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []


def module_cell(path, title):
    """The code of one src/stage2 file as a notebook cell: the file's description becomes
    comments, and imports between the Stage 2 files are dropped (those names are already
    defined in earlier cells)."""
    text = (ROOT / path).read_text()
    tree = ast.parse(text)
    lines = text.splitlines()
    doc = ast.get_docstring(tree)
    start = tree.body[0].end_lineno if doc else 0
    body = [l for l in lines[start:]
            if not l.startswith("from __future__") and not l.startswith("from .")]
    while body and not body[0].strip():
        body.pop(0)
    header = [f"# {title}  (also saved as {path})"]
    if doc:
        header += ["#"] + [("# " + l).rstrip() for l in doc.splitlines()]
    return code("\n".join(header + [""] + body).rstrip())


cells.append(md("""# Stage 2: Letting an LLM Find New Factors

**PG-S2-36 - Automatic Discovery of Interpretable Financial Factors**

Abu Wakkas Bastob

In Stage 1 I built 22 factors by hand and found that none of the models could predict Apple's next-day direction reliably.

Dr Liu's question for Stage 2 is: **can an LLM come up with factors on its own that do better than ours?**

To test this I built a loop that runs by itself. Claude suggests new factors, my code tests them, and the results go back to Claude so it can try again. All the code is in this notebook, step by step, like in Stage 1."""))

cells.append(code("""# Section 1: Setup, the data, and the two saved runs
import sys, json, re, time, random, warnings
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, '.')                 # to use my Stage 1 code in src/ (features, split, models, backtest)
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 120)

from src.features import build_stage1_features

raw = pd.read_csv('data/processed/aligned_daily_ohlcv_5y.csv')   # same data as Stage 1
features = build_stage1_features(raw)                              # my 22 Stage 1 factors + the target

RUNS = sorted(Path('results/stage2/runs').glob('*'))
LLM_RUN = [r for r in RUNS if r.name.endswith('_claude')][-1]      # the Claude run
RANDOM_RUN = [r for r in RUNS if r.name.endswith('_random')][-1]   # the random-formula run

def load(run):
    board = pd.read_csv(run / 'scoreboard.csv').fillna({'reason': '', 'parent': '', 'closest': ''})
    kept = json.loads((run / 'kept_factors.json').read_text())
    return board, kept

llm_board, llm_kept = load(LLM_RUN)
rnd_board, rnd_kept = load(RANDOM_RUN)
config = json.loads((LLM_RUN / 'config.json').read_text())
print('Claude run :', LLM_RUN.name, '| model:', config.get('model'), '| rounds:', llm_board['round'].max())
print('Random run :', RANDOM_RUN.name, '| rounds:', rnd_board['round'].max())"""))

cells.append(md("""## Step 2: How the Loop Works

Every round goes through the same five steps:

1. **Claude suggests 6 factors.** 3 are improvements on something that already worked, and 3 are completely new ideas. *(code in Step 5)*
2. **My code builds each factor** from the formula Claude wrote, using only past prices. *(code in Step 3)*
3. **I test each one.** I add it to the 22 Stage 1 factors, retrain the models, and check if the score on Q3 goes up, for each year from 2022 to 2025. *(code in Step 6)*
4. **I keep it or throw it away.** It has to help in at least 3 of the 4 years, and by more than random luck would. *(code in Step 6)*
5. **The results go back to Claude** in the next round, including what failed and why. *(code in Steps 5 and 6)*

Nobody picks the factors by hand, and Q4 is never used during the loop. It is saved for the final test in Step 10.

Talking to Claude costs a little money and takes a few minutes, so I ran the 8 rounds once and saved everything Claude sent back. This notebook uses those saved answers. In Step 6 I re-run a full round with the code, and Step 11 has the switch to run the whole loop again live."""))

cells.append(md("""## Step 3: Turning Claude's Formula into Numbers

Claude writes each factor as a formula. For example:

`subtract(pct_change(AAPL_Close, 5), pct_change(QQQ_Close, 5))`

means "Apple's 5-day return minus the market's 5-day return".

The code below reads the formula, checks it only uses allowed data and operations, and calculates it for every day. Every operation only looks backwards in time."""))

cells.append(module_cell("src/stage2/formula.py", "Section 3a: The formula language"))

cells.append(module_cell("src/stage2/context.py", "Section 3b: My 22 Stage 1 factors as formulas, and the market summary for Claude"))

cells.append(md("""Before trusting it, I checked two things:
* If I write my 22 Stage 1 factors as formulas, I get exactly the same numbers as in Stage 1.
* If I delete all the data after a certain day, that day's value doesn't change. So no factor can secretly use future prices."""))

cells.append(code("""# Section 3c: Check that the formula code works properly
prices = wide_prices(raw)
stage1 = features.set_index(pd.to_datetime(features['Date']))

# Check 1: rebuild my 22 Stage 1 factors from formulas
same = {name: np.allclose(evaluate(f, prices).reindex(stage1.index), stage1[name]) for name, f in BASE_FORMULAS.items()}
print(f'Check 1: {sum(same.values())} of {len(same)} Stage 1 factors match exactly')

# Check 2: cut the data at one day and make sure no factor changes
day = pd.Timestamp('2024-03-15')
formulas = llm_board.loc[llm_board.status != 'invalid', 'formula']
changed = sum(not np.isclose(evaluate(f, prices.loc[:day]).iloc[-1], evaluate(f, prices).loc[day], equal_nan=True)
              for f in formulas)
print(f'Check 2: {len(formulas)} Claude factors recalculated with data cut at {day.date()}: {changed} changed')

# Formulas that break the rules are refused
for bad in ['lag(AAPL_Close, -1)', 'AAPL_Close + QQQ_Close', 'pct_change(TSLA_Close, 1)']:
    try:
        validate(bad)
    except FormulaError as e:
        print(f'Refused  {bad:28} -> {e}')"""))

cells.append(md("""## Step 4: What I Tell Claude

Each round Claude gets the same instructions, plus everything it has tried so far and how each one scored:
* What we are predicting, and which data it can use
* How each factor will be tested and the rule for keeping it
* My 22 existing factors, so it doesn't repeat them
* A short summary of each stock's yearly returns, only up to June 2025 so it learns nothing about the test periods

The instructions are written by `build_prompt` (Step 5). Below is what Claude actually received in round 8."""))

cells.append(code("""# Section 4: The instructions Claude got in the last round (beginning, then the results part)
last = llm_board['round'].max()
prompt = (LLM_RUN / f'round_{last:02d}_prompt.txt').read_text()
print(prompt[:1400])
print('\\n.........\\n')
start = prompt.index('SCOREBOARD OF EVERYTHING TRIED SO FAR')   # where the past results start
print(prompt[start:start + 1600])"""))

cells.append(md("""## Step 5: Asking Claude

The code below writes the instructions (`build_prompt`) and sends them to Claude (`ClaudeProposer`). Claude answers with a short reflection on the results so far and 6 factors. `RandomProposer` is the version that makes random formulas instead, used in Step 8."""))

cells.append(module_cell("src/stage2/llm.py", "Section 5a: Writing the instructions and asking Claude"))

cells.append(md("""Before suggesting new factors, Claude writes a few lines about what it learned from the last round. These are its own words, unedited."""))

cells.append(code("""# Section 5b: What Claude wrote each round, then its first 6 factors
for r in sorted(llm_board['round'].unique()):
    resp = json.loads((LLM_RUN / f'round_{r:02d}_response.json').read_text())
    print(f'Round {r}: {resp.get(\"reflection\", \"\")}\\n')

first = llm_board[llm_board['round'] == 1][['name', 'mode', 'formula', 'hypothesis']]
first"""))

cells.append(md("""## Step 6: Testing Each Factor and the Keep Rule

**Testing (Section 6a).** For every factor, in every year: train on Q1-Q2 with the 22 factors plus the new one, and score on Q3. Then compare with the 22 factors alone. Same models and seeds as Stage 1 (Logistic Regression seed 42, Random Forest seeds 0, 1, 2).

**The loop (Section 6b).** Builds each factor, tests it, applies the keep rule, and writes the scoreboard that goes back to Claude. A factor is kept only if:
* it helped in at least **3 of 4 years**
* the average gain is at least **0.01**
* it is not a copy of a factor I already have (similarity below **0.9**)"""))

cells.append(module_cell("src/stage2/scoring.py", "Section 6a: Testing a factor on Q3 of 2022-2025"))

cells.append(module_cell("src/stage2/loop.py", "Section 6b: The loop and the keep rule"))

cells.append(code("""# Section 6c: Re-run round 4 with the code above, using the answer Claude gave in that round
import tempfile

saved_answer = json.loads((LLM_RUN / 'round_04_response.json').read_text())
replay = Loop(proposer=lambda prompt: saved_answer,              # Claude's saved reply instead of a new API call
              raw=raw, features=features, prices=prices,
              out_dir=Path(tempfile.mkdtemp()),
              stage1_importance_path=Path('results/multi_year/permutation_importance_summary.csv'))
replay.board = llm_board[llm_board['round'] < 4].to_dict('records')   # what had happened in rounds 1-3
replay.run_round(4)

round_4 = pd.DataFrame(replay.board[-6:])[['name', 'auc_gain', 'years_helped', 'importance', 'max_corr', 'status', 'reason']]
round_4.round(4)"""))

cells.append(md("""**Reading this output**

* This is round 4 run again with the code above, using Claude's saved answer.
* `nvda_range_vs_aapl_range` helped in 3 of 4 years by about 0.012 on average, so it is **kept**. The other 5 are rejected, each with its reason.
* All 6 factors get the same keep or reject decision as in the real run (next table). The numbers can differ slightly, because the run was done on a different computer with a slightly different version of scikit-learn."""))

cells.append(md("""**The full scoreboard of the real run** (all 8 rounds):

* **auc_gain**: how much the score went up (positive) or down (negative), averaged over the 4 years and both models
* **years**: in how many of the 4 years it helped
* **max_corr**: how similar it is to a factor I already have (0.9 or more means it is basically a copy)"""))

cells.append(code("""# Section 6d: Every factor Claude suggested and how it scored
show = llm_board.assign(years=llm_board.years_helped.astype(str) + '/4')[
    ['round', 'name', 'mode', 'auc_gain', 'years', 'max_corr', 'status', 'reason']]
print(llm_board.status.value_counts().to_string(), '\\n')
show.sort_values(['status', 'auc_gain'], ascending=[True, False]).round(4)"""))

cells.append(md("""**Reading this output**

* Claude suggested **48 factors** over 8 rounds, and every formula was valid.
* **47 were thrown away and 1 was kept.** Most only helped in 1 or 2 years, or the gain was too small to tell apart from luck.
* 4 were thrown away because they were almost the same as a factor I already had. For example, one was 0.95 similar to `MSFT_return_1d`."""))

cells.append(md("""## Step 7: Why the Keep Rule Is Strict

One quarter is only about 63 days, so some factors will look useful just by luck.

To see how much luck can do, I tested **32 fake factors made of random numbers**. They contain no information at all.
* If the rule was only "helps in 3 of 4 years", **6 of the 32** fake factors would pass.
* Also needing the average gain to be at least 0.01, only **1 of 32** passes.

So anything I keep has to beat what random numbers can do. The test takes about 10 minutes, so it is switched off below and the saved result is used."""))

cells.append(code("""# Section 7a: The fake-factor test
RUN_FAKE_FACTOR_TEST = False      # True = run it again (about 10 minutes)

def fake_factor_test(n=32):
    scorer = Scorer(features)
    dates = pd.to_datetime(features['Date'])
    rows = []
    for i in range(n):
        fake = pd.Series(np.random.default_rng(i).normal(size=len(dates)), index=dates.values)   # pure random numbers
        rows.append(summarise(scorer.score(f'noise{i}', fake)).reset_index())
    return pd.concat(rows, ignore_index=True)

noise = fake_factor_test() if RUN_FAKE_FACTOR_TEST else pd.read_csv('results/stage2/noise_null.csv')
for gain in (0, 0.01):
    passed = ((noise.years_helped >= 3) & (noise.auc_gain >= gain)).sum()
    print(f'Fake factors passing \"3 of 4 years and gain at least {gain}\": {passed} of {len(noise)}')"""))

cells.append(code("""# Section 7b: Claude's factors vs fake factors made of random numbers
scored = llm_board[llm_board.status != 'invalid']

fig, ax = plt.subplots(figsize=(8, 3.2))
bins = np.linspace(-0.04, 0.04, 33)
ax.hist(scored.auc_gain.clip(-0.04, 0.04), bins=bins, weights=np.full(len(scored), 100 / len(scored)), color='#2f6db5', alpha=0.8, label=f'Claude factors ({len(scored)})')
ax.hist(noise.auc_gain, bins=bins, weights=np.full(len(noise), 100 / len(noise)), histtype='step', lw=2, color='#555', label='Random numbers (32)')
ax.axvline(0.01, color='#333', lw=1.2, ls='--')
ax.text(0.0105, ax.get_ylim()[1] * 0.9, 'keep line 0.01', fontsize=9)
ax.set_xlabel('Average validation AUC gain'); ax.set_ylabel('% of factors')
ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False)
plt.tight_layout(); plt.show()
print(f'Random numbers above 0.01: {(noise.auc_gain >= 0.01).mean():.0%} | Claude above 0.01: {(scored.auc_gain >= 0.01).mean():.0%}')"""))

cells.append(md("""**Reading this output**

* The blue bars (Claude's factors) and the grey outline (random numbers) sit mostly in the same place. Most single factors don't help much, which shows how hard this problem is.
* Only factors to the right of the dashed line can be kept. One of Claude's factors made it there and also helped in 3 of the 4 years.
* Without this rule, I would have kept factors that only looked good by luck."""))

cells.append(md("""## Step 8: Is Claude Better Than Random Guessing?

To be fair, I ran the exact same loop again, but this time the factors were random formulas (`RandomProposer`, Step 5) instead of Claude's. Same number of rounds, same tests, same keep rule.

If Claude is actually useful, its factors should score better than the random ones."""))

cells.append(code("""# Section 8: Claude vs random formulas, round by round
from matplotlib.ticker import MaxNLocator

def per_round(board):
    s = board[board.status != 'invalid']
    g = s.groupby('round').agg(mean_gain=('auc_gain', 'mean'), best_gain=('auc_gain', 'max'))
    g['kept_total'] = board.assign(k=board.status.eq('kept')).groupby('round').k.sum().cumsum()
    return g

llm_r, rnd_r = per_round(llm_board), per_round(rnd_board)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for g, name, col in [(llm_r, 'Claude', '#2f6db5'), (rnd_r, 'Random', '#e08a2b')]:
    axes[0].plot(g.index, g.mean_gain, marker='o', color=col, lw=2, label=name)
    axes[1].step(g.index, g.kept_total, where='post', color=col, lw=2, label=name)
axes[0].axhline(0, color='#999', lw=0.8)
axes[0].set_title('Average AUC gain of the round\\'s factors'); axes[0].set_xlabel('Round')
axes[1].set_title('Factors kept so far'); axes[1].set_xlabel('Round')
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout(); plt.show()

summary = pd.DataFrame({
    name: {'factors tried': len(b), 'skipped (broken or repeated)': int(b.status.eq('invalid').sum()),
           'average AUC gain': round(b.loc[b.status != 'invalid', 'auc_gain'].mean(), 4),
           'helped 3+ of 4 years': int((b.years_helped >= 3).sum()),
           'kept': int(b.status.eq('kept').sum())}
    for name, b in [('Claude', llm_board), ('Random', rnd_board)]})
summary"""))

cells.append(md("""**Reading this output**

* **Claude learns from the feedback.** Its average score went from negative in rounds 1-2 to positive in rounds 3-5. After it found the NVIDIA-vs-Apple range factor in round 4, it spent the next rounds trying versions of the same idea. One version helped in all 4 years but was just under the 0.01 line.
* **Claude's factors were a bit better on average** (-0.0006 vs -0.0020), and every one of Claude's 48 formulas was new and usable, while 9 of the random ones were repeats of earlier formulas and had to be skipped.
* **Both runs kept 1 factor.** The random run got lucky once, which fits the fake-factor test from Step 7.
* So on Q3 alone, Claude is only slightly ahead. The real comparison is the final test in Step 10."""))

cells.append(md("""## Step 9: The Factor Claude Found"""))

cells.append(code("""# Section 9: The kept factor, Claude's reason for it, and its score in each year
by_year = pd.read_csv(LLM_RUN / 'scores_by_year.csv')
if llm_kept:
    kept_names = [k['name'] for k in llm_kept]
    table = by_year[by_year.factor.isin(kept_names)].pivot_table(index='factor', columns='year', values='auc_gain').round(4)
    table['average'] = table.mean(axis=1).round(4)
    display(pd.DataFrame(llm_kept)[['name', 'round', 'formula', 'hypothesis']])
    display(table)
else:
    print('No Claude factor met the selection rule.')"""))

cells.append(md("""**Reading this output**

* **`nvda_range_vs_aapl_range`** = how much NVIDIA moved between its daily high and low, minus how much Apple did (both as % of price).
* Claude's reasoning: when NVIDIA is much jumpier than Apple, something is happening in tech stocks that hasn't reached Apple yet.
* It helped in 2023, 2024 and 2025, and hurt a little in 2022.
* It is nothing like my 22 factors (only 0.15 similar), so it is new information, not a copy."""))

cells.append(md("""## Step 10: The Final Test on Q4

This is the first and only time Q4 is used. For each year I train on Q1-Q2 twice:
* with my 22 Stage 1 factors
* with the 22 factors plus the kept factor

Then I predict Q4 and run my Stage 1 backtest (hold Apple only on days the model says "up"). I do the same for the random run's factor so the comparison is fair."""))

cells.append(module_cell("src/stage2/final_test.py", "Section 10a: The final test"))

cells.append(code("""# Section 10b: Run the final test for Claude's factor and the random run's factor
results = []
for name, kept in [('Claude', llm_kept), ('Random', rnd_kept)]:
    if kept:
        m, eq = run_final_test(features, prices, raw, kept)
        results.append(m.assign(source=name))
final = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

if len(final):
    base = final[final.features == 'stage1_22'].drop_duplicates(['year', 'model']).assign(source='Stage 1 (22 factors)')
    plus = final[final.features == 'stage1_plus_llm'].assign(source=lambda d: d.source + ' factors added')
    view = pd.concat([base, plus])
    ba = view.pivot_table(index=['model', 'source'], columns='year', values='balanced_accuracy').round(3)
    ba['average'] = ba.mean(axis=1).round(3)
    ret = view.pivot_table(index=['model', 'source'], columns='year', values='strategy_return').round(3)
    ret.loc[('buy and hold', 'Apple'), :] = final.groupby('year').buy_hold_return.first().round(3)
    print('Test balanced accuracy (0.5 = coin flip)'); display(ba)
    print('Backtest return in Q4'); display(ret)"""))

cells.append(md("""**Reading this output**

* **Logistic Regression:** adding Claude's factor raised the test score from **0.448 to 0.513** on average, and it was better in **all 4 years**. The random factor got 0.497.
* **Random Forest:** **0.490 to 0.500**, better in 3 of 4 years. The random factor made it slightly worse (0.487).
* **Backtest:** with Claude's factor, Logistic Regression made more money than the Stage 1 model in every year, and beat simply holding Apple in 2023 (+13.8% vs +11.0%).
* **To be careful:** this is one factor from one run, and each test quarter is only 63 days. 2025 is still below 0.5."""))

cells.append(code("""# Section 10c: Test score with and without the new factor
if len(final):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, model in zip(axes, ['logistic_regression', 'random_forest']):
        d = view[view.model == model]
        for src, col in [('Stage 1 (22 factors)', '#b9bec7'), ('Claude factors added', '#2f6db5'), ('Random factors added', '#e08a2b')]:
            s = d[d.source == src].sort_values('year')
            if len(s):
                ax.plot(s.year, s.balanced_accuracy, marker='o', lw=2, color=col, label=src)
        ax.axhline(0.5, color='#999', lw=0.8, ls='--')
        ax.set_title(model.replace('_', ' ').title()); ax.set_xticks([2022, 2023, 2024, 2025])
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Test balanced accuracy'); axes[0].legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.show()"""))

cells.append(md("""## Step 11: Running the Whole Loop Live

This cell runs all the rounds with Claude, using the code above. It is switched off so the notebook doesn't call Claude every time it is opened. To run it: set `RUN_LIVE = True`, and set your API key first (`export ANTHROPIC_API_KEY=...` in Terminal before starting Jupyter). 8 rounds take about 6 minutes and cost well under US$1. Every prompt and reply is saved in a new folder in `results/stage2/runs/`.

The same loop can also be started from Terminal with `python experiments/run_stage2_loop.py --rounds 8`."""))

cells.append(code("""# Section 11: Run the loop live
RUN_LIVE = False                   # True = ask Claude for real (needs ANTHROPIC_API_KEY)
ROUNDS = 8
USE_CLAUDE = True                  # False = random formulas instead of Claude

def run_loop(rounds=ROUNDS, use_claude=USE_CLAUDE):
    name = 'claude' if use_claude else 'random'
    out = Path('results/stage2/runs') / f'{time.strftime(\"%Y%m%d_%H%M\")}_{name}'
    proposer = ClaudeProposer('claude-sonnet-5-5') if use_claude else RandomProposer(0)
    loop = Loop(proposer=proposer, raw=raw, features=features, prices=prices, out_dir=out,
                stage1_importance_path=Path('results/multi_year/permutation_importance_summary.csv'))
    (out / 'config.json').write_text(json.dumps({'rounds': rounds, 'proposer': name, 'model': 'claude-sonnet-5-5',
                                                 'started': time.strftime('%Y-%m-%dT%H:%M:%S')}, indent=2))
    for r in range(1, rounds + 1):
        print(f'Round {r}')
        loop.run_round(r)
    print(f'Done. {len(loop.kept)} factor(s) kept. Saved in {out}')
    return loop

if RUN_LIVE:
    live = run_loop()
else:
    print('RUN_LIVE is False: using the saved run', LLM_RUN.name)"""))

cells.append(md("""## Summary

**Can an LLM find factors that do better than mine?** From this first run, the answer is **yes, a little.**

* The loop runs completely on its own: Claude suggests, my code tests, and the results go back to Claude. 8 rounds and 48 factors cost well under US$1.
* It is safe: the formula code matches Stage 1 exactly, no factor can see the future, and Q4 was never touched during the loop.
* Claude did learn from the feedback. Once it found the NVIDIA-vs-Apple range idea, it kept building on it.
* On the final test, Claude's factor improved both models, and did better than the factor found by random guessing.

**What I can't claim yet:** it is only one factor from one run, most of Claude's ideas didn't help, and the improvement is small.

**Next steps:**
* Run the loop and the random version a few more times, to check Claude's lead isn't luck
* Run more rounds, and add each kept factor to the base set so the next ones have to beat it
* Give Claude more detail in the feedback, such as which years each factor failed in"""))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, ROOT / "Stage 2_Demo.ipynb")
print("written")
