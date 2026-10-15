"""Builds 'Stage 2_Demo.ipynb'. Run: python experiments/build_stage2_notebook.py"""

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# Stage 2: Automated LLM Factor Discovery Demo

**PG-S2-36 - Automatic Discovery of Interpretable Financial Factors**

Abu Wakkas Bastob

**The research question (Dr Liu):** can an LLM, on its own, propose factors that improve on our 22 Stage 1 factors?

This notebook shows the fully automated loop and its results:
* What the LLM is told
* What it proposed, and why
* How each factor is built and scored
* How the selection rule works
* Whether it learns over rounds
* LLM compared with a random-formula baseline
* One final test on the locked Q4 quarters"""))

cells.append(code("""# Section 1: Setup - point to the two loop runs
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 120)

RUNS = sorted(Path('results/stage2/runs').glob('*'))
LLM_RUN = [r for r in RUNS if r.name.endswith('_claude')][-1]      # latest Claude run
RANDOM_RUN = [r for r in RUNS if r.name.endswith('_random')][-1]   # latest random baseline

def load(run):
    board = pd.read_csv(run / 'scoreboard.csv').fillna({'reason': '', 'parent': '', 'closest': ''})
    kept = json.loads((run / 'kept_factors.json').read_text())
    return board, kept

llm_board, llm_kept = load(LLM_RUN)
rnd_board, rnd_kept = load(RANDOM_RUN)
config = json.loads((LLM_RUN / 'config.json').read_text())
print('Claude run :', LLM_RUN.name, '| model:', config.get('model'), '| rounds:', llm_board['round'].max())
print('Random run :', RANDOM_RUN.name, '| rounds:', rnd_board['round'].max())"""))

cells.append(md("""## Step 2: The Loop

```
 ┌──────────────► 1. PROPOSE   Claude gets the brief + scoreboard, writes factor formulas
 │                     │
 │                2. BUILD     my formula code turns each formula into a daily column (past data only)
 │                     │
 │                3. SCORE     add it to the 22 Stage 1 factors, retrain, score on Q3 of 2022-2025
 │                     │
 │                4. SELECT    keep only if it helps in 3+ of 4 years, by more than noise, and is not a copy
 │                     │
 └─────────────── 5. FEEDBACK  every score (and every error) goes back to Claude next round
```

* Each round Claude proposes **3 variants** (improve something that scored well) and **3 new ideas** (a different family).
* Everything is automatic through the Claude API. No human picks factors.
* The **Q4 test quarters are never used inside the loop.**"""))

cells.append(md("""## Step 3: The Formula Language (how a recipe becomes a number)

Claude writes each factor as nested functions, for example `subtract(pct_change(AAPL_Close, 5), pct_change(QQQ_Close, 5))`
= "Apple's 5-day return minus the market's 5-day return".

My code checks it (only allowed inputs, operators and windows, nothing that can look forward) and computes it.

**Two proofs it is correct:**
1. Written in this language, the 22 Stage 1 factors come out **exactly** the same as in Stage 1.
2. Deleting all data after a day never changes that day's value, so **no future data** is used."""))

cells.append(code("""# Section 3: Check the formula language
from src.features import build_stage1_features
from src.stage2.context import BASE_FORMULAS
from src.stage2.formula import evaluate, validate, wide_prices, FormulaError

raw = pd.read_csv('data/processed/aligned_daily_ohlcv_5y.csv')
prices = wide_prices(raw)
features = build_stage1_features(raw)
stage1 = features.set_index(pd.to_datetime(features['Date']))

# Proof 1: rebuild the 22 Stage 1 factors from formulas
same = {name: np.allclose(evaluate(f, prices).reindex(stage1.index), stage1[name]) for name, f in BASE_FORMULAS.items()}
print(f'Proof 1: {sum(same.values())} of {len(same)} Stage 1 factors reproduced exactly')

# Proof 2: no look-ahead, on every factor Claude proposed
day = pd.Timestamp('2024-03-15')
formulas = llm_board.loc[llm_board.status != 'invalid', 'formula']
changed = sum(not np.isclose(evaluate(f, prices.loc[:day]).iloc[-1], evaluate(f, prices).loc[day], equal_nan=True)
              for f in formulas)
print(f'Proof 2: {len(formulas)} Claude factors recomputed with data cut at {day.date()}: {changed} changed')

# The checker rejects unsafe formulas
for bad in ['lag(AAPL_Close, -1)', 'AAPL_Close + QQQ_Close', 'pct_change(TSLA_Close, 1)']:
    try:
        validate(bad)
    except FormulaError as e:
        print(f'Rejected  {bad:28} -> {e}')"""))

cells.append(md("""## Step 4: What Claude Is Told

The brief has: the goal, the data, how scoring works, the 22 existing factors as formulas, the most useful Stage 1 factors,
a market summary **up to 30 June 2025 only** (nothing from any 2025 validation or test quarter), and the full scoreboard."""))

cells.append(code("""# Section 4: The prompt Claude received in the last round (start and scoreboard part)
last = llm_board['round'].max()
prompt = (LLM_RUN / f'round_{last:02d}_prompt.txt').read_text()
print(prompt[:1400])
print('\\n.........\\n')
start = prompt.index('SCOREBOARD OF EVERYTHING TRIED SO FAR')
print(prompt[start:start + 1600])"""))

cells.append(md("""## Step 5: What Claude Proposed

Each round Claude first writes a short **reflection** on the scoreboard, then its factors with a one-line hypothesis."""))

cells.append(code("""# Section 5: Claude's reflection every round, then its first-round proposals
for r in sorted(llm_board['round'].unique()):
    resp = json.loads((LLM_RUN / f'round_{r:02d}_response.json').read_text())
    print(f'Round {r}: {resp.get(\"reflection\", \"\")}\\n')

first = llm_board[llm_board['round'] == 1][['name', 'mode', 'formula', 'hypothesis']]
first"""))

cells.append(md("""## Step 6: How Every Factor Scored

For each factor and each year (2022-2025): train on Q1-Q2 with the 22 factors **plus** the new one, score on Q3,
and compare with the 22 alone.

* **auc_gain**: change in validation AUC (0.5 = coin flip), averaged over Logistic Regression and Random Forest and the 4 years
* **years**: in how many of the 4 years it helped
* **corr**: highest correlation with an existing factor (0.9+ means it's a copy)"""))

cells.append(code("""# Section 6: The full scoreboard of the Claude run
show = llm_board.assign(years=llm_board.years_helped.astype(str) + '/4')[
    ['round', 'name', 'mode', 'auc_gain', 'years', 'max_corr', 'status', 'reason']]
print(llm_board.status.value_counts().to_string(), '\\n')
show.sort_values(['status', 'auc_gain'], ascending=[True, False]).round(4)"""))

cells.append(md("""## Step 7: Why the Selection Rule Is Strict (the noise test)

Single-quarter scores are noisy, so some factors look useful by luck. To measure luck, I scored **32 pure-noise factors**
(random numbers with no information).

* With the easy rule "helps in 3 of 4 years", **6 of 32** noise factors passed.
* Adding **"average gain at least 0.01"**, only **1 of 32** passes.

So a factor is kept only if it beats what noise can do."""))

cells.append(code("""# Section 7: Claude's factors compared with pure noise
noise = pd.read_csv('results/stage2/noise_null.csv')
scored = llm_board[llm_board.status != 'invalid']

fig, ax = plt.subplots(figsize=(8, 3.2))
bins = np.linspace(-0.04, 0.04, 33)
ax.hist(noise.auc_gain, bins=bins, color='#b9bec7', label='Pure noise (32)')
ax.hist(scored.auc_gain.clip(-0.04, 0.04), bins=bins, color='#2f6db5', alpha=0.85, label=f'Claude factors ({len(scored)})')
ax.axvline(0.01, color='#333', lw=1.2, ls='--')
ax.text(0.0105, ax.get_ylim()[1] * 0.9, 'keep line 0.01', fontsize=9)
ax.set_xlabel('Average validation AUC gain'); ax.set_ylabel('Number of factors')
ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False)
plt.tight_layout(); plt.show()
print(f'Noise above 0.01: {(noise.auc_gain >= 0.01).mean():.0%} | Claude above 0.01: {(scored.auc_gain >= 0.01).mean():.0%}')"""))

cells.append(md("""## Step 8: Does Claude Learn? (Claude vs random formulas)

To check that the LLM adds real value, the **same loop** was run with a **random-formula generator** instead of Claude:
same rounds, same scoring, same rule. If Claude is useful, its factors should score better and more should be kept."""))

cells.append(code("""# Section 8: Claude vs random, round by round
def per_round(board):
    s = board[board.status != 'invalid']
    g = s.groupby('round').agg(mean_gain=('auc_gain', 'mean'), best_gain=('auc_gain', 'max'))
    g['kept_total'] = board.assign(k=board.status.eq('kept')).groupby('round').k.sum().cumsum()
    g['invalid'] = board.groupby('round').apply(lambda x: x.status.eq('invalid').sum())
    return g

llm_r, rnd_r = per_round(llm_board), per_round(rnd_board)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for g, name, col in [(llm_r, 'Claude', '#2f6db5'), (rnd_r, 'Random', '#e08a2b')]:
    axes[0].plot(g.index, g.mean_gain, marker='o', color=col, lw=2, label=name)
    axes[1].step(g.index, g.kept_total, where='post', color=col, lw=2, label=name)
    axes[1].text(g.index[-1] + 0.1, g.kept_total.iloc[-1], name, color='#333', va='center', fontsize=9)
axes[0].axhline(0, color='#999', lw=0.8)
axes[0].set_title('Average AUC gain of the round\\'s factors'); axes[0].set_xlabel('Round')
axes[1].set_title('Factors kept so far'); axes[1].set_xlabel('Round')
from matplotlib.ticker import MaxNLocator
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout(); plt.show()

summary = pd.DataFrame({
    name: {'factors tried': len(b), 'invalid formulas': int(b.status.eq('invalid').sum()),
           'average AUC gain': round(b.loc[b.status != 'invalid', 'auc_gain'].mean(), 4),
           'helped 3+ of 4 years': int((b.years_helped >= 3).sum()),
           'kept': int(b.status.eq('kept').sum())}
    for name, b in [('Claude', llm_board), ('Random', rnd_board)]})
summary"""))

cells.append(md("""## Step 9: The Factors Claude Kept"""))

cells.append(code("""# Section 9: Kept factors with Claude's own reasoning and their score in each year
by_year = pd.read_csv(LLM_RUN / 'scores_by_year.csv')
if llm_kept:
    kept_names = [k['name'] for k in llm_kept]
    table = by_year[by_year.factor.isin(kept_names)].pivot_table(index='factor', columns='year', values='auc_gain').round(4)
    table['average'] = table.mean(axis=1).round(4)
    display(pd.DataFrame(llm_kept)[['name', 'round', 'formula', 'hypothesis']])
    display(table)
else:
    print('No Claude factor met the selection rule.')"""))

cells.append(md("""## Step 10: The Final Test (Q4, used once)

Now, and only now, the locked Q4 quarters are used. For each year: train on Q1-Q2 with
**(a)** the 22 Stage 1 factors and **(b)** the 22 plus the kept factors, predict Q4, and backtest
(hold Apple only on days the model says "up")."""))

cells.append(code("""# Section 10: One-time Q4 test for the Claude factors and the random-baseline factors
from src.stage2.final_test import run_final_test

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

cells.append(code("""# Section 10b: Test balanced accuracy, Stage 1 vs with the kept factors
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

cells.append(md("""## Summary

*(Filled in after the runs.)*"""))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "Stage 2_Demo.ipynb")
print("written")
