"""Builds 'Stage 2_v2_Demo.ipynb'. Run: python experiments/build_stage2_v2_notebook.py

Version 2 of Stage 2 (Dr Liu, 20 Oct): factors are tested together, not only one at a time.
The new code is written into the notebook cells from src/stage2/*.py when the notebook is built,
so the notebook and the Terminal commands run exactly the same code. The parts that did not change
since 15 Oct are imported and are shown in full in 'Stage 2_Demo.ipynb'.
The "Reading this output" notes are kept in experiments/stage2_v2_notes.json, written after the runs.
"""

import ast
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTES_FILE = ROOT / "experiments/stage2_v2_notes.json"
NOTES = json.loads(NOTES_FILE.read_text()) if NOTES_FILE.exists() else {}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []


def note(key):
    """A 'Reading this output' cell, if a note for this step has been written."""
    if key in NOTES:
        cells.append(md("**Reading this output**\n\n" + NOTES[key]))


def module_cell(path, title):
    """The code of one src/stage2 file as a notebook cell (its description becomes comments,
    imports between the Stage 2 files are dropped because those names are defined in earlier cells)."""
    text = (ROOT / path).read_text()
    tree = ast.parse(text)
    lines = text.splitlines()
    doc = ast.get_docstring(tree)
    start = tree.body[0].end_lineno if doc else 0
    body = [l for l in lines[start:] if not l.startswith("from __future__") and not l.startswith("from .")]
    while body and not body[0].strip():
        body.pop(0)
    header = [f"# {title}  (also saved as {path})"]
    if doc:
        header += ["#"] + [("# " + l).rstrip() for l in doc.splitlines()]
    return code("\n".join(header + [""] + body).rstrip())


cells.append(md("""# Stage 2, Version 2: Testing Factors Together

**PG-S2-36 - Automatic Discovery of Interpretable Financial Factors**

Abu Wakkas Bastob

Last week's loop judged every factor **on its own**: a factor was kept only if it helped by itself. Dr Liu pointed out (20 Oct) that this throws away factors that only work **together with other factors**. He asked for:

1. **More detailed feedback to Claude** for every factor: the formula, how it did in each year, Logistic Regression vs Random Forest, kept or rejected, how similar it is to other factors, and a short reason why
2. **A team trial**: test each new factor on top of the factors already selected, not only on its own
3. **Pair trials** for promising factors that may work well together
4. **Telling Claude which factors work together**, and asking it for partners
5. **A comparison with last week's one-at-a-time approach**

In his 21 Oct email he also suggested generating a large set of factors and using **feature selection** to find a good combination (Step 11).

This notebook shows the new version, what the results taught me about the data (Steps 9 and 10), and the comparison (Step 12). The new code is written out in the cells below. The parts that did not change since last week (the formula language, the information about the data, the call to Claude and the final test) are imported, and are shown in full in `Stage 2_Demo.ipynb`."""))

cells.append(code("""# Section 1: Setup, the data, and the three runs I compare
import sys, json, re, time, random, warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, '.')                 # my Stage 1 code in src/ (features, split, models, backtest)
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 110)

from src.features import build_stage1_features
# unchanged since 15 Oct (shown in full in 'Stage 2_Demo.ipynb'):
from src.stage2.formula import FormulaError, INPUTS, WINDOWS, evaluate, operator_guide, validate, wide_prices
from src.stage2.context import BASE_FORMULAS, market_summary, stage1_importance
from src.stage2.llm import ClaudeProposer, RandomProposer, dump
from src.stage2.final_test import run_final_test

raw = pd.read_csv('data/processed/aligned_daily_ohlcv_5y.csv')   # same data as Stage 1
features = build_stage1_features(raw)                              # my 22 Stage 1 factors + the target
prices = wide_prices(raw)

RUN_A = Path('results/stage2/runs/20261015_1635_claude')           # A: last week, every factor judged alone
V2_RUNS = sorted(Path('results/stage2/runs_v2').glob('*'))
RUN_B = [r for r in V2_RUNS if '_replay_' in r.name][-1]            # B: last week's 48 factors, judged the new way
RUN_C = [r for r in V2_RUNS if r.name.endswith('_claude') and '_replay_' not in r.name][-1]      # C: the new loop with Claude

a_board = pd.read_csv(RUN_A / 'scoreboard.csv').fillna({'reason': '', 'closest': ''})
a_kept = json.loads((RUN_A / 'kept_factors.json').read_text())
print('A:', RUN_A.name, '|', len(a_board), 'factors tried |', len(a_kept), 'kept')
for name, run in [('B', RUN_B), ('C', RUN_C)]:
    n = len(pd.read_csv(run / 'scoreboard.csv'))
    team = json.loads((run / 'kept_factors.json').read_text())
    print(f'{name}:', run.name, '|', n, 'factors tried |', len(team), 'in the selected team')"""))

cells.append(md("""## Step 2: What Changed

| | Last week (A) | This week (C) |
|---|---|---|
| What Claude sees | a scoreboard line per factor: average gain, years helped | a **report card** per factor: gain in each year, LR vs RF, similarity, kept or not and why, and its own lesson |
| How a factor is tested | **alone**: added to my 22 factors | alone, then **on top of the selected team**, and **in pairs** |
| Who gets kept | factors that help alone | factors that make the team better, alone or as a pair |
| Weak-alone factors | thrown away | kept on a **promising list** for pair trials |
| What Claude is asked for | 3 variants + 3 new ideas | 2 **improve** + 2 **new** + 2 **partner** factors, plus a lesson for every factor of the last round |

Everything else is the same as last week: the formula language (past data only), the four years 2022-2025 (train on Q1-Q2, check on Q3), Logistic Regression and Random Forest, and Q4 is only used once, in the final test.

**One round of the new loop:**
```
Claude proposes 6 factors (2 improve, 2 new, 2 partner)
        |
check the formula and compute it (past data only)
        |
1. alone:        added to my 22 factors                      (same as last week)
2. team trial:   added to my 22 factors + the selected team  (new)
3. pair trials:  two promising factors added together        (new)
        |
joins the team  /  stays on the promising list  /  rejected
        |
a report card for every factor goes back to Claude in the next round
```"""))

cells.append(md("""## Step 3: The Report Card Claude Gets

Every factor Claude has tried gets a few lines like the ones below. The "why" line is written automatically from the numbers, so it never contains anything that wasn't measured. In the next round Claude has to write its own one-line lesson about each factor, and that lesson is shown on the card from then on."""))

cells.append(module_cell("src/stage2/feedback.py", "Section 3a: The report card"))

cells.append(code("""# Section 3b: Report cards from the new run (C): a team member, a promising one and a rejected one
c_records = board_records(pd.read_csv(RUN_C / 'scoreboard.csv'))
examples = []
for status in ['kept', 'promising', 'rejected']:
    with_lesson = [r for r in c_records if r['status'] == status and r['lesson']]
    found = with_lesson or [r for r in c_records if r['status'] == status]
    if found:
        examples.append(found[0])
for r in examples:
    print(report_card(r), end='\\n\\n')"""))
note("cards")

cells.append(md("""## Step 4: What I Tell Claude Now

The instructions have the same parts as last week (goal, data, the 22 factors, formula language), plus three new parts: **the selected team**, **the promising list** and **the pair trials so far**. The scoreboard is replaced by the report cards. The task now asks for improve, new and partner factors, and a lesson for every factor of the last round."""))

cells.append(module_cell("src/stage2/prompt_v2.py", "Section 4a: The new instructions for Claude"))

cells.append(code("""# Section 4b: The new parts of the instructions Claude got in round 4 of run C
prompt = (RUN_C / 'round_04_prompt.txt').read_text()
start, end = prompt.index('THE SELECTED TEAM'), prompt.index('YOUR LAB NOTEBOOK')
print(prompt[start:end])
print(prompt[prompt.index('TASK'):])"""))

cells.append(code("""# Section 4c: What Claude wrote back in round 4: its reflection, its lessons, and the factors it proposed
reply = json.loads((RUN_C / 'round_04_response.json').read_text())
print('REFLECTION:', reply['reflection'], '\\n')
print('LESSONS ABOUT ROUND 3:')
for item in reply.get('lessons', []):
    print(f"- {item['name']}: {item['lesson']}")
print('\\nFACTORS PROPOSED:')
for f in reply['factors']:
    parent = f" (parent: {f['parent']})" if f.get('parent') else ''
    print(f"- [{f['mode']}{parent}] {f['name']} | role: {f['role']}\\n    {f['formula']}\\n    {f['hypothesis']}")"""))
note("prompt")

cells.append(md("""## Step 5: Team Trial and Pair Trial

The scoring code from last week now takes two inputs: the factor(s) to add, and the team they are added on top of.
* **Alone:** add one factor to my 22 factors (exactly last week's test; the numbers are identical)
* **Team trial:** add one factor to my 22 factors plus the selected team
* **Pair trial:** add two factors at once to my 22 factors plus the selected team

The gain is always "check score (ROC AUC on Q3) with the new factor(s) minus without", for each of the four years, averaged over Logistic Regression and the three Random Forests."""))

cells.append(module_cell("src/stage2/scoring.py", "Section 5a: Testing factors alone, on top of the team, or in a pair"))
cells.append(module_cell("src/stage2/selection.py", "Section 5b: The rules for the team, the pairs and the promising list"))

cells.append(md("""The loop itself. One round: ask Claude, check and compute each formula, test each factor alone, run the team trial, run a few pair trials, then update the team, the promising list and the report cards."""))

cells.append(module_cell("src/stage2/curve.py", "Section 5c: Grading on a curve (tried in Step 10, not used)"))
cells.append(module_cell("src/stage2/loop_v2.py", "Section 5d: The new loop"))

cells.append(md("""## Step 6: Setting the New Bars with Random Numbers

Same idea as last week's fake-factor test: before any real run, I put factors made of **pure random numbers** through the new tests and counted how often they would get into the team by luck. The bars were fixed before the real runs and are the same for B and C. This check found three problems, and each one changed the design:

1. **Lucky pairs.** My first pair rule was "the pair must beat the better of its two members alone". A random factor that looks promising was a little lucky on its own. Put two lucky ones together and their luck simply adds up, so the pair always looks better than either one: 42% of promising random pairs passed. The rule now is: **the pair's gain must be clearly above the two gains added up.** Luck adds up; real teamwork gives more than the sum.
2. **The forest changed by itself.** With its default setting, the Random Forest looks at 4 factors at each split when it has up to 24 factors, but at 5 from 25 factors on. So on a team of 2, adding *any* extra column changed the forest itself, and 8 of 18 promising random factors passed the team trial. Now the forest always looks at 4 factors per split. For my 22 factors and for every single-factor test this is exactly what it did before, so last week's numbers do not change.
3. **Second chances.** I had planned to give every factor on the promising list one more team trial at the end. Even after fix 2, about 1 in 18 promising random factors got in that way, so this step is switched off."""))

cells.append(code("""# Section 6: How often random numbers would get into the team
noise_team = pd.read_csv('results/stage2/noise_null_v2_team.csv', index_col=0)
noise_pairs = pd.read_csv('results/stage2/noise_null_v2_pairs.csv')
rules = json.loads(Path('results/stage2/rules_v2.json').read_text())
print('Bars used in runs B and C:', rules, '\\n')

rows = []
for bar in [0.0, 0.005, rules['team_gain']]:
    share = ((noise_team.years >= 3) & (noise_team.gain >= bar)).mean()
    rows.append({'test': f'team trial ({len(noise_team)} random factors)',
                 'rule': f'helps 3+ of 4 years, gain >= {bar}', 'random factors getting in': f'{share:.0%}'})
for kind in ['random', 'promising']:
    d = noise_pairs[noise_pairs.kind == kind]
    first = ((d.years >= 3) & (d.gain >= 0.015) & (d.extra_vs_better >= 0.005)).mean()
    final = ((d.years >= 3) & (d.gain >= rules['pair_gain']) & (d.extra >= rules['pair_synergy'])).mean()
    rows.append({'test': f'pair trial ({len(d)} {kind} pairs)', 'rule': 'first rule: beats the better one alone by 0.005',
                 'random factors getting in': f'{first:.0%}'})
    rows.append({'test': f'pair trial ({len(d)} {kind} pairs)',
                 'rule': f"final rule: gain >= {rules['pair_gain']}, and {rules['pair_synergy']} above the two added up",
                 'random factors getting in': f'{final:.0%}'})
second = pd.read_csv('results/stage2/noise_null_v2_second_chance.csv')
share = ((second.years >= 3) & (second.gain >= rules['team_gain'])).mean()
rows.append({'test': f'second chance ({len(second)} promising random factors)',
             'rule': 'team trial again, on a 2-factor team', 'random factors getting in': f'{share:.0%}'})
display(pd.DataFrame(rows))"""))
note("noise")

cells.append(md("""## Step 7: What Happened in the New Run (C)

8 rounds, 6 factors per round (48 in total, the same budget as last week)."""))

cells.append(code("""# Section 7a: Round by round: team size, promising list, pair trials
c_rounds = pd.read_csv(RUN_C / 'rounds.csv')
display(c_rounds[['round', 'team_size', 'joined', 'promising', 'pairs_tried', 'pairs_joined', 'mean_alone_gain', 'seconds']].round(4))

c_board = pd.read_csv(RUN_C / 'scoreboard.csv').fillna({'reason': '', 'lesson': '', 'how_selected': ''})
print('\\nStatus of all factors tried:', c_board.status.value_counts().to_dict())"""))

cells.append(code("""# Section 7b: The selected team: each factor's formula, Claude's reason, and how it got in
c_team = json.loads((RUN_C / 'kept_factors.json').read_text())
if not c_team:
    print('No factor met the team or pair rules in this run.')
for k in c_team:
    r = c_board.set_index('name').loc[k['name']]
    years = ', '.join(f"{y}: {r[f'gain_{y}']:+.3f}" for y in [2022, 2023, 2024, 2025])
    print(f"{k['name']}  (round {k['round']}, {k['how_selected']})")
    print(f"  formula: {k['formula']}")
    print(f"  Claude's reason: {k['hypothesis']}")
    print(f"  alone, by year: {years}")
    print(f"  {r['reason']}\\n")"""))

cells.append(code("""# Section 7c: All pair trials: the pair's gain, each one alone on the team, and the extra over the two added up
c_pairs = pd.read_csv(RUN_C / 'pairs.csv') if (RUN_C / 'pairs.csv').stat().st_size > 2 else pd.DataFrame()
if len(c_pairs):
    display(c_pairs[['round', 'a', 'b', 'group_a', 'group_b', 'gain', 'years', 'a_alone_on_team', 'b_alone_on_team', 'synergy', 'joined']]
            .rename(columns={'synergy': 'extra over the two added up'}).round(4))
else:
    print('No pair trials.')"""))
note("run_c")

cells.append(md("""## Step 8: Last Week's Factors, Judged the New Way (B)

To separate the effect of the new **judging** from the effect of the new **feedback to Claude**, I took the same 48 factors Claude proposed last week, in the same order, and put them through the new team trial, pair trials and promising list. No new Claude call is needed for this."""))

cells.append(code("""# Section 8: What the new rules pick from last week's 48 factors
b_board = pd.read_csv(RUN_B / 'scoreboard.csv').fillna({'reason': '', 'how_selected': ''})
b_team = json.loads((RUN_B / 'kept_factors.json').read_text())
print('Last week (A) kept:', [k['name'] for k in a_kept])
print('New rules (B) selected:')
for k in b_team:
    print(f"  {k['name']}  ({k['how_selected']})\\n    {k['formula']}")
print('\\nStatus of the 48 factors under the new rules:', b_board.status.value_counts().to_dict())"""))
note("run_b")

cells.append(md("""## Step 9: Why So Few Factors Pass: Each Year Reacts the Same Way to Almost Any Factor

Looking at the report cards, Claude noticed it first (round 2 of run C): *"every factor helped in 2023 and 2025 but hurt in 2024"*. Below I count, for each year, how many factors helped: Claude's from both weeks, and the template formulas made with no LLM at all."""))

cells.append(code("""# Section 9: Share of factors that helped in each year (checked on Q3)
def share_by_year(per_year):
    return per_year.groupby('year').apply(lambda d: (d.auc_gain > 0).mean())

a_years = pd.read_csv(RUN_A / 'scores_by_year.csv')
t_years = pd.read_csv('results/stage2/runs/20261015_1645_random/scores_by_year.csv')
gain_cols = [f'gain_{y}' for y in [2022, 2023, 2024, 2025]]
c_long = pd.read_csv(RUN_C / 'scoreboard.csv').query('status != \"invalid\"')[['name'] + gain_cols].melt(
    id_vars='name', value_vars=gain_cols, value_name='auc_gain')
c_long['year'] = c_long.variable.str[-4:].astype(int)
table = pd.DataFrame({'Claude, 15 Oct (48)': share_by_year(a_years[a_years.valid.astype(bool)]),
                      'Claude, 21 Oct (47)': share_by_year(c_long),
                      'Templates, no LLM (39)': share_by_year(t_years[t_years.valid.astype(bool)])})
display((table * 100).round(0).astype(int).astype(str) + '%')"""))
note("year_effect")

cells.append(md("""## Step 10: A Longer Measuring Stick: 8 Checks Instead of 4

Each year has only one checking quarter (Q3, about 63 trading days), so luck is almost as large as the effects we are looking for. I added a second, earlier check every year: train on Q1, check on Q2. Every factor now gets **8 checks instead of 4**. The Q3 checks are exactly the same as before.

(I also tried "grading on a curve", comparing each factor with a typical template factor in the same year. Random numbers then looked better than real price factors in the bad years, so the bar had to rise to 0.02 and nothing passed. I did not use it.)

As before, the bar was set with random numbers and template formulas first, before looking at Claude's factors."""))

cells.append(code("""# Section 10: The new bar, and Claude's factors measured with 8 checks
null8 = pd.read_csv('results/stage2/checks8_null.csv')
rules8 = json.loads(Path('results/stage2/rules_v2_8checks.json').read_text())
rows = []
for kind in ['noise', 'template']:
    x = null8[null8.kind == kind]
    for k in [6, 7]:
        rows.append({'factors': f'{kind} ({len(x)})', 'rule': f'helped >= {k} of 8 checks, gain >= {rules8["team_gain"]}',
                     'getting in': f"{((x.checks >= k) & (x.gain >= rules8['team_gain'])).mean():.0%}"})
display(pd.DataFrame(rows))
print('Bar used:', rules8['min_years'], 'of 8 checks and average gain >=', rules8['team_gain'])

real8 = pd.read_csv('results/stage2/checks8_real.csv').drop_duplicates('formula')
real8['week'] = real8.kind.map(lambda k: '15 Oct' if k.startswith('20261015') else '21 Oct')
print('\\nChecks helped (out of 8): Claude factors vs random numbers')
display(pd.DataFrame({'Claude factors': real8.checks.value_counts().sort_index(),
                      'random numbers': null8[null8.kind == 'noise'].checks.value_counts().sort_index()}).fillna(0).astype(int))
print('Claude factors with the highest average gain:')
display(real8.sort_values('gain', ascending=False).head(8)[['week', 'factor', 'gain', 'checks', 'lr', 'rf']].round(4))"""))
note("checks8")

cells.append(md("""## Step 11: Dr Liu's Suggestion: One Big Pile, Then Feature Selection

Dr Liu suggested (21 Oct email) generating a large set of factors with the LLM and then using feature selection to find a good combination. Every factor Claude has written so far goes into one pile, and **forward selection** picks the team: add the factor that improves the team most, if it passes the 8-check bar; re-test the rest on the bigger team; repeat. Then a few pairs are tried on top of the team."""))

cells.append(code("""# Section 11: Forward selection on the whole pile
RUN_P = Path('results/stage2/pile_8checks')
p_team = json.loads((RUN_P / 'kept_factors.json').read_text())
p_log = pd.read_csv(RUN_P / 'selection_log.csv')
p_pairs = pd.read_csv(RUN_P / 'pairs.csv')
print(f"Pile: {len(pd.read_csv('results/stage2/checks8_real.csv').drop_duplicates('formula'))} different factors,",
      f"{len(pd.read_csv(RUN_P / 'scoreboard.csv'))} after removing near-copies of my 22 factors")
for k in p_team:
    print(f"Selected: {k['name']}  ({k['how_selected']})\\n  {k['formula']}\\n  Claude's reason: {k['hypothesis']}")
print('\\nStep 2: best candidates on top of the selected factor')
display(p_log[p_log.step == 2].sort_values('gain', ascending=False).head(5).round(4))
print('Pair trials (gain, checks helped, and the extra over the two gains added up):')
display(p_pairs[['a', 'b', 'group_a', 'group_b', 'gain', 'checks', 'extra', 'joined']].round(4))"""))
note("pile")

cells.append(md("""## Step 12: The Comparison, and the Final Test on Q4

For each way of choosing factors I take the selected factors and do two things:
* **Check (Q3):** add all of them together to my 22 factors, and measure the gain in each year
* **Final test (Q4), used once per selected set:** train on Q1-Q2 with my 22 factors, and with the 22 plus the selected factors, predict Q4, and run the Stage 1 backtest (hold Apple only on days the model says "up")

All selections were finished before this test, and nothing is chosen based on it."""))

cells.append(code("""# Section 12a: Check and final test for every way of choosing factors
scorer = Scorer(features)
summary, by_year = [], []
for label, kept in [('A: one at a time (15 Oct)', a_kept),
                    ('B: 15 Oct factors, team rules', json.loads((RUN_B / 'kept_factors.json').read_text())),
                    ('C: new loop with Claude (21 Oct)', c_team),
                    ('P: whole pile, 8 checks', p_team)]:
    row = {'run': label, 'selected': len(kept)}
    if kept:
        values = {k['name']: evaluate(k['formula'], prices) for k in kept}
        check = summarise_result(scorer.gains(values, {}))
        row.update({'check gain (Q3)': check['gain'], 'check years helped': check['years']})
        m, _ = run_final_test(features, prices, raw, kept)
        by_year.append(m.assign(run=label))
        for model, short in [('logistic_regression', 'LR'), ('random_forest', 'RF')]:
            d = m[m.model == model].pivot(index='year', columns='features')
            row[f'{short} test AUC change'] = (d['auc', 'stage1_plus_llm'] - d['auc', 'stage1_22']).mean()
            row[f'{short} test bal. acc. change'] = (d['balanced_accuracy', 'stage1_plus_llm'] - d['balanced_accuracy', 'stage1_22']).mean()
            row[f'{short} years better (AUC)'] = int((d['auc', 'stage1_plus_llm'] > d['auc', 'stage1_22']).sum())
    summary.append(row)
summary = pd.DataFrame(summary).set_index('run')
display(summary.T.round(4))"""))

cells.append(code("""# Section 12b: Test balanced accuracy in each year (0.5 = coin flip), and the backtest
final = pd.concat(by_year, ignore_index=True)
base = final[final.features == 'stage1_22'].drop_duplicates(['year', 'model']).assign(run='Stage 1 (22 factors)')
view = pd.concat([base, final[final.features == 'stage1_plus_llm']])
colors = {'Stage 1 (22 factors)': '#b9bec7', 'A: one at a time (15 Oct)': '#e08a2b',
          'B: 15 Oct factors, team rules': '#2f6db5', 'P: whole pile, 8 checks': '#3a9e6f'}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, model in zip(axes, ['logistic_regression', 'random_forest']):
    for run, col in colors.items():
        s = view[(view.model == model) & (view.run == run)].sort_values('year')
        if len(s):
            ax.plot(s.year, s.balanced_accuracy, marker='o', lw=2, color=col, label=run)
    ax.axhline(0.5, color='#999', lw=0.8, ls='--')
    ax.set_title(model.replace('_', ' ').title()); ax.set_xticks([2022, 2023, 2024, 2025])
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Test balanced accuracy'); axes[0].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()

ret = view.pivot_table(index=['model', 'run'], columns='year', values='strategy_return').round(3)
ret.loc[('buy and hold', 'Apple'), :] = final.groupby('year').buy_hold_return.first().round(3)
print('Backtest return in Q4 of each year'); display(ret)"""))
note("compare")

cells.append(md("""## Step 13: Running the New Loop Live

Switched off so the notebook doesn't call Claude every time it is opened. To run it: set `RUN_LIVE = True` (and set the API key in Terminal before starting Jupyter). The same loop runs from Terminal with `python experiments/run_stage2_loop_v2.py --rounds 8` (add `--rules results/stage2/rules_v2_8checks.json` for 8 checks). Every prompt, reply and result is saved in a new folder in `results/stage2/runs_v2/`."""))

cells.append(code("""# Section 13: Run the new loop live
RUN_LIVE = False                    # True = ask Claude for real (needs ANTHROPIC_API_KEY)
ROUNDS = 8

def run_loop_v2(rounds=ROUNDS, rules_file='results/stage2/rules_v2.json'):
    out = Path('results/stage2/runs_v2') / f'{time.strftime(\"%Y%m%d_%H%M\")}_claude'
    r = json.loads(Path(rules_file).read_text())
    loop = LoopV2(proposer=ClaudeProposer('claude-sonnet-5-5', tool=TOOL_V2, system=SYSTEM_V2),
                  features=features, prices=prices, out_dir=out,
                  stage1_importance_path=Path('results/multi_year/permutation_importance_summary.csv'),
                  rules=Rules(**{**r, 'checks': tuple(r.get('checks', ['q3']))}))
    for i in range(1, rounds + 1):
        print(f'Round {i}')
        loop.run_round(i)
    print('Selected team:', loop.team, '| saved in', out)
    return loop

if RUN_LIVE:
    live = run_loop_v2()
else:
    print('RUN_LIVE is False: using the saved run', RUN_C.name)"""))

if "summary" in NOTES:
    cells.append(md("## Summary\n\n" + NOTES["summary"]))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, ROOT / "Stage 2_v2_Demo.ipynb")
print("written")
