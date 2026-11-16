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

PG-S2-36, Automatic Discovery of Interpretable Financial Factors
Abu Wakkas Bastob, 22 October 2026

Last week my loop judged each factor on its own, and only kept the ones that helped by themselves. On 20 October Dr Liu pointed out that this can throw away factors that are only useful together with other factors. He asked me to:

1. give Claude more detailed feedback on every factor (formula, result in each year, LR vs RF, kept or not, similarity to other factors, and a short reason)
2. test each new factor on top of the factors I have already selected, not just on its own
3. try simple pairs of promising factors
4. tell Claude which factors seem to work together and ask it for partners
5. compare the new version with last week's one-at-a-time approach

In his email on 21 October he also suggested generating a bigger set of factors and using feature selection to pick a good combination, so I tried that as well (Step 11).

The new code is written out in the cells below. The parts that haven't changed since last week (the formula language, the market summary, the Claude call and the final test) are just imported here. They are shown in full in `Stage 2_Demo.ipynb`."""))

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
    print(f'{name}:', run.name, '|', n, 'factors tried |', len(team), 'in the selected team')
p_kept = json.loads(Path('results/stage2/pile_8checks/kept_factors.json').read_text())
print('P: pile_8checks |', len(pd.read_csv('results/stage2/pile_8checks/scoreboard.csv')),
      'factors in one pile (all of A and C, near-copies removed) |', len(p_kept), 'selected  (Step 11)')"""))

cells.append(md("""## Step 2: What changed

| | Last week (A) | This week (C) |
|---|---|---|
| What Claude sees | one scoreboard line per factor (average gain, years helped) | a report card per factor (gain in each year, LR vs RF, similarity, kept or not and why, its own lesson) |
| How a factor is tested | alone, added to my 22 factors | alone, then on top of the selected team, and in pairs |
| What gets kept | factors that help alone | factors that make the team better, alone or as a pair |
| Factors that are weak alone | thrown away | kept on a "promising" list for pair trials |
| What Claude is asked for | 3 variants and 3 new ideas | 2 improve, 2 new and 2 partner factors, plus a lesson about each factor from the last round |

I compare four ways of choosing factors in Step 12:

- **A** is last week's run, where every factor was judged alone (1 kept).
- **B** uses last week's same 48 factors, but judges them with the new team and pair rules. No new Claude call is needed.
- **C** is the new loop with Claude, run this week (48 new factors).
- **P** is Dr Liu's idea from 21 October. All the factors from A and C go into one pile, and forward selection picks from it (Step 11).

Everything else is the same as last week: the formula language (only past data), the four years 2022 to 2025 (train on Q1 and Q2, check on Q3), the two models, and Q4 only being used once at the very end.

One round of the new loop looks like this:
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

cells.append(md("""## Step 3: The report card Claude gets

Every factor Claude tries gets a short report card like the ones below. The "why" line is written automatically from the measured numbers, so it can't say anything that wasn't actually measured. In the next round Claude has to write a one-line lesson about each of its factors, and that lesson stays on the card after that."""))

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

cells.append(md("""## Step 4: What I tell Claude now

The instructions still have last week's parts (the goal, the data, my 22 factors and the formula language). Three new parts are added: the selected team, the promising list, and the pair trials so far. The old scoreboard is replaced by the report cards, and the task now asks for improve, new and partner factors plus a lesson for every factor from the previous round."""))

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

cells.append(md("""## Step 5: Team trial and pair trial

The scoring code from last week now takes two inputs, the factor (or factors) to add and the team to add them on top of. That gives three kinds of test:

- **Alone:** one factor added to my 22 factors. This is last week's test and gives the same numbers.
- **Team trial:** one factor added to my 22 factors plus the selected team.
- **Pair trial:** two factors added at the same time, on top of the 22 factors and the team.

In every case the gain is the Q3 ROC AUC with the new factor(s) minus the AUC without them, for each of the four years, averaged over Logistic Regression and the three Random Forests."""))

cells.append(module_cell("src/stage2/scoring.py", "Section 5a: Testing factors alone, on top of the team, or in a pair"))
cells.append(module_cell("src/stage2/selection.py", "Section 5b: The rules for the team, the pairs and the promising list"))

cells.append(md("""This is the loop itself. In each round it asks Claude for factors, checks and computes each formula, tests each factor alone, runs the team trial and a few pair trials, and then updates the team, the promising list and the report cards."""))

cells.append(module_cell("src/stage2/curve.py", "Section 5c: Grading on a curve (tried in Step 10, not used)"))
cells.append(module_cell("src/stage2/loop_v2.py", "Section 5d: The new loop"))

cells.append(md("""## Step 6: Setting the new bars with random numbers

This is the same idea as last week's fake-factor test. Before running anything real, I put factors made of pure random numbers through the new tests to see how often they would get into the team just by luck. The bars were fixed before the real runs, and B and C use the same ones. The check turned up three problems, and each one changed the design.

**Lucky pairs.** My first pair rule was that a pair had to beat the better of its two members on its own. But a random factor that looks promising has usually been a bit lucky already, and when you put two of them together their luck adds up, so the pair nearly always looks better than either one. With that rule, 30% of the promising random pairs got in. So now a pair has to do clearly better than the two gains added together.

**The forest changing by itself.** With the default setting, the Random Forest looks at 4 factors at each split while it has up to 24 factors, and at 5 once it has 25 or more. On a team of 2, adding any extra column (even random numbers) changed the forest itself, and 8 out of 18 promising random factors passed the team trial. I now fix it at 4 factors per split. That's what the forest was already using for my 22 factors and for every single-factor test, so last week's numbers don't change.

**Second chances.** I had planned to give everything on the promising list one more team trial at the end. Even after fixing the forest, about 1 in 18 promising random factors got in that way, so I turned that step off."""))

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

cells.append(md("""## Step 7: What happened in the new run (C)

Eight rounds of 6 factors each, so 48 factors in total, the same as last week."""))

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

cells.append(md("""## Step 8: Last week's factors, judged the new way (B)

I wanted to separate two things: the effect of the new judging, and the effect of the new feedback to Claude. So I took the same 48 factors Claude suggested last week, in the same order, and ran them through the new team trial, pair trials and promising list. This didn't need any new Claude calls."""))

cells.append(code("""# Section 8: What the new rules pick from last week's 48 factors
b_board = pd.read_csv(RUN_B / 'scoreboard.csv').fillna({'reason': '', 'how_selected': ''})
b_team = json.loads((RUN_B / 'kept_factors.json').read_text())
print('Last week (A) kept:', [k['name'] for k in a_kept])
print('New rules (B) selected:')
for k in b_team:
    print(f"  {k['name']}  ({k['how_selected']})\\n    {k['formula']}")
print('\\nStatus of the 48 factors under the new rules:', b_board.status.value_counts().to_dict())"""))
note("run_b")

cells.append(md("""## Step 9: Why so few factors pass

Claude actually spotted this first, in round 2 of run C, when it wrote that "every factor helped in 2023 and 2025 but hurt in 2024". The table below counts how many factors helped in each year, for Claude's factors from both weeks and for the template formulas that were made without any LLM."""))

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

cells.append(md("""## Step 10: A longer measuring stick (8 checks instead of 4)

Each year only has one checking quarter (Q3, roughly 63 trading days), so the effect of luck is about as big as the effects I'm trying to find. To get more evidence I added a second, earlier check every year, where the model trains on Q1 and is checked on Q2. That gives each factor 8 checks instead of 4. The Q3 checks are exactly the same as before.

I also tried "grading on a curve", where each factor is compared with a typical template factor from the same year. The problem was that random numbers then looked better than real price factors in the bad years, so the bar had to go up to 0.02 and nothing passed at all. I didn't use it in the end.

As before, I set the bar using random numbers and template formulas first, before looking at any of Claude's factors."""))

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

cells.append(md("""## Step 11: Dr Liu's suggestion, one big pile and then feature selection

In his 21 October email Dr Liu suggested generating a large set of factors with the LLM and then using feature selection to find a good combination. Here, every factor Claude has written so far goes into one pile, and forward selection builds the team from it. It adds the factor that improves the team the most (if it passes the 8-check bar), re-tests the rest on top of the bigger team, and repeats. After that, a few pairs are tried on top of the team."""))

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

cells.append(md("""## Step 12: The comparison and the final test on Q4

For each way of choosing factors I take the selected factors and do two things.

- **Check on Q3:** add them all to my 22 factors together and measure the gain in each year.
- **Final test on Q4 (used once per selected set):** train on Q1 and Q2 with my 22 factors, and again with the 22 plus the selected factors, predict Q4, and run the Stage 1 backtest (hold Apple only on days the model says "up").

All the selections were finished before this test, and nothing here was chosen based on Q4."""))

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

cells.append(md("""## Step 13: Running the new loop live

This is switched off, so the notebook doesn't call Claude every time it's opened. To run it, set `RUN_LIVE = True` (and set the API key in Terminal before starting Jupyter). The same loop can be run from Terminal with `python experiments/run_stage2_loop_v2.py --rounds 8`, adding `--rules results/stage2/rules_v2_8checks.json` to use 8 checks. Every prompt, reply and result is saved in a new folder under `results/stage2/runs_v2/`."""))

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
