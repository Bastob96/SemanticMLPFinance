"""Builds 'Stage 2_v3_Ablation.ipynb'. Run: python experiments/build_stage2_v3_notebook.py

Stage 2, version 3 (Dr Liu, 22 Oct): an ablation study of the loop. Each part of the design is switched off
one at a time and the result is compared with the full method. The runs come from experiments/run_ablations.sh;
the notebook only reads their saved folders and the one-time Q4 test in results/stage2/ablation/, so it opens fast
and never calls Claude. Runs that have not been made yet are shown as "not run yet".
The "Reading this output" notes are kept in experiments/stage2_v3_notes.json, written after the runs.
"""

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTES_FILE = ROOT / "experiments/stage2_v3_notes.json"
NOTES = json.loads(NOTES_FILE.read_text()) if NOTES_FILE.exists() else {}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []


def note(key):
    if key in NOTES:
        cells.append(md("**Reading this output**\n\n" + NOTES[key]))


cells.append(md("""# Stage 2, Version 3: Which Parts of the Loop Matter? (Ablation Study)

Group PG-S2-36, Automatic Discovery of Interpretable Financial Factors
Abu Wakkas Bastob, 5 November 2026

At the prototype demo on 22 October, Dr Liu asked us to:

1. compare our approaches with each other, now that we all use the same setting
2. do an **ablation study**: switch off one part of the design at a time and check how much it matters (for example, which language model is used)
3. try **more iterations**

An ablation works like testing a recipe: bake the cake again without one ingredient and see how much worse it gets. Here the "cake" is my full loop from 22 October (run C), and the "ingredients" are:

| Part | Full method | Switched off / changed |
|---|---|---|
| Feedback to Claude | report card for every factor | Claude sees only the formulas already tried, no results |
| Language model | Claude Sonnet | a smaller model (Claude Haiku) |
| Number of rounds | 8 | 16 |
| Testing together | team trial | each factor judged alone (last week's run A) |
| Pair trials | on | off |
| The LLM itself | Claude | template formulas, no LLM |

Every run uses exactly the same data, split, models and bars as on 22 October. The Q4 test quarters are only used at the very end, once per selected set (Step 4)."""))

cells.append(code("""# Section 1: Setup and the list of runs
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 90)

R = Path('results/stage2/runs_v2')
RUNS = {   # label: (folder, what is different from the full method)
    'full_C':          (R / '20261021_1105_claude', 'full method (22 Oct run C)'),
    'no_feedback':     (R / 'abl_claude_nofeedback', 'Claude sees no results'),
    'haiku':           (R / 'abl_claude_haiku', 'smaller model: Claude Haiku'),
    'rounds_16':       (R / 'abl_claude_16rounds', '16 rounds instead of 8'),
    'C_repeat2':       (R / 'abl_claude_repeat2', 'full method, run again'),
    'C_repeat3':       (R / 'abl_claude_repeat3', 'full method, run again'),
    'C_no_pairs':      (R / 'abl_C_nopairs', "C's factors, pair trials off"),
    'full_B':          (R / '20261021_1110_replay_20261015_1635_claude', "15 Oct factors, team rules (22 Oct run B)"),
    'B_no_pairs':      (R / 'abl_B_nopairs', "15 Oct factors, pair trials off"),
    'one_at_a_time_A': (Path('results/stage2/runs/20261015_1635_claude'), '15 Oct factors judged alone (run A)'),
    'template_0':      (R / 'abl_template_seed0', 'template formulas, no LLM'),
    'template_1':      (R / 'abl_template_seed1', 'template formulas, no LLM'),
    'template_2':      (R / 'abl_template_seed2', 'template formulas, no LLM'),
    'C_check':         (R / 'abl_C_check', "C's factors re-judged (should equal C)"),
    'B_check':         (R / 'abl_B_check', "15 Oct factors re-judged (should equal B)"),
}

ROUNDS = {'rounds_16': 16}          # all other runs have 8 rounds

def finished(label):
    folder = RUNS[label][0]
    if not (folder / 'kept_factors.json').exists():
        return False
    log = folder / 'rounds.csv'           # version 2 runs; run A (version 1) only has the scoreboard
    rounds = pd.read_csv(log)['round'] if log.exists() else pd.read_csv(folder / 'scoreboard.csv')['round']
    done_rounds = pd.to_numeric(rounds, errors='coerce').max()
    return done_rounds >= ROUNDS.get(label, 8)

rows = []
for label, (folder, what) in RUNS.items():
    done = finished(label)
    row = {'run': label, 'what is different': what, 'done': 'yes' if done else 'not run yet'}
    if done:
        board = pd.read_csv(folder / 'scoreboard.csv')
        kept = json.loads((folder / 'kept_factors.json').read_text())
        row.update(rounds=int(board['round'].max()), tried=len(board), invalid=int((board.status == 'invalid').sum()),
                   selected=len(kept))
    rows.append(row)
runs = pd.DataFrame(rows).set_index('run')
display(runs)"""))
note("runs")

cells.append(md("""## Step 2: The two new switches

Two parts of the loop needed a switch so they could be turned off. Everything else stays the same.

- `feedback=False`: Claude is still asked for 6 factors per round, but its lab notebook only lists the names and formulas already tried (so it doesn't repeat them). No scores, no report cards, no team, no promising list and no pair results.
- `pairs_on=False`: the pair trials are skipped, so a factor can only join the team on its own.

The cell below prints the part of `LoopV2.run_round` (in `src/stage2/loop_v2.py`) that does this."""))

cells.append(code("""# Section 2: The ablation switches in the loop code
import inspect
from src.stage2.loop_v2 import LoopV2
src = inspect.getsource(LoopV2.run_round).splitlines()
start = next(i for i, l in enumerate(src) if 'if self.feedback' in l)
print('\\n'.join(src[start:start + 15]))
print('    ...')
print('\\n'.join(l for l in src if 'pairs_on' in l or '_pair_trials(' in l))"""))

cells.append(code("""# Section 2b: What Claude sees without feedback (round 2 of the no-feedback run)
p = RUNS['no_feedback'][0] / 'round_02_prompt.txt'
if p.exists():
    text = p.read_text()
    a = text.index('THE SELECTED TEAM'); b = text.index('FORMULA LANGUAGE')
    print(text[a:b][:1800])
else:
    print('Not run yet: run  bash experiments/run_ablations.sh  first')"""))

cells.append(md("""## Step 3: Validation results (Q3 checks, used for choosing)

For every run: how many factors Claude (or the templates) tried, how many were selected, and how much the selected set adds on the validation quarters (Q3, 2022-2025). This is what the loop itself sees. Nothing here uses the Q4 test."""))

cells.append(code("""# Section 3: What each run selected
S = Path('results/stage2/ablation/summary.csv')
summary = pd.read_csv(S).set_index('run') if S.exists() else pd.DataFrame()
order = [r for r in RUNS if r in summary.index]
if len(summary):
    view = summary.loc[order, ['tried', 'invalid', 'selected', 'val_gain', 'val_years', 'val_lr', 'val_rf']]
    display(view.round(4))
    print('\\nSelected factors:')
    for r in order:
        names = summary.loc[r, 'selected_names']
        print(f'  {r:16} {names if isinstance(names, str) and names else \"(none)\"}')
else:
    print('No summary yet: run  bash experiments/run_ablations.sh  first')"""))
note("validation")

cells.append(md("""## Step 4: The one-time Q4 test

Each selected set is added to the 22 Stage 1 factors and tested once on Q4 of each year (2022-2025), the quarters the loop never saw. The table shows the change compared with the 22 factors alone, averaged over the four years. A run that selected nothing equals the 22 factors, so its changes are zero."""))

cells.append(code("""# Section 4: Q4 test, change compared with the 22 Stage 1 factors (average of 4 years)
cols = {'q4_auc_change_lr': 'LR AUC', 'q4_auc_change_rf': 'RF AUC',
        'q4_bal_acc_change_lr': 'LR balanced acc.', 'q4_bal_acc_change_rf': 'RF balanced acc.',
        'q4_return_plus_lr': 'LR backtest', 'q4_return_plus_rf': 'RF backtest'}
if len(summary):
    q4 = summary.loc[order].reindex(columns=list(cols)).rename(columns=cols)
    q4.insert(0, 'selected', summary.loc[order, 'selected'])
    display(q4.round(3).fillna(0))
    b = summary.dropna(subset=['q4_return_22_lr']).iloc[0]
    print(f\"22 factors alone, backtest per Q4: LR {b.q4_return_22_lr:+.3f}, RF {b.q4_return_22_rf:+.3f}; \"
          f\"buy and hold {b.q4_buy_hold:+.3f}\")"""))
note("q4")

cells.append(md("""## Step 5: The ablation table

Each row compares the full method with one part switched off or changed, using the same measures as above. The validation gain is what the loop sees when it chooses; the Q4 columns are the one-time test. For the template formulas and the repeats, the average of the runs is shown (with the spread in brackets in Step 6)."""))

cells.append(code("""# Section 5: Ablation table, one part at a time
def value(run, col):
    if run not in summary.index: return np.nan
    v = summary.loc[run, col]
    return 0.0 if (pd.isna(v) and summary.loc[run, 'selected'] == 0) else v

def mean_of(runs_, col):
    vals = [value(r, col) for r in runs_ if r in summary.index]
    return np.nanmean(vals) if vals else np.nan

COMPARE = [   # part tested, full method run(s), changed run(s)
    ('Feedback to Claude (report cards off)', ['full_C'], ['no_feedback']),
    ('Language model (Sonnet -> Haiku)', ['full_C'], ['haiku']),
    ('More rounds (8 -> 16)', ['full_C'], ['rounds_16']),
    ('Pair trials off (on C factors)', ['full_C'], ['C_no_pairs']),
    ('Pair trials off (on 15 Oct factors)', ['full_B'], ['B_no_pairs']),
    ('Testing together off (judge alone)', ['full_B'], ['one_at_a_time_A']),
    ('Claude -> template formulas', ['full_C', 'C_repeat2', 'C_repeat3'], ['template_0', 'template_1', 'template_2']),
]
MEASURES = {'selected': 'selected', 'val_gain': 'val gain', 'q4_bal_acc_change_lr': 'Q4 LR bal. acc.',
            'q4_bal_acc_change_rf': 'Q4 RF bal. acc.', 'q4_auc_change_lr': 'Q4 LR AUC', 'q4_auc_change_rf': 'Q4 RF AUC'}
rows = []
if len(summary):
    for part, full, changed in COMPARE:
        if not any(r in summary.index for r in changed):
            rows.append({'part tested': part, 'status': 'not run yet'}); continue
        row = {'part tested': part, 'status': 'done'}
        for col, name in MEASURES.items():
            row[f'{name} (full)'] = mean_of(full, col)
            row[f'{name} (changed)'] = mean_of(changed, col)
        rows.append(row)
    ablation = pd.DataFrame(rows).set_index('part tested')
    display(ablation.round(3))
    ablation.round(4).to_csv('results/stage2/ablation/ablation_table.csv')"""))
note("ablation")

cells.append(md("""## Step 6: Is it luck? Repeats of the full method

Claude doesn't give the same factors twice, so the full method was run three times with the same settings. If the three runs select very different factors or give very different Q4 results, a single difference in Step 5 could just be luck. The template formulas were also run with three different seeds."""))

cells.append(code("""# Section 6: Spread over repeated runs
groups = {'Claude, full method': ['full_C', 'C_repeat2', 'C_repeat3'],
          'Template formulas': ['template_0', 'template_1', 'template_2']}
rows = []
for name, members in groups.items():
    have = [m for m in members if m in summary.index] if len(summary) else []
    for col in ['selected', 'val_gain', 'q4_bal_acc_change_lr', 'q4_bal_acc_change_rf']:
        vals = [value(m, col) for m in have]
        rows.append({'runs': name, 'measure': col, 'n': len(vals),
                     'mean': np.nanmean(vals) if vals else np.nan,
                     'min': np.nanmin(vals) if vals else np.nan, 'max': np.nanmax(vals) if vals else np.nan})
display(pd.DataFrame(rows).set_index(['runs', 'measure']).round(3))"""))
note("repeats")

cells.append(code("""# Section 6b: Chart, Q4 balanced accuracy change for every run
if len(summary):
    show = [r for r in order if not r.endswith('_check')]
    x = np.arange(len(show))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    lr = [value(r, 'q4_bal_acc_change_lr') for r in show]
    rf = [value(r, 'q4_bal_acc_change_rf') for r in show]
    ax.bar(x - 0.2, lr, 0.4, label='Logistic Regression', color='#2f6db5')
    ax.bar(x + 0.2, rf, 0.4, label='Random Forest', color='#e08a2b')
    ax.axhline(0, color='#555', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(show, rotation=30, ha='right')
    ax.set_ylabel('Q4 balanced accuracy\\nchange vs 22 factors')
    ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False)
    plt.tight_layout(); plt.show()"""))

cells.append(md("""## Step 7: Checks

Two runs only re-judge saved factors with exactly the same rules, so they must select the same factors as the original runs. If they do, the code changes for the switches didn't change anything else."""))

cells.append(code("""# Section 7: Re-judged runs must match the originals
def team(label):
    f = RUNS[label][0] / 'kept_factors.json'
    return [k['name'] for k in json.loads(f.read_text())] if finished(label) else None
for check, original in [('C_check', 'full_C'), ('B_check', 'full_B')]:
    a, b = team(check), team(original)
    if a is None:
        print(f'{check}: not run yet')
    else:
        print(f'{check}: {\"matches\" if a == b else \"DOES NOT MATCH\"} {original}  ({a or \"(none)\"})')"""))

cells.append(md("""## Step 8: Comparing with Taiyu's and Leyao's approaches

Dr Liu also asked us to compare our approaches with each other. We all start from the same 22 factors, the same data and the same quarterly split. The quarter we all test on is **Q4 2025**, so the table below uses that quarter only. Leyao's numbers are from her 22 October presentation. Taiyu's are still to come: his loop measures MAE and RMSE, so we first need the same up/down measures (AUC and balanced accuracy) from it.

A quick check that we really share the setting: my 22-factor Logistic Regression scores AUC 0.478 on Q4 2025, the same 0.478 Leyao reported for the original 22 factors."""))

cells.append(code("""# Section 8: Q4 2025 only, for comparing with the team
BY = Path('results/stage2/ablation/by_year.csv')
by = pd.read_csv(BY) if BY.exists() else pd.read_csv('results/stage2/comparison/by_year.csv')
q = by[by.year == 2025]
mine = []
base = q[q.features == 'stage1_22'].drop_duplicates('model').set_index('model')
mine.append({'approach': 'Stage 1: 22 factors', 'LR AUC': base.loc['logistic_regression', 'auc'],
             'RF AUC': base.loc['random_forest', 'auc']})
for run in [r for r in ['full_B', 'full_C', 'no_feedback', 'haiku', 'rounds_16'] if r in set(q.run)]:
    s = q[(q.run == run) & (q.features == 'stage1_plus_llm')].set_index('model')
    mine.append({'approach': f'Abu: {run}', 'LR AUC': s.loc['logistic_regression', 'auc'],
                 'RF AUC': s.loc['random_forest', 'auc']})
team_rows = [
    {'approach': 'Leyao: 22 + 16 selected (22 Oct demo)', 'LR AUC': 0.540, 'RF AUC': np.nan},
    {'approach': "Taiyu: (to add)", 'LR AUC': np.nan, 'RF AUC': np.nan},
]
display(pd.DataFrame(mine + team_rows).set_index('approach').round(3))"""))
note("team")

if "summary" in NOTES:
    cells.append(md("## Summary\n\n" + NOTES["summary"]))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, ROOT / "Stage 2_v3_Ablation.ipynb")
print("written")
