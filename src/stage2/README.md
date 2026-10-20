# Stage 2: automated LLM factor discovery

Supervisor brief (Dr Liu): fully automated loop through an LLM API; the question is whether
an LLM can propose factors that give a *relative* improvement over the Stage 1 factors,
using only the data we already have (AAPL, QQQ, MSFT, NVDA daily OHLCV).

## The loop (`src/stage2/loop.py`)
1. **Propose** (`llm.py`): Claude gets the goal, the data, the 22 Stage 1 factors written as formulas,
   the most useful Stage 1 factors, a market summary up to 30 Jun 2025, and the full scoreboard.
   It returns a short reflection plus VARIANT factors (improve one that scored well) and NEW factors
   (a different idea). Output is forced to JSON through a tool call.
2. **Build** (`formula.py`): the formula is validated (allowed operators, inputs and windows only)
   and computed. Every operator looks backwards only. Tests prove the 22 Stage 1 factors are
   reproduced exactly and that cutting off future data never changes a value.
3. **Score** (`scoring.py`): the factor is added to the 22, LR and RF (seeds 0/1/2) are refitted on
   Q1-Q2 of 2022, 2023, 2024 and 2025 and scored on Q3. Q4 is never touched.
4. **Select**: keep if it helped in >= 3 of 4 years, mean AUC gain >= 0.01 and |Spearman| < 0.9 with
   any existing or kept factor. The 0.01 threshold comes from scoring 32 pure-noise factors
   (`experiments/calibrate_noise.py`): 6/32 passed "3 of 4 years, gain > 0", only 1/32 passes the rule.
5. **Feed back**: every result, including invalid formulas and the reason, goes on the scoreboard
   in the next prompt.

## Fair comparison
`--proposer random` runs the same loop with random formulas instead of the LLM.
LLM loop vs random search, same budget, same selection, then one Q4 test each, answers
"does the LLM add anything beyond chance?"

## Commands
```
export ANTHROPIC_API_KEY=...            # your key, never commit it
python experiments/run_stage2_loop.py --rounds 10                 # LLM loop
python experiments/run_stage2_loop.py --rounds 10 --proposer random   # baseline
python experiments/run_stage2_final_test.py --run results/stage2/runs/<run>   # once, at the end
python -m pytest tests/test_stage2_formula.py
```
Each run folder keeps every prompt, raw LLM response, the scoreboard, per-year scores and kept factors.

## Version 2: testing factors together (Dr Liu, 20 Oct)
Version 1 judges every factor alone. A factor that only helps in combination with another one is
thrown away. Version 2 (`loop_v2.py`) keeps the same formula language, data split and models, and adds:
1. **Report card** (`feedback.py`): for every factor tried, Claude sees its formula, the gain in each
   year, the gain for Logistic Regression and Random Forest, its similarity to existing factors,
   kept/promising/rejected with the reason, an automatic "why" line, and its own lesson from the
   following round.
2. **Team trial**: each factor is also added on top of the 22 factors plus the selected team. The best
   one that helps the team in >= 3 of 4 years with an average gain >= the team bar joins (team max 6),
   then the rest are re-tested on the bigger team.
3. **Pair trial**: a few promising factors, preferably of different kinds (market, Apple-relative,
   Apple-own), are tested two at a time. Both join if the pair helps in >= 3 of 4 years, reaches the
   pair bar, and beats the better of the two alone by the extra-gain bar.
4. **Promising list**: factors that are weak alone but show some sign of life are kept for later pairs
   and get one more team trial at the end. Nothing is thrown away only for being weak alone.
5. **Prompt** (`prompt_v2.py`): each round Claude writes a lesson for every factor of the last round
   and proposes IMPROVE, NEW and PARTNER factors (a partner is designed to work with a promising or
   selected factor), labelling each with a role.
The team and pair bars are set with random-number factors before any real run
(`experiments/calibrate_noise_v2.py` writes `results/stage2/rules_v2.json`).

```
python experiments/calibrate_noise_v2.py --n 48                       # once: sets the bars
python experiments/run_stage2_loop_v2.py --rounds 8                    # new loop with Claude
python experiments/run_stage2_loop_v2.py --proposer replay --replay results/stage2/runs/20261015_1635_claude
python experiments/compare_stage2_runs.py A=<v1 run> B=<replay run> C=<v2 run>   # includes the one-time Q4 test
python -m pytest tests/test_stage2_v2.py
```
Version 2 runs are saved in `results/stage2/runs_v2/` (prompts, replies, scoreboard with yearly
results, pairs.csv, rounds.csv, the selected team in kept_factors.json).
