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
