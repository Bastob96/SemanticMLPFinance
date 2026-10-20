"""Instructions for Claude in the interaction-aware loop (version 2, after Liu's 20 Oct feedback).

Differences from version 1 (llm.build_prompt):
  - Claude sees a full report card for every factor tried (feedback.lab_notebook):
    yearly results, which model it helped, similarity, why it worked or failed, its own lesson.
  - Claude sees the selected team, the promising list and the pair results.
  - Each round asks for IMPROVE (fix a factor that worked only partly), NEW and PARTNER
    (a factor designed to work together with a promising or selected one) factors.
  - Claude labels each factor's role, so complementary pairs can be found
    (for example a market-condition factor with an Apple-relative factor).
  - Claude writes a one-line lesson for every factor from the previous round.
"""

from __future__ import annotations

from .formula import INPUTS, WINDOWS, operator_guide

ROLES = {
    "market_condition": "the state of the whole market or tech sector (e.g. QQQ trend or volatility, how much the peers disagree)",
    "relative_move": "Apple compared with the market or with MSFT/NVDA",
    "apple_own": "Apple's own price, range or volume pattern",
    "peer_lead": "moves of QQQ, MSFT or NVDA that might lead Apple's next move",
}

TOOL_V2 = {
    "name": "propose_factors",
    "description": "Return your lessons from the last round and the proposed factors.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reflection": {"type": "string",
                           "description": "3-5 sentences: what the lab notebook shows works or fails, in which years and for which model, and what you will try."},
            "lessons": {
                "type": "array",
                "description": "One entry for every factor from the previous round (empty in round 1).",
                "items": {"type": "object",
                          "properties": {"name": {"type": "string"},
                                         "lesson": {"type": "string", "description": "one sentence: why it probably worked or failed"}},
                          "required": ["name", "lesson"]},
            },
            "factors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "snake_case, unique, descriptive"},
                        "mode": {"type": "string", "enum": ["improve", "new", "partner"]},
                        "parent": {"type": "string", "description": "improve: the factor being fixed; partner: the factor it should work with; new: empty"},
                        "role": {"type": "string", "enum": list(ROLES)},
                        "formula": {"type": "string"},
                        "hypothesis": {"type": "string", "description": "one or two sentences: why it could help, and for improve/partner what you changed or how it complements the parent"},
                    },
                    "required": ["name", "mode", "parent", "role", "formula", "hypothesis"],
                },
            },
        },
        "required": ["reflection", "lessons", "factors"],
    },
}

SYSTEM_V2 = """You are a quantitative researcher searching for interpretable daily factors, working like a scientist with a lab notebook.
Every factor you propose is tested automatically, alone and in combination with others, and the full results come back to you.
Learn from them: work out why a factor helped in some years or for one model and not others, then fix it;
design factors that work together; drop ideas that keep failing. Never use future information.
Write formulas only with the operators and inputs listed."""


def rules_text(rules) -> str:
    return (f"A factor joins the SELECTED TEAM if, on top of the 22 factors and the current team, it helps in at least "
            f"{rules['min_years']} of 4 years, the average gain is at least {rules['team_gain']:.3f}, and its similarity "
            f"to every existing and selected factor is below {rules['max_corr']:.2f}. The team holds at most {rules['team_size']} factors.\n"
            f"A PAIR joins the team together if the two factors added together help in at least {rules['min_years']} of 4 years, "
            f"the pair's average gain is at least {rules['pair_gain']:.3f}, and the pair beats the better of the two alone by at least "
            f"{rules['pair_synergy']:.3f}.\n"
            f"A factor that is weak alone but shows some sign of life (helped in 2+ years, or helped one model) stays on the "
            f"PROMISING list and can still join the team later in a pair. Nothing is rejected only for being weak alone.\n"
            f"For reference, factors made of pure random numbers rarely pass these bars; small gains are noise.")


def build_prompt_v2(*, round_no, n_improve, n_new, n_partner, base_formulas, stage1_top, market,
                    notebook, team_text, promising_text, pairs_text, rules, previous_names):
    base = "\n".join(f"- {k} = {v}" for k, v in base_formulas.items())
    roles = "\n".join(f"- {k}: {v}" for k, v in ROLES.items())
    lessons_task = ("Write a one-sentence lesson for each of these factors from the previous round: "
                    + ", ".join(previous_names)) if previous_names else "Round 1: no lessons yet (return an empty list)."
    return f"""ROUND {round_no}

GOAL
Predict whether AAPL closes higher tomorrow than today (binary). A factor is computed after the close of day t.

DATA (daily, adjusted OHLCV, Sept 2021 - Dec 2025)
Inputs: {", ".join(INPUTS)}
Market summary up to 30 June 2025:
{market}

HOW EACH FACTOR IS TESTED
For each year 2022, 2023, 2024 and 2025: train on Q1-Q2, score on Q3 (validation). Q4 is never used in this search.
The score is ROC AUC; "gain" is the change in AUC when a factor is added, averaged over Logistic Regression (LR)
and Random Forest (RF), and shown for each year and each model.
1. Alone: the factor is added to the 22 existing factors.
2. Team trial: the factor is added to the 22 factors plus the selected team.
3. Pair trial: promising factors of different roles are added two at a time.
{rules_text(rules)}

THE 22 EXISTING FACTORS (do not repeat them)
{base}

MOST USEFUL EXISTING FACTORS IN STAGE 1
{stage1_top}

FACTOR ROLES (label every factor with one)
{roles}

THE SELECTED TEAM
{team_text}

PROMISING LIST (weak alone, may work in a pair; good parents for PARTNER factors)
{promising_text}

PAIR TRIALS SO FAR
{pairs_text}

YOUR LAB NOTEBOOK (every factor tried so far)
{notebook}

FORMULA LANGUAGE
Nested function calls only, no + - * / symbols. Windows n must be one of {WINDOWS}.
{operator_guide()}
Example: safe_divide(rolling_std(pct_change(AAPL_Close, 1), 5), rolling_std(pct_change(QQQ_Close, 1), 20))

TASK
1. {lessons_task}
2. Propose {n_improve} IMPROVE factors: pick factors from the notebook that helped in some years or for one model,
   say in the hypothesis why they probably failed where they failed, and change them to fix that. Parent = that factor.
3. Propose {n_new} NEW factors: genuinely different ideas from everything in the notebook. Parent is empty.
4. Propose {n_partner} PARTNER factors: each designed to work together with one factor on the selected team or the
   promising list (parent = that factor), with a different role. For example, if the parent describes the market
   condition, design one that describes Apple's relative move, and the other way round.
Label every factor with its role. Do not resubmit any formula from the notebook.
Answer ONLY by calling the tool propose_factors (no other text)."""
