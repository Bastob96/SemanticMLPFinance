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

import copy

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
    if len(rules.get("checks", ["q3"])) > 1:      # 8 checks: Q2 (trained on Q1) and Q3 (trained on Q1-Q2) each year
        text = rules_text({**rules, "checks": ["q3"]}).replace("of 4 years", "of 8 checks")
        return text + ("\nEach factor is checked twice per year: on Q2 after training on Q1, and on Q3 after training "
                       "on Q1-Q2, so there are 8 checks. Yearly results shown are the Q3 checks.")
    return (f"A factor joins the SELECTED TEAM if, on top of the 22 factors and the current team, it helps in at least "
            f"{rules['min_years']} of 4 years, the average gain is at least {rules['team_gain']:.3f}, and its similarity "
            f"to every existing and selected factor is below {rules['max_corr']:.2f}. The team holds at most {rules['team_size']} factors.\n"
            f"A PAIR joins the team together if the two factors added together help in at least {rules['min_years']} of 4 years, "
            f"the pair's average gain is at least {rules['pair_gain']:.3f}, and the pair's gain is at least "
            f"{rules['pair_synergy']:.3f} more than the two factors' gains (each on top of the team) added up. Two factors "
            f"that each got lucky simply add up their luck; real teamwork gives more than the sum.\n"
            f"A factor that is weak alone but shows some sign of life (helped in 2+ years, or helped one model) stays on the "
            f"PROMISING list and can still join the team later in a pair. Nothing is rejected only for being weak alone.\n"
            f"For reference, factors made of pure random numbers rarely pass these bars; small gains are noise."
            + ("\nGRADED ON A CURVE: every gain shown for a factor alone or on the team is its gain MINUS the typical "
               "(median) gain of a fixed class of 24 template formulas (no LLM) in the same year and on the same team. "
               "Each year's validation quarter reacts to almost any new factor in the same way (in 2024 nearly every "
               "factor lowered the score, in 2023 nearly every factor raised it), so positive now means 'better than a "
               "typical factor in that year'. Pair trials are not curved." if rules.get("curve") else ""))


def build_prompt_v2(*, round_no, n_improve, n_new, n_partner, base_formulas, stage1_top, market,
                    notebook, team_text, promising_text, pairs_text, rules, previous_names, lessons_block=""):
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
{lessons_block}
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


# ---------------- batch mode (Liu, 12 Nov) ----------------
# Liu: "let [the LLM] generate a lot of solutions at once ... try to generate 100 factors, and then search
# those 100 factors", and make sure every factor has a semantic meaning, something that "makes sense in
# 5 minutes". One call, no feedback rounds: Claude proposes the whole batch, then the same tests and keep
# rules (solo test, team trial, pair trials) search it.

TOOL_BATCH = {
    "name": "propose_factors",
    "description": "Return the whole batch of proposed factors.",
    "input_schema": {
        "type": "object",
        "properties": {
            "factors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "snake_case, unique, says what it measures"},
                        "role": {"type": "string", "enum": list(ROLES)},
                        "formula": {"type": "string"},
                        "meaning": {"type": "string",
                                    "description": "one or two plain-English sentences a finance student understands "
                                                   "in 5 minutes: what the factor measures and why it could tell "
                                                   "whether AAPL closes higher tomorrow"},
                    },
                    "required": ["name", "role", "formula", "meaning"],
                },
            },
        },
        "required": ["factors"],
    },
}

SYSTEM_BATCH = """You are a quantitative researcher designing interpretable daily factors for a stock-prediction study.
Every factor must have a clear economic or behavioural meaning that a finance student could understand in
five minutes, and its formula must do exactly what that meaning says. Prefer short, readable formulas.
Never use future information. Write formulas only with the operators and inputs listed."""


def build_prompt_batch(*, n, base_formulas, stage1_top, market, rules):
    base = "\n".join(f"- {k} = {v}" for k, v in base_formulas.items())
    roles = "\n".join(f"- {k}: {v}" for k, v in ROLES.items())
    return f"""BATCH OF {n}

GOAL
Predict whether AAPL closes higher tomorrow than today (binary). A factor is computed after the close of day t.

DATA (daily, adjusted OHLCV, Sept 2021 - Dec 2025)
Inputs: {", ".join(INPUTS)}
Market summary up to 30 June 2025:
{market}

HOW THE BATCH IS SEARCHED
You propose all {n} factors at once. For each year 2022, 2023, 2024 and 2025 the model is trained on Q1-Q2 and
scored on Q3 (validation); Q4 is never used in this search. The score is ROC AUC, averaged over Logistic
Regression and Random Forest. Every factor is tested alone (added to the 22 existing factors), then in a team
trial (on top of the factors already selected), and promising factors are also tested in pairs.
{rules_text(rules)}

THE 22 EXISTING FACTORS (do not repeat them)
{base}

MOST USEFUL EXISTING FACTORS IN STAGE 1
{stage1_top}

FACTOR ROLES (label every factor with one)
{roles}

FORMULA LANGUAGE
Nested function calls only, no + - * / symbols. Windows n must be one of {WINDOWS}.
{operator_guide()}
Example: safe_divide(rolling_std(pct_change(AAPL_Close, 1), 5), rolling_std(pct_change(QQQ_Close, 1), 20))

TASK
Propose exactly {n} factors in one answer.
1. Interpretable first: every factor needs a clear meaning that makes sense to a finance student in 5 minutes,
   written in the "meaning" field, and the formula must match that meaning. Keep formulas short: at most
   about four operators. A factor you cannot explain in plain words does not belong in the batch.
2. Spread the batch: about a quarter of the factors for each of the four roles, and many different ideas
   (trend, reversal, volatility, intraday range and close position, gaps, volume, peers leading Apple,
   Apple against the market).
3. No repeats: nothing that copies one of the 22 existing factors or another factor in the batch.
Answer ONLY by calling the tool propose_factors (no other text)."""


# ---------------- informed mode (16 Nov): what the earlier searches taught us ----------------
# Only validation-based facts are given (which idea earlier SEARCHES kept; near-copies; luck), never anything
# measured on the Q4 test quarters, so the Q4 test of this run stays a fair test.

TOOL_V2_INFORMED = copy.deepcopy(TOOL_V2)
_item = TOOL_V2_INFORMED["input_schema"]["properties"]["factors"]["items"]
_item["properties"]["expected_sign"] = {
    "type": "string", "enum": ["higher_means_up", "higher_means_down"],
    "description": "stated in advance: does a HIGHER value of the factor make an up day tomorrow more or less likely?"}
_item["required"] = _item["required"] + ["expected_sign"]

INFORMED_LESSONS = """
WHAT EARLIER SEARCHES TAUGHT US (use it)
1. Some ideas were selected again and again by separate earlier searches (selection on the validation
   quarters only). They have the most support; explore their neighbourhoods (another peer or QQQ, a 3- or
   5-day average, a z-score) and design partners of a different role, but do not resubmit them as they are:
   a. Apple's volume surge relative to NVIDIA's (3 searches, two different models), e.g.
      subtract(log(safe_divide(AAPL_Volume, rolling_mean(AAPL_Volume, 20))), log(safe_divide(NVDA_Volume, rolling_mean(NVDA_Volume, 20))))
   b. Apple's 3-day return against its peers (2 searches), e.g.
      subtract(rolling_mean(pct_change(AAPL_Close, 1), 3), multiply(add(rolling_mean(pct_change(MSFT_Close, 1), 3), rolling_mean(pct_change(NVDA_Close, 1), 3)), 0.5))
   c. NVIDIA's daily range relative to Apple's (1 search, and a close variant in another), e.g.
      subtract(safe_divide(subtract(NVDA_High, NVDA_Low), NVDA_Close), safe_divide(subtract(AAPL_High, AAPL_Low), AAPL_Close))
2. Many earlier factors were near-copies of the 22 existing factors (similarity 0.9 or more) and added nothing:
   Apple's price against its 5-, 10- or 20-day average or its 20-day high, z-scores of returns or of volume,
   peers' 1-day returns. Already covered by the 22: Apple's returns over 1-20 days, its volatility, distance
   from its 20-day average, volume against its average, its daily range, and QQQ, MSFT and NVDA returns over
   1 and 5 days. Look for information the 22 do not carry.
3. Small validation gains are mostly luck: a random-number factor passes a single bar about 1 time in 30, and
   the best of many random factors looks good. Do not chase a factor that helped in one year; propose ideas
   with a clear mechanism.
4. For EVERY factor state expected_sign in advance: higher_means_up if a higher value should make an up day
   tomorrow more likely, higher_means_down otherwise. After the search each factor's direction is checked in
   every year; a factor whose direction flips between years is not trusted.
"""


# ---------------- direction test (16 Nov): is the story right about which way the factor points? ----------------
# Round 1: the LLM proposes factors and states each one's direction in advance, with no results at all.
# Round 2: it sees how the round-1 directions turned out on the search days (never the sealed Q4 days) and
# proposes new factors, again with directions. See src/stage2/direction.py and experiments/run_direction_test.py.

TOOL_DIRECTION = copy.deepcopy(TOOL_BATCH)
_item = TOOL_DIRECTION["input_schema"]["properties"]["factors"]["items"]
_item["properties"]["expected_sign"] = {
    "type": "string", "enum": ["higher_means_up", "higher_means_down"],
    "description": "your prediction, stated in advance: does a HIGHER value of the factor make an up day "
                   "tomorrow MORE likely (higher_means_up) or LESS likely (higher_means_down)?"}
_item["required"] = _item["required"] + ["expected_sign"]
TOOL_DIRECTION["input_schema"]["properties"]["lesson"] = {
    "type": "string", "description": "round 2 only: two or three sentences on what the direction evidence taught you"}

SYSTEM_DIRECTION = """You are a quantitative researcher designing interpretable daily factors for a stock-prediction study.
Every factor must have a clear economic or behavioural meaning that a finance student could understand in
five minutes, and that meaning must commit to a direction: whether a higher value makes an up day tomorrow
more likely or less likely. Your stated directions will be checked against the data.
Never use future information. Write formulas only with the operators and inputs listed."""


def build_prompt_direction(*, n, base_formulas, stage1_top, market, evidence=""):
    base = "\n".join(f"- {k} = {v}" for k, v in base_formulas.items())
    roles = "\n".join(f"- {k}: {v}" for k, v in ROLES.items())
    round_no = 2 if evidence else 1
    new_rule = ("3. New ideas only: do not resubmit or lightly reword a round-1 factor, and nothing that copies one of the\n"
                "   22 existing factors. Use what the direction evidence taught you about how this stock behaves."
                if evidence else
                "3. No repeats: nothing that copies one of the 22 existing factors or another factor in the batch.")
    return f"""DIRECTION TEST, ROUND {round_no}: BATCH OF {n}

GOAL
Predict whether AAPL closes higher tomorrow than today (binary). A factor is computed after the close of day t.

DATA (daily, adjusted OHLCV, Sept 2021 - Dec 2025)
Inputs: {", ".join(INPUTS)}
Market summary up to 30 June 2025:
{market}

HOW YOUR FACTORS ARE CHECKED
For every factor you state in advance whether a higher value makes an up day tomorrow more likely
(higher_means_up) or less likely (higher_means_down). The data then shows which way the factor really points:
its values are compared on the days before an up day and before a down day, inside each year 2022-2025.
Guessing gives 50% right. No model is trained and nothing is selected, so only the quality of your reasoning counts.
{evidence}
THE 22 EXISTING FACTORS (do not repeat them)
{base}

MOST USEFUL EXISTING FACTORS IN STAGE 1
{stage1_top}

FACTOR ROLES (label every factor with one)
{roles}

FORMULA LANGUAGE
Nested function calls only, no + - * / symbols. Windows n must be one of {WINDOWS}.
{operator_guide()}
Example: safe_divide(rolling_std(pct_change(AAPL_Close, 1), 5), rolling_std(pct_change(QQQ_Close, 1), 20))

TASK
Propose exactly {n} factors in one answer.
1. Interpretable first: every factor needs a clear meaning that makes sense to a finance student in 5 minutes,
   written in the "meaning" field, and the formula must match that meaning. Keep formulas short: at most
   about four operators. The meaning must explain WHY the factor points the way you state in expected_sign.
2. Spread the batch: about a quarter of the factors for each of the four roles, and many different ideas
   (trend, reversal, volatility, intraday range and close position, gaps, volume, peers leading Apple,
   Apple against the market).
{new_rule}
Answer ONLY by calling the tool propose_factors (no other text)."""


def direction_evidence(rows: list, n_right: int) -> str:
    """Round-2 text: how the round-1 directions turned out on the search days (Q4 is never shown)."""
    lines = [f"- {r['name']} = {r['formula']} | you said {r['expected_sign']} | data: "
             f"{'higher_means_up' if r['edge_search'] > 0 else 'higher_means_down'} "
             f"(edge {r['edge_search']:+.3f}, this direction in {r['years_same']} of 4 years) | "
             f"{'RIGHT' if r['right'] else 'WRONG'}" for r in rows]
    return f"""
DIRECTION EVIDENCE FROM ROUND 1
Measured on January-September of 2022, 2023, 2024 and 2025 only. The October-December quarters are sealed:
your round-2 directions will be checked on them, and you never see them.
"edge" = ROC AUC of the factor against tomorrow's direction, minus 0.5 (positive: higher values went with up
days). With about 740 days an edge within about +/-0.04 can be chance, so trust directions that hold in 3 or 4
of the 4 years and patterns shared by many factors more than any single small edge.
Your stated direction was right for {n_right} of {len(rows)} factors (guessing gives {len(rows) // 2}).
{chr(10).join(lines)}
"""
