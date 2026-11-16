"""Ask an LLM for new factors. The prompt carries the context and the scoreboard so far."""

from __future__ import annotations

import json
import random

from .formula import INPUTS, WINDOWS, operator_guide

TOOL = {
    "name": "propose_factors",
    "description": "Return the proposed factors.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reflection": {"type": "string",
                           "description": "2-4 sentences: what the scoreboard suggests works or fails, and what you will try."},
            "factors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "snake_case, unique, descriptive"},
                        "mode": {"type": "string", "enum": ["variant", "new"]},
                        "parent": {"type": "string", "description": "for a variant: the factor it modifies; for new: empty"},
                        "formula": {"type": "string"},
                        "hypothesis": {"type": "string", "description": "one sentence: why it could predict AAPL's next-day direction"},
                    },
                    "required": ["name", "mode", "parent", "formula", "hypothesis"],
                },
            },
        },
        "required": ["reflection", "factors"],
    },
}

SYSTEM = """You are a quantitative researcher proposing interpretable daily factors.
Each factor is scored automatically and the scores are shown back to you in later rounds,
so learn from them: build on what helps, abandon what does not.
Never use future information. Write formulas only with the operators and inputs listed."""


def build_prompt(*, round_no, n_variant, n_new, base_formulas, stage1_top, market, scoreboard, kept):
    base = "\n".join(f"- {k} = {v}" for k, v in base_formulas.items())
    kept_txt = "\n".join(f"- {k['name']} = {k['formula']}" for k in kept) or "(none yet)"
    return f"""ROUND {round_no}

GOAL
Predict whether AAPL closes higher tomorrow than today (binary). A factor is computed after the close of day t.

DATA (daily, adjusted OHLCV, Sept 2021 - Dec 2025)
Inputs: {", ".join(INPUTS)}
Market summary up to 30 June 2025:
{market}

HOW EACH FACTOR IS SCORED
For each year 2022, 2023, 2024 and 2025: train on Q1-Q2, score on Q3 (validation).
The factor is added to the 22 existing factors below; Logistic Regression and Random Forest are refitted.
- auc_gain: validation ROC AUC with the factor minus without it (averaged over the models and years)
- years_helped: in how many of the 4 years auc_gain > 0
- importance: drop in validation AUC when the factor is shuffled
- max_corr: largest |Spearman correlation| with any existing or already kept factor
A factor is KEPT only if years_helped >= 3, mean auc_gain >= 0.01 and max_corr < 0.9.
For reference, pure random noise gets mean auc_gain around -0.002 (std 0.008), so small gains are noise.
Single-quarter AUC is noisy: consistency across years matters as much as size.

THE 22 EXISTING FACTORS (do not repeat them)
{base}

MOST USEFUL EXISTING FACTORS IN STAGE 1
{stage1_top}

FACTORS KEPT SO FAR
{kept_txt}

SCOREBOARD OF EVERYTHING TRIED SO FAR
{scoreboard}

FORMULA LANGUAGE
Nested function calls only, no + - * / symbols. Windows n must be one of {WINDOWS}.
{operator_guide()}
Example: safe_divide(rolling_std(pct_change(AAPL_Close, 1), 5), rolling_std(pct_change(QQQ_Close, 1), 20))

TASK
Propose {n_variant} VARIANT factors and {n_new} NEW factors.
- VARIANT: change a factor that scored well (from the scoreboard or the useful Stage 1 factors) to make it
  stronger or more stable, e.g. a different window, normalisation, a peer or market comparison. Name the parent.
- NEW: a genuinely different idea from everything above (a different family, e.g. intraday position,
  gaps, volume-price agreement, cross-stock lead-lag, regime or volatility interaction). Parent is empty.
Do not resubmit any formula on the scoreboard.\nAnswer ONLY by calling the tool propose_factors (no other text)."""


class ClaudeProposer:
    """Calls the Claude API. Needs ANTHROPIC_API_KEY in the environment."""

    def __init__(self, model="claude-sonnet-5-5", max_tokens=8000, tool=None, system=None):
        import anthropic
        # ask for gzip: some Anaconda installs ship an old brotli that breaks response decoding
        self.client = anthropic.Anthropic(default_headers={"Accept-Encoding": "gzip, deflate"})
        self.model, self.max_tokens = model, max_tokens
        self.tool, self.system = tool or TOOL, system or SYSTEM   # version 2 passes its own tool and system text

    def __call__(self, prompt: str) -> dict:
        # tool_choice "auto": newer models do not allow forcing a specific tool;
        # the prompt asks for the tool call, and plain-JSON text is accepted as a fallback.
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=self.system, tools=[self.tool], tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": prompt}],
        )
        block = next((b for b in msg.content if getattr(b, "type", "") == "tool_use"), None)
        if block is not None:
            out = dict(block.input)
        else:
            text = "".join(getattr(b, "text", "") for b in msg.content)
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(f"no tool call or JSON in the reply (stop_reason={msg.stop_reason})")
            out = json.loads(text[start:end + 1])
        if not isinstance(out.get("factors"), list):
            raise ValueError("reply has no 'factors' list")
        out["_usage"] = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
        out["_model"] = msg.model
        out["_stop_reason"] = msg.stop_reason
        return out


class RandomProposer:
    """Offline stand-in for testing the loop without an API key (and as a no-LLM baseline)."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def _formula(self):
        r = self.rng
        t1, t2 = r.sample(["AAPL", "QQQ", "MSFT", "NVDA"], 2)
        w1, w2 = r.choice(WINDOWS[1:]), r.choice(WINDOWS[2:])
        return r.choice([
            f"subtract(pct_change({t1}_Close, {w1}), pct_change({t2}_Close, {w1}))",
            f"safe_divide(rolling_std(pct_change({t1}_Close, 1), {w2}), rolling_std(pct_change({t2}_Close, 1), {w2}))",
            f"zscore({t1}_Volume, {w2})",
            f"safe_divide(subtract({t1}_Close, {t1}_Open), subtract({t1}_High, {t1}_Low))",
            f"rolling_mean(multiply(sign(pct_change({t1}_Close, 1)), zscore({t1}_Volume, {w2})), {w1})",
            f"subtract(safe_divide({t1}_Close, rolling_max({t1}_High, {w2})), 1)",
        ])

    def __call__(self, prompt: str) -> dict:
        import re
        n_var = int(re.search(r"Propose (\d+) VARIANT", prompt).group(1))
        n_new = int(re.search(r"and (\d+) NEW", prompt).group(1))
        out = []
        for i in range(n_var + n_new):
            out.append({"name": f"random_{self.rng.randrange(10**6)}", "mode": "variant" if i < n_var else "new",
                        "parent": "", "formula": self._formula(), "hypothesis": "random baseline"})
        return {"reflection": "random proposer (no LLM)", "factors": out}


def dump(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
