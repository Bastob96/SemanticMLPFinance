"""Claude API factor generation (live) and a clearly labelled offline mock (dry run).

Safeguards: API key only from the environment / .env; configurable model; SDK
timeout and bounded SDK retries; at most ``LLM_MAX_REPAIR_ATTEMPTS`` extra
requests per round for refused, truncated or unparseable output; optional
response cache; token usage and estimated cost logged for every call.
Structured outputs constrain the response to a JSON schema; each candidate is
still validated individually, so one malformed candidate never discards the rest.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import config

log = logging.getLogger(__name__)

PROMPT_VERSION = "phase2-v1"

RELATION_TYPES = ["new", "refinement", "response_to_feedback"]

CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Unique snake_case factor name"},
        "category": {"type": "string", "description": "e.g. momentum, volatility, volume, cross_asset, "
                                                      "price_volume, mean_reversion, relative_strength"},
        "interpretation": {"type": "string", "description": "Financial interpretation"},
        "math_definition": {"type": "string", "description": "Mathematical definition in plain notation"},
        "inputs": {"type": "array", "items": {"type": "string"},
                   "description": "Input columns used, from the allowed input list"},
        "expression": {"type": "string", "description": "Computable expression in the factor language"},
        "windows": {"type": "array", "items": {"type": "integer"},
                    "description": "Rolling-window / lag parameters used"},
        "predictive_rationale": {"type": "string",
                                 "description": "Why it might help predict the next-day AAPL log return"},
        "relation_to_previous": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": RELATION_TYPES},
                "reference": {"type": "string",
                              "description": "Earlier factor(s) or feedback item this responds to; empty if new"},
                "justification": {"type": "string",
                                  "description": "How the previous results motivated this candidate"},
            },
            "required": ["type", "reference", "justification"],
            "additionalProperties": False,
        },
    },
    "required": ["name", "category", "interpretation", "math_definition", "inputs", "expression",
                 "windows", "predictive_rationale", "relation_to_previous"],
    "additionalProperties": False,
}

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "feedback_analysis": {"type": "string",
                              "description": "Your reading of the previous results and how they shape this round"},
        "candidates": {"type": "array", "items": CANDIDATE_SCHEMA},
    },
    "required": ["feedback_analysis", "candidates"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = f"""\
You are a quantitative research assistant in a university capstone experiment. You propose
candidate financial factors for predicting Apple's (AAPL) next-trading-day log return from daily
OHLCV data of AAPL, MSFT, NVDA and QQQ. A fixed HistGradientBoostingRegressor is retrained with
the 10 fixed baseline factors plus selected candidates, and is evaluated on a validation period.
You receive the results of earlier rounds and should use them.

Rules:
- Use only the factor language described in the user message. Expressions are interpreted by a
  restricted parser, not executed as Python. Anything outside the language is rejected.
- Use only information available at the close of day t. Never reference future values.
- Respect the lookback limit of {config.MAX_LOOKBACK} trading days, counting nested windows.
- Each candidate needs a unique name, should not duplicate earlier expressions or baseline factors,
  and should not be highly correlated with existing factors.
- Aim for economically meaningful and diverse ideas (for example momentum, reversal, volatility,
  volume dynamics, relative strength, cross-asset relationships, price-volume interaction).
- When earlier rounds exist, state explicitly in relation_to_previous how their results (accepted or
  rejected candidates, rejection reasons, metrics, correlations, importance) motivated each candidate.
  Refining a promising factor or proposing a better replacement for a weak retained factor is allowed.
- Do not claim that a factor will improve results; the experiment decides.
- Return exactly the number of candidates requested, as JSON matching the schema."""


class FatalLLMError(RuntimeError):
    """Configuration problems (bad key, unknown model, invalid request) that retrying cannot fix."""


@dataclass
class GenerationResult:
    status: str                         # "ok", "partial", "failed", "dry_run"
    candidates: list = field(default_factory=list)
    feedback_analysis: str = ""
    attempts: list = field(default_factory=list)
    error: str = ""


def build_user_message(payload):
    n = payload["candidates_requested"]
    intro = (f"Round {payload['round']}. Propose exactly {n} new candidate factors.\n"
             "All experimental information is in the JSON below: data context (training-period statistics "
             "only), factor language, baseline factors, current retained LLM factors and their validation "
             "metrics, feedback from every previous round, and expressions already tried.\n\n")
    return intro + json.dumps(payload, indent=2, ensure_ascii=False)


def estimate_cost(model, usage):
    prices = config.MODEL_PRICES_PER_MTOK.get(model)
    if not prices or not usage:
        return None
    return round((usage.get("input_tokens", 0) * prices["input"]
                  + usage.get("output_tokens", 0) * prices["output"]
                  + (usage.get("cache_read_input_tokens") or 0) * prices["cache_read"]
                  + (usage.get("cache_creation_input_tokens") or 0) * prices["cache_write"]) / 1e6, 6)


def parse_proposal(text):
    """Parse the response JSON. Returns (candidates, analysis); raises ValueError if unusable."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
        raise ValueError("Response JSON must be an object with a 'candidates' list")
    candidates = list(data["candidates"])
    if not candidates:
        raise ValueError("Response contains no candidates")
    return candidates, str(data.get("feedback_analysis", ""))


class ClaudeFactorGenerator:
    def __init__(self, settings, cache_dir=config.CACHE_DIR):
        try:
            import anthropic
        except ImportError as exc:
            raise FatalLLMError("The 'anthropic' package is not installed; run pip install -r requirements.txt") from exc
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise FatalLLMError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
                                "or run with --dry-run.")
        self.anthropic = anthropic
        self.settings = settings
        self.client = anthropic.Anthropic(timeout=config.LLM_TIMEOUT_SECONDS,
                                          max_retries=config.LLM_SDK_MAX_RETRIES)
        self.cache_dir = Path(cache_dir)
        self.mode = "live"

    # ------------------------------------------------------------------ request
    def _request_params(self, user_text):
        output_config = {"format": {"type": "json_schema", "schema": PROPOSAL_SCHEMA}}
        if self.settings.effort and self.settings.effort.lower() != "none":
            output_config["effort"] = self.settings.effort
        params = {"model": self.settings.model, "max_tokens": config.LLM_MAX_TOKENS,
                  "system": SYSTEM_PROMPT, "messages": [{"role": "user", "content": user_text}],
                  "output_config": output_config}
        if self.settings.use_fallbacks:
            params["betas"] = [config.FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def _cache_path(self, params):
        key = hashlib.sha256(json.dumps({"prompt_version": PROMPT_VERSION, **params},
                                        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _call(self, params):
        """One streamed request (streaming avoids HTTP timeouts on long outputs)."""
        if self.settings.use_fallbacks:
            with self.client.beta.messages.stream(**params) as stream:
                return stream.get_final_message()
        with self.client.messages.stream(**params) as stream:
            return stream.get_final_message()

    # ------------------------------------------------------------------ public
    def generate(self, round_id, payload):
        user_text = build_user_message(payload)
        params = self._request_params(user_text)
        attempts, last_error = [], ""
        cache_path = self._cache_path(params)
        if self.settings.use_cache and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            candidates, analysis = parse_proposal(cached["response_text"])
            attempts.append({**cached, "from_cache": True, "cache_file": cache_path.name})
            log.info("Round %d: using cached Claude response %s", round_id, cache_path.name)
            return GenerationResult("ok", candidates, analysis, attempts)

        for attempt in range(1 + config.LLM_MAX_REPAIR_ATTEMPTS):
            record = {"attempt": attempt + 1, "from_cache": False, "model_requested": params["model"],
                      "request": {k: v for k, v in params.items()}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
            try:
                log.info("Round %d: calling Claude (%s), attempt %d", round_id, params["model"], attempt + 1)
                started = time.time()
                response = self._call(params)
                record["seconds"] = round(time.time() - started, 2)
            except (self.anthropic.AuthenticationError, self.anthropic.PermissionDeniedError,
                    self.anthropic.NotFoundError, self.anthropic.BadRequestError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                attempts.append(record)
                raise FatalLLMError(f"Claude API rejected the request ({type(exc).__name__}). Check "
                                    f"ANTHROPIC_API_KEY, CLAUDE_MODEL and account access. Details: {exc}") from exc
            except (self.anthropic.RateLimitError, self.anthropic.InternalServerError,
                    self.anthropic.APIConnectionError, self.anthropic.APIStatusError) as exc:
                # The SDK has already retried retryable errors; one further attempt at most.
                last_error = record["error"] = f"{type(exc).__name__}: {exc}"
                attempts.append(record)
                log.warning("Round %d: API error: %s", round_id, last_error)
                continue

            raw = response.model_dump(mode="json")
            usage = raw.get("usage") or {}
            text = "".join(block.get("text", "") for block in raw.get("content", []) if block.get("type") == "text")
            record.update({"model_served": raw.get("model"), "stop_reason": raw.get("stop_reason"),
                           "stop_details": raw.get("stop_details"), "usage": usage,
                           "estimated_cost_usd": estimate_cost(raw.get("model") or params["model"], usage),
                           "response_text": text, "response_raw": raw})
            if raw.get("stop_reason") == "refusal":
                last_error = record["error"] = f"Model refused: {raw.get('stop_details')}"
            elif raw.get("stop_reason") == "max_tokens":
                last_error = record["error"] = "Output truncated at max_tokens"
            else:
                try:
                    candidates, analysis = parse_proposal(text)
                except ValueError as exc:
                    last_error = record["error"] = str(exc)
                else:
                    attempts.append(record)
                    if self.settings.use_cache:
                        self.cache_dir.mkdir(parents=True, exist_ok=True)
                        cache_path.write_text(json.dumps({k: v for k, v in record.items() if k != "request"},
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
                    status = "ok" if len(candidates) == payload["candidates_requested"] else "partial"
                    return GenerationResult(status, candidates, analysis, attempts)
            attempts.append(record)
            log.warning("Round %d: unusable response: %s", round_id, last_error)
        return GenerationResult("failed", [], "", attempts, error=last_error)


# ---------------------------------------------------------------------- dry run
def _candidate(name, expression, inputs, category, interpretation, windows, relation="new", reference=""):
    return {"name": name, "category": category, "interpretation": interpretation,
            "math_definition": expression, "inputs": inputs, "expression": expression, "windows": windows,
            "predictive_rationale": "DRY RUN placeholder rationale (not generated by an LLM).",
            "relation_to_previous": {"type": relation, "reference": reference,
                                     "justification": "DRY RUN: fixed offline candidate; does not use feedback."}}


MOCK_LIBRARY = [
    [   # round 1: two plausible factors and three deliberately invalid ones
        _candidate("aapl_qqq_rel_mom_5", "diff(log(aapl_close), 5) - diff(log(qqq_close), 5)",
                   ["aapl_close", "qqq_close"], "relative_strength", "AAPL 5-day momentum relative to QQQ", [5]),
        _candidate("aapl_qqq_vol_ratio_10", "rolling_std(aapl_ret1, 10) / rolling_std(qqq_ret1, 10)",
                   ["aapl_ret1", "qqq_ret1"], "volatility", "AAPL volatility relative to the market", [10]),
        _candidate("future_close_leak", "lag(aapl_close, -1) / aapl_close - 1",
                   ["aapl_close"], "invalid_test", "Deliberately invalid: references tomorrow's close", [1]),
        _candidate("unknown_vwap_ratio", "aapl_vwap / aapl_close",
                   ["aapl_vwap", "aapl_close"], "invalid_test", "Deliberately invalid: column does not exist", []),
        _candidate("copy_of_log_return_5d", "diff(log(aapl_close), 5)",
                   ["aapl_close"], "invalid_test", "Deliberately redundant with baseline log_return_5d", [5]),
    ],
    [   # round 2
        _candidate("rel_move_x_volume_z", "(aapl_ret1 - qqq_ret1) * zscore(aapl_logvol, 20)",
                   ["aapl_ret1", "qqq_ret1", "aapl_logvol"], "price_volume",
                   "Relative move amplified by abnormal volume", [20], "refinement", "aapl_qqq_rel_mom_5"),
        _candidate("aapl_qqq_corr_20", "rolling_corr(aapl_ret1, qqq_ret1, 20)",
                   ["aapl_ret1", "qqq_ret1"], "cross_asset", "Rolling co-movement of AAPL with QQQ", [20]),
        _candidate("constant_ratio", "aapl_close / aapl_close",
                   ["aapl_close"], "invalid_test", "Deliberately invalid: constant", []),
        _candidate("aapl_vs_peers_10", "rolling_mean(aapl_ret1 - (msft_ret1 + nvda_ret1) / 2, 10)",
                   ["aapl_ret1", "msft_ret1", "nvda_ret1"], "relative_strength",
                   "AAPL 10-day return relative to MSFT/NVDA average", [10]),
        _candidate("too_long_lookback", "zscore(rolling_mean(aapl_ret1, 10), 20)",
                   ["aapl_ret1"], "invalid_test", "Deliberately invalid: needs 29 days of history", [10, 20]),
    ],
    [   # round 3
        _candidate("aapl_range_position_20",
                   "(aapl_close - rolling_min(aapl_low, 20)) / (rolling_max(aapl_high, 20) - rolling_min(aapl_low, 20))",
                   ["aapl_close", "aapl_low", "aapl_high"], "mean_reversion",
                   "Position of the close within its 20-day range", [20]),
        _candidate("range_ratio_aapl_qqq_5",
                   "rolling_mean((aapl_high - aapl_low) / aapl_close, 5) / rolling_mean((qqq_high - qqq_low) / qqq_close, 5)",
                   ["aapl_high", "aapl_low", "aapl_close", "qqq_high", "qqq_low", "qqq_close"], "volatility",
                   "AAPL intraday range relative to QQQ", [5]),
        _candidate("market_signed_aapl_move", "sign(qqq_ret1) * abs(aapl_ret1)",
                   ["qqq_ret1", "aapl_ret1"], "cross_asset", "AAPL move size signed by the market direction", []),
        _candidate("nvda_minus_qqq_3d", "rolling_sum(nvda_ret1, 3) - rolling_sum(qqq_ret1, 3)",
                   ["nvda_ret1", "qqq_ret1"], "cross_asset", "NVDA 3-day excess return over QQQ", [3]),
        _candidate("import_attempt", "__import__('os')",
                   [], "invalid_test", "Deliberately invalid: disallowed function call", []),
    ],
]


class MockFactorGenerator:
    """Offline candidates for testing the pipeline. Makes no API calls and ignores feedback."""

    mode = "dry_run"

    def generate(self, round_id, payload):
        n = payload["candidates_requested"]
        order = [(round_id - 1 + k) % len(MOCK_LIBRARY) for k in range(len(MOCK_LIBRARY))]
        suffix = "" if round_id <= len(MOCK_LIBRARY) else f"_r{round_id}"
        pool = [dict(c, name=c["name"] + suffix) for i in order for c in MOCK_LIBRARY[i]]
        candidates = pool[:n]
        attempt = {"attempt": 1, "from_cache": False, "model_requested": "MOCK (dry run, no API call)",
                   "user_message": build_user_message(payload), "response_text": json.dumps(
                       {"feedback_analysis": "DRY RUN: no LLM analysis.", "candidates": candidates}, indent=1),
                   "usage": {}, "estimated_cost_usd": 0.0}
        return GenerationResult("dry_run", candidates, "DRY RUN: no LLM analysis.", [attempt])

