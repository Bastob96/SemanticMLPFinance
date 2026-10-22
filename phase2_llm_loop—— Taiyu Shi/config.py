"""Fixed experimental configuration for Capstone Phase 2.

Everything that must stay constant across the baseline, the LLM rounds and the
final test lives here. Values that may be changed from the command line
(number of rounds, number of candidates, dry-run, caching, LLM model) are
collected in ``RunSettings`` and recorded in every run's ``run_config.json``.
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "aligned_daily_ohlcv_2025_split.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "cache" / "llm"

# ---------------------------------------------------------------- data / task
TICKERS = ("AAPL", "MSFT", "NVDA", "QQQ")
TARGET_TICKER = "AAPL"
PRICE_FIELDS = ("open", "high", "low", "close", "volume")
TARGET_NAME = "target_log_return"
TARGET_DEFINITION = (
    "AAPL next-trading-day log return: y(t) = ln(Close_AAPL(t+1) / Close_AAPL(t)). "
    "Features X(t) use only information available at the close of trading day t."
)
PRICE_ADJUSTMENT_NOTE = (
    "Yahoo Finance daily bars downloaded with yfinance auto_adjust=True: Open/High/Low/Close "
    "are split- and dividend-adjusted (back-adjusted as of the download date). The same "
    "treatment applies to all four assets. Volume is unadjusted share volume."
)
# Daily |log return| above this value would suggest an unadjusted split.
SPLIT_JUMP_THRESHOLD = 0.25

# Phase 1 Experiment A split: rows with complete baseline factors and target,
# split chronologically 70 / 15 / 15 by row position. The resulting dates are
# frozen and reused for every model in the run.
SPLIT_FRACTIONS = (0.70, 0.85)

# ---------------------------------------------------------------- model (Phase 1, unchanged)
RANDOM_SEED = 42
IMPUTER_STRATEGY = "median"
HGBR_PARAMS = {
    "learning_rate": 0.04,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 15,
    "l2_regularization": 2.0,
    "early_stopping": False,
    "random_state": RANDOM_SEED,
}

# ---------------------------------------------------------------- factor language
MAX_LOOKBACK = 20            # trading days of history a factor may use
MAX_EXPRESSION_CHARS = 600
MAX_EXPRESSION_NODES = 120
MAX_EXPRESSION_DEPTH = 15
MAX_ABS_CONSTANT = 1000.0

# ---------------------------------------------------------------- validation
MAX_MISSING_FRACTION = 0.02   # on modelling rows; remaining gaps use the train-fitted median
MIN_UNIQUE_FRACTION = 0.10    # distinct values / train rows
MIN_TRAIN_STD = 1e-12
REDUNDANCY_THRESHOLD = 0.95   # |Spearman| on training rows
TRUNCATION_CHECK_POINTS = (0.4, 0.7, 1.0)  # fractions of the development rows

# ---------------------------------------------------------------- retention policy
MIN_RELATIVE_MAE_IMPROVEMENT = 0.005   # 0.5 % lower validation return MAE
REQUIRE_RMSE_NOT_WORSE = True
MAX_ACCEPTED_PER_ROUND = 2             # additions and replacements both count
MAX_RETAINED_LLM_FACTORS = 5
ALLOW_REPLACEMENT = True

# ---------------------------------------------------------------- diagnostics
PERMUTATION_REPEATS = 30

# ---------------------------------------------------------------- LLM defaults
DEFAULT_ROUNDS = 3
DEFAULT_CANDIDATES = 5
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_CLAUDE_EFFORT = "medium"
LLM_MAX_TOKENS = 16000
LLM_TIMEOUT_SECONDS = 180.0
LLM_SDK_MAX_RETRIES = 2        # SDK retries for 408/409/429/5xx and connection errors
LLM_MAX_REPAIR_ATTEMPTS = 1    # extra request per round for truncated/refused/unparseable output
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens, used only for the cost estimate in llm_usage.csv.
# Update if prices change; unknown models are logged with cost "null".
MODEL_PRICES_PER_MTOK = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
}


def _env_flag(name, default):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class RunSettings:
    """Settings that may differ between runs; all are written to run_config.json."""

    rounds: int = DEFAULT_ROUNDS
    candidates: int = DEFAULT_CANDIDATES
    dry_run: bool = False
    baseline_only: bool = False
    use_cache: bool = True
    model: str = DEFAULT_CLAUDE_MODEL
    effort: str = DEFAULT_CLAUDE_EFFORT
    use_fallbacks: bool = True
    data_path: str = str(DATA_PATH)
    verbose_llm: bool = False

    @classmethod
    def from_env(cls, **overrides):
        settings = cls(
            model=os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL).strip() or DEFAULT_CLAUDE_MODEL,
            effort=os.environ.get("CLAUDE_EFFORT", DEFAULT_CLAUDE_EFFORT).strip() or DEFAULT_CLAUDE_EFFORT,
            use_fallbacks=_env_flag("CLAUDE_USE_FALLBACKS", True),
            use_cache=_env_flag("LLM_CACHE", True),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(settings, key, value)
        return settings

    def to_dict(self):
        return asdict(self)


def fixed_config_snapshot():
    """All fixed experimental settings, for run_config.json."""
    return {
        "tickers": TICKERS,
        "target_ticker": TARGET_TICKER,
        "target_definition": TARGET_DEFINITION,
        "price_adjustment": PRICE_ADJUSTMENT_NOTE,
        "split_fractions": SPLIT_FRACTIONS,
        "random_seed": RANDOM_SEED,
        "model": {"pipeline": ["SimpleImputer", "HistGradientBoostingRegressor"],
                  "imputer_strategy": IMPUTER_STRATEGY, "hgbr_params": HGBR_PARAMS},
        "factor_language": {"max_lookback": MAX_LOOKBACK, "max_chars": MAX_EXPRESSION_CHARS,
                            "max_nodes": MAX_EXPRESSION_NODES, "max_depth": MAX_EXPRESSION_DEPTH},
        "validation": {"max_missing_fraction": MAX_MISSING_FRACTION,
                       "min_unique_fraction": MIN_UNIQUE_FRACTION,
                       "redundancy_threshold_abs_spearman_train": REDUNDANCY_THRESHOLD,
                       "truncation_check_points": TRUNCATION_CHECK_POINTS},
        "retention_policy": {
            "primary_metric": "validation MAE of next-day log return",
            "min_relative_mae_improvement": MIN_RELATIVE_MAE_IMPROVEMENT,
            "require_rmse_not_worse": REQUIRE_RMSE_NOT_WORSE,
            "max_accepted_per_round": MAX_ACCEPTED_PER_ROUND,
            "max_retained_llm_factors": MAX_RETAINED_LLM_FACTORS,
            "allow_replacement": ALLOW_REPLACEMENT,
        },
        "permutation_repeats": PERMUTATION_REPEATS,
        "llm": {"max_tokens": LLM_MAX_TOKENS, "timeout_seconds": LLM_TIMEOUT_SECONDS,
                "sdk_max_retries": LLM_SDK_MAX_RETRIES,
                "max_repair_attempts": LLM_MAX_REPAIR_ATTEMPTS},
    }
