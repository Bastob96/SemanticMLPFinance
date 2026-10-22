"""The original 10 Phase 1 AAPL factors (Experiment A), reproduced without changes.

Definitions follow notebook cells 22-31 of ``stage1_extra_hgbr_regression_taiyu``.
In particular the RSI uses Wilder smoothing (EWM, alpha = 1/14, min_periods = 14)
and is scaled as (RSI - 50) / 50, as in Experiment A (not the simple-mean RSI of
the later ``make_factors`` helper used by Experiments B and C).
"""

import numpy as np
import pandas as pd

BASELINE_FACTOR_DEFINITIONS = [
    {"name": "log_return_1d", "formula": "ln(C_t) - ln(C_{t-1})",
     "description": "AAPL 1-day log return", "lookback": 1},
    {"name": "log_return_5d", "formula": "ln(C_t) - ln(C_{t-5})",
     "description": "AAPL 5-day log return (short-term momentum)", "lookback": 5},
    {"name": "log_return_20d", "formula": "ln(C_t) - ln(C_{t-20})",
     "description": "AAPL 20-day log return (monthly momentum)", "lookback": 20},
    {"name": "close_to_ema_10", "formula": "C_t / EMA_10(C)_t - 1, EMA span=10, adjust=False",
     "description": "Distance of AAPL close from its 10-day exponential moving average", "lookback": None},
    {"name": "close_to_ema_20", "formula": "C_t / EMA_20(C)_t - 1, EMA span=20, adjust=False",
     "description": "Distance of AAPL close from its 20-day exponential moving average", "lookback": None},
    {"name": "rsi_14_scaled",
     "formula": "(RSI_14 - 50) / 50, RSI with Wilder smoothing (EWM alpha=1/14, min_periods=14)",
     "description": "AAPL 14-day relative strength index rescaled to about [-1, 1]", "lookback": 14},
    {"name": "realized_vol_5", "formula": "std(log_return_1d over 5 days) * sqrt(252)",
     "description": "AAPL annualised 5-day realised volatility", "lookback": 5},
    {"name": "realized_vol_20", "formula": "std(log_return_1d over 20 days) * sqrt(252)",
     "description": "AAPL annualised 20-day realised volatility", "lookback": 20},
    {"name": "volume_change_1d", "formula": "ln(V_t) - ln(V_{t-1})",
     "description": "AAPL 1-day log volume change", "lookback": 1},
    {"name": "volume_z_20", "formula": "(ln V_t - mean_20(ln V)) / std_20(ln V)",
     "description": "AAPL 20-day z-score of log volume", "lookback": 19},
]

BASELINE_FACTOR_NAMES = [spec["name"] for spec in BASELINE_FACTOR_DEFINITIONS]


def rsi(series, window=14):
    change = series.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + relative_strength)


def compute_baseline_factors(wide):
    """Return the 10 baseline factors on the date index of ``wide``."""
    close = wide["aapl_close"]
    log_close = np.log(close)
    log_return_1d = log_close.diff()
    log_volume = np.log(wide["aapl_volume"].replace(0, np.nan))

    factors = pd.DataFrame(index=wide.index)
    factors["log_return_1d"] = log_return_1d
    factors["log_return_5d"] = log_close.diff(5)
    factors["log_return_20d"] = log_close.diff(20)
    factors["close_to_ema_10"] = close / close.ewm(span=10, adjust=False).mean() - 1
    factors["close_to_ema_20"] = close / close.ewm(span=20, adjust=False).mean() - 1
    factors["rsi_14_scaled"] = (rsi(close, 14) - 50) / 50
    factors["realized_vol_5"] = log_return_1d.rolling(5).std() * np.sqrt(252)
    factors["realized_vol_20"] = log_return_1d.rolling(20).std() * np.sqrt(252)
    factors["volume_change_1d"] = log_volume.diff()
    factors["volume_z_20"] = ((log_volume - log_volume.rolling(20).mean())
                              / log_volume.rolling(20).std())
    return factors[BASELINE_FACTOR_NAMES]
