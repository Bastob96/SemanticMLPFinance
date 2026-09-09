"""Small, interpretable, leakage-safe Stage 1 feature set."""

from __future__ import annotations

import numpy as np
import pandas as pd


AAPL_FEATURES = [
    "AAPL_return_1d",
    "AAPL_mean_return_3d",
    "AAPL_mean_return_5d",
    "AAPL_mean_return_10d",
    "AAPL_mean_return_20d",
    "AAPL_cum_return_5d",
    "AAPL_cum_return_10d",
    "AAPL_cum_return_20d",
    "AAPL_volatility_5d",
    "AAPL_volatility_10d",
    "AAPL_volatility_20d",
    "AAPL_close_to_ma20",
    "AAPL_volume_to_ma20",
    "AAPL_daily_range",
]

FEATURE_SETS = {
    "A": AAPL_FEATURES,
    "B": AAPL_FEATURES
    + ["QQQ_return_1d", "QQQ_cum_return_5d", "AAPL_minus_QQQ_1d"],
    "C": AAPL_FEATURES
    + [
        "QQQ_return_1d",
        "QQQ_cum_return_5d",
        "MSFT_return_1d",
        "MSFT_cum_return_5d",
        "NVDA_return_1d",
        "NVDA_cum_return_5d",
        "AAPL_minus_QQQ_1d",
        "AAPL_minus_peer_mean_1d",
    ],
}


def _validate_input(data: pd.DataFrame) -> None:
    required = {"Ticker", "Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    missing_tickers = {"AAPL", "QQQ", "MSFT", "NVDA"}.difference(
        data["Ticker"].unique()
    )
    if missing_tickers:
        raise ValueError(f"Missing tickers: {sorted(missing_tickers)}")
    if data.duplicated(["Ticker", "Date"]).any():
        raise ValueError("Duplicate Ticker-Date rows found")


def build_stage1_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build features at date t and AAPL direction at the next trading date.

    Rolling windows end at t. Only ``target_next_day_up`` and ``target_date``
    access the next AAPL trading observation.
    """
    data = raw.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    _validate_input(data)

    wide = data.pivot(index="Date", columns="Ticker", values=["Open", "High", "Low", "Close", "Volume"]).sort_index()
    result = pd.DataFrame(index=wide.index)
    daily_returns: dict[str, pd.Series] = {}

    for ticker in ("AAPL", "QQQ", "MSFT", "NVDA"):
        close = wide[("Close", ticker)]
        daily_returns[ticker] = close.pct_change(fill_method=None)
        result[f"{ticker}_return_1d"] = daily_returns[ticker]
        result[f"{ticker}_cum_return_5d"] = close.pct_change(5, fill_method=None)

    aapl_close = wide[("Close", "AAPL")]
    aapl_return = daily_returns["AAPL"]
    for window in (3, 5, 10, 20):
        result[f"AAPL_mean_return_{window}d"] = aapl_return.rolling(window).mean()
    for window in (5, 10, 20):
        result[f"AAPL_cum_return_{window}d"] = aapl_close.pct_change(window, fill_method=None)
        result[f"AAPL_volatility_{window}d"] = aapl_return.rolling(window).std()

    result["AAPL_close_to_ma20"] = aapl_close / aapl_close.rolling(20).mean() - 1
    aapl_volume = wide[("Volume", "AAPL")]
    result["AAPL_volume_to_ma20"] = aapl_volume / aapl_volume.rolling(20).mean() - 1
    result["AAPL_daily_range"] = (wide[("High", "AAPL")] - wide[("Low", "AAPL")]) / aapl_close
    result["AAPL_minus_QQQ_1d"] = aapl_return - daily_returns["QQQ"]
    result["AAPL_minus_peer_mean_1d"] = aapl_return - (
        daily_returns["MSFT"] + daily_returns["NVDA"]
    ) / 2
    result["previous_direction"] = (aapl_return > 0).astype(int)

    next_close = aapl_close.shift(-1)
    result["target_next_day_up"] = (next_close > aapl_close).astype(float)
    result.loc[next_close.isna(), "target_next_day_up"] = np.nan
    result["target_date"] = result.index.to_series().shift(-1)

    feature_columns = sorted({name for names in FEATURE_SETS.values() for name in names})
    result = result.reset_index().dropna(subset=feature_columns + ["target_next_day_up", "target_date"])
    result["target_next_day_up"] = result["target_next_day_up"].astype(int)
    return result
