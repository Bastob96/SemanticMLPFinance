"""What the LLM is told: the 22 Stage 1 factors written in the formula language,
which of them mattered in Stage 1, and a short description of the market data.

Market statistics are computed only up to 30 June 2025 (the end of the last
training period), so nothing from any Q3 validation or Q4 test quarter of 2025
is described to the LLM.
"""

from __future__ import annotations

import pandas as pd

BASE_FORMULAS = {
    "AAPL_return_1d": "pct_change(AAPL_Close, 1)",
    "AAPL_mean_return_3d": "rolling_mean(pct_change(AAPL_Close, 1), 3)",
    "AAPL_mean_return_5d": "rolling_mean(pct_change(AAPL_Close, 1), 5)",
    "AAPL_mean_return_10d": "rolling_mean(pct_change(AAPL_Close, 1), 10)",
    "AAPL_mean_return_20d": "rolling_mean(pct_change(AAPL_Close, 1), 20)",
    "AAPL_cum_return_5d": "pct_change(AAPL_Close, 5)",
    "AAPL_cum_return_10d": "pct_change(AAPL_Close, 10)",
    "AAPL_cum_return_20d": "pct_change(AAPL_Close, 20)",
    "AAPL_volatility_5d": "rolling_std(pct_change(AAPL_Close, 1), 5)",
    "AAPL_volatility_10d": "rolling_std(pct_change(AAPL_Close, 1), 10)",
    "AAPL_volatility_20d": "rolling_std(pct_change(AAPL_Close, 1), 20)",
    "AAPL_close_to_ma20": "subtract(safe_divide(AAPL_Close, rolling_mean(AAPL_Close, 20)), 1)",
    "AAPL_volume_to_ma20": "subtract(safe_divide(AAPL_Volume, rolling_mean(AAPL_Volume, 20)), 1)",
    "AAPL_daily_range": "safe_divide(subtract(AAPL_High, AAPL_Low), AAPL_Close)",
    "QQQ_return_1d": "pct_change(QQQ_Close, 1)",
    "QQQ_cum_return_5d": "pct_change(QQQ_Close, 5)",
    "MSFT_return_1d": "pct_change(MSFT_Close, 1)",
    "MSFT_cum_return_5d": "pct_change(MSFT_Close, 5)",
    "NVDA_return_1d": "pct_change(NVDA_Close, 1)",
    "NVDA_cum_return_5d": "pct_change(NVDA_Close, 5)",
    "AAPL_minus_QQQ_1d": "subtract(pct_change(AAPL_Close, 1), pct_change(QQQ_Close, 1))",
    "AAPL_minus_peer_mean_1d": "subtract(pct_change(AAPL_Close, 1), multiply(add(pct_change(MSFT_Close, 1), pct_change(NVDA_Close, 1)), 0.5))",
}

CONTEXT_END = "2025-06-30"


def market_summary(prices: pd.DataFrame, end: str = CONTEXT_END) -> str:
    """Yearly return and daily volatility per ticker, up to ``end`` only."""
    p = prices.loc[:end]
    lines = []
    for ticker in ("AAPL", "QQQ", "MSFT", "NVDA"):
        close = p[f"{ticker}_Close"]
        r = close.pct_change()
        parts = []
        for year, c in close.groupby(close.index.year):
            ret = c.iloc[-1] / c.iloc[0] - 1
            vol = r.loc[str(year)].std()
            parts.append(f"{year}: {ret:+.0%} (daily vol {vol:.1%})")
        corr = r.corr(p["AAPL_Close"].pct_change()) if ticker != "AAPL" else None
        tail = f"; daily-return correlation with AAPL {corr:.2f}" if corr is not None else ""
        lines.append(f"- {ticker}: " + ", ".join(parts) + tail)
    return "\n".join(lines)


def stage1_importance(path) -> str:
    """Top Stage 1 factors by permutation importance (validation AUC drop)."""
    try:
        s = pd.read_csv(path)
    except OSError:
        return "(not available)"
    top = (s.groupby("factor").agg(importance=("mean_importance", "mean"),
                                   years=("positive_years", "min"))
           .sort_values("importance", ascending=False).head(6))
    return "\n".join(f"- {f}: mean AUC drop when shuffled {r.importance:+.4f}"
                     for f, r in top.iterrows())
