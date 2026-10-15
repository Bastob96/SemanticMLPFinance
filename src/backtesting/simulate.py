"""Idealized close-to-next-close, test-only, fully invested long/flat simulation."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import load_predictions


def simulate_long_flat(predictions, prices):
    """Simulate one saved run on its exact, contiguous Q4 test window.

    Signals at Date earn Close(target_date)/Close(Date)-1 when predicted=1.
    This assumes execution at the observed close, despite close-based features:
    an optimistic frictionless research diagnostic, not an executable strategy.
    Cash earns zero; no leverage, shorts, costs or additional signal lag.
    Prices must contain Date, Ticker, Close and the actual next trading dates.
    """
    frame = load_predictions(predictions)
    if not frame.split.eq('test').all():
        raise ValueError('Backtesting accepts test rows only')
    if len(frame[['experiment', 'model', 'seed']].drop_duplicates()) != 1:
        raise ValueError('Supply exactly one model/experiment/seed')
    frame = frame.sort_values('Date').reset_index(drop=True)
    if frame.Date.dt.year.nunique() != 1 or not frame.Date.dt.quarter.eq(4).all() or not frame.target_date.dt.quarter.eq(4).all() or not frame.Date.dt.year.eq(frame.target_date.dt.year).all():
        raise ValueError('Stage 1 test dates must stay inside Q4 of one year')
    prices = pd.read_csv(prices) if isinstance(prices, (str, Path)) else prices.copy()
    if not {'Date', 'Ticker', 'Close'}.issubset(prices.columns):
        raise ValueError('Prices require Date, Ticker, Close')
    aapl = prices.loc[prices.Ticker.eq('AAPL'), ['Date', 'Close']].copy()
    aapl['Date'] = pd.to_datetime(aapl.Date, errors='raise')
    aapl['Close'] = pd.to_numeric(aapl.Close, errors='raise')
    if aapl.empty or aapl.Date.isna().any() or aapl.Date.duplicated().any() or not np.isfinite(aapl.Close).all() or not aapl.Close.gt(0).all():
        raise ValueError('AAPL prices must have unique dates and finite positive closes')
    aapl = aapl.sort_values('Date').reset_index(drop=True)
    aapl['next_date'] = aapl.Date.shift(-1)
    aapl['next_close'] = aapl.Close.shift(-1)
    joined = frame.merge(aapl, on='Date', how='left', validate='one_to_one')
    if joined.Close.isna().any() or joined.next_close.isna().any() or not joined.target_date.eq(joined.next_date).all():
        raise ValueError('Every target must match the next available AAPL trading date')
    if len(joined) > 1 and not np.array_equal(joined.target_date.iloc[:-1].to_numpy(), joined.Date.iloc[1:].to_numpy()):
        raise ValueError('Missing intervals: test window must be contiguous')
    returns = joined.next_close / joined.Close - 1
    if not joined.actual.eq((returns > 0).astype(int)).all():
        raise ValueError('Actual labels disagree with supplied AAPL prices')
    result = joined[['Date', 'target_date', 'actual', 'predicted']].copy()
    result['position'] = joined.predicted.astype(int)
    result['aapl_return'] = returns
    result['strategy_return'] = returns * result.position
    result['strategy_equity'] = (1 + result.strategy_return).cumprod()
    result['buy_hold_equity'] = (1 + result.aapl_return).cumprod()
    return result
