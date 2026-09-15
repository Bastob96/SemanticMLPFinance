"""Performance summaries and validation-selected test backtesting."""

import numpy as np
import pandas as pd

from src.evaluation import comparison_table, load_predictions, select_best_configuration
from .simulate import simulate_long_flat


def performance_metrics(returns, periods_per_year=252):
    """Compound returns, negative peak drawdown, annualized sample-SD Sharpe.

    Include starting wealth=1 in the high-water mark. Sharpe uses zero risk-free
    return and all trading intervals including flat days; NaN if fewer than two
    observations or sample volatility is zero. No annualized return estimate.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError('Returns must be a finite nonempty vector greater than -1')
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError('periods_per_year must be finite and positive')
    equity = np.concatenate(([1.0], np.cumprod(1 + values)))
    if not np.isfinite(equity).all():
        raise ValueError('Compounded wealth overflow')
    drawdown = equity / np.maximum.accumulate(equity) - 1
    sd = np.std(values, ddof=1) if len(values) > 1 else np.nan
    if np.all(values == values[0]):
        sd = 0.0  # Avoid round-off volatility for a constant nonzero return.
    sharpe = float(np.sqrt(periods_per_year) * np.mean(values) / sd) if sd > 0 else np.nan
    return {'cumulative_return': float(equity[-1] - 1),
            'max_drawdown': float(drawdown.min()), 'sharpe': sharpe}


def backtest_best(predictions, prices):
    """Select on validation only, then simulate exactly one test configuration."""
    frame = load_predictions(predictions)
    selection = select_best_configuration(comparison_table(frame))
    mask = frame.split.eq('test') & frame.experiment.eq(selection['experiment']) & frame.model.eq(selection['model'])
    mask &= frame.seed.isna() if selection['seed'] is None else frame.seed.eq(selection['seed'])
    daily = simulate_long_flat(frame.loc[mask], prices)
    summary = pd.DataFrame([
        {'strategy': 'long_flat', **performance_metrics(daily.strategy_return)},
        {'strategy': 'buy_and_hold_aapl', **performance_metrics(daily.aapl_return)},
    ])
    selection.update(start_date=daily.Date.iloc[0].strftime('%Y-%m-%d'),
                     end_date=daily.target_date.iloc[-1].strftime('%Y-%m-%d'),
                     n_intervals=len(daily), long_intervals=int(daily.position.sum()),
                     periods_per_year=252, risk_free_rate=0, transaction_cost=0,
                     execution='Idealized same-close to next-close; close-based features; optimistic execution assumption')
    return selection, daily, summary
