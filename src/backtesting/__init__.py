"""Simple test-only AAPL backtesting."""

from .simulate import simulate_long_flat
from .report import backtest_best, performance_metrics

__all__ = ['simulate_long_flat', 'backtest_best', 'performance_metrics']
