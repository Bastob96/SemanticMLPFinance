import math
import unittest

import numpy as np
import pandas as pd

from src.backtesting import backtest_best, performance_metrics, simulate_long_flat


def toy():
    dates = ['2025-10-01', '2025-10-02', '2025-10-03', '2025-10-06']
    prices = pd.DataFrame({'Date': dates, 'Ticker': 'AAPL', 'Close': [100., 90., 99., 108.9]})
    predictions = pd.DataFrame({'Date': dates[:-1], 'target_date': dates[1:],
        'actual': [0, 1, 1], 'predicted': [1, 0, 1], 'probability_up': [.7, .2, .8],
        'split': 'test', 'experiment': 'A', 'model': 'logistic_regression', 'seed': 42})
    return predictions, prices


class BacktestingTests(unittest.TestCase):
    def test_hand_computed_positions_compounding_drawdown_and_sharpe(self):
        p, prices = toy(); result = simulate_long_flat(p, prices)
        np.testing.assert_allclose(result.strategy_return, [-.1, 0, .1], atol=1e-15)
        np.testing.assert_allclose(result.strategy_equity, [.9, .9, .99], atol=1e-15)
        np.testing.assert_allclose(result.buy_hold_equity, [.9, .99, 1.089], atol=1e-15)
        metrics = performance_metrics(result.strategy_return)
        self.assertAlmostEqual(metrics['cumulative_return'], -.01)
        self.assertAlmostEqual(metrics['max_drawdown'], -.1)
        self.assertAlmostEqual(metrics['sharpe'], 0)
        bh = performance_metrics(result.aapl_return)
        self.assertAlmostEqual(bh['cumulative_return'], .089)
        self.assertAlmostEqual(bh['max_drawdown'], -.1)
        self.assertAlmostEqual(bh['sharpe'], math.sqrt(252)*(.1/3)/math.sqrt(.04/3))

    def test_all_long_equals_buy_hold_all_flat_stays_one(self):
        for signal in [0, 1]:
            p, prices = toy(); p.predicted = signal
            result = simulate_long_flat(p, prices)
            expected = result.buy_hold_equity if signal else np.ones(3)
            np.testing.assert_allclose(result.strategy_equity, expected)

    def test_initial_loss_and_zero_volatility(self):
        self.assertAlmostEqual(performance_metrics([-.2, .1])['max_drawdown'], -.2)
        flat = performance_metrics([0, 0, 0])
        self.assertEqual(flat['cumulative_return'], 0)
        self.assertEqual(flat['max_drawdown'], 0)
        self.assertTrue(math.isnan(flat['sharpe']))
        self.assertTrue(math.isnan(performance_metrics([.1])['sharpe']))
        self.assertTrue(math.isnan(performance_metrics([.1, .1, .1])['sharpe']))

    def test_invalid_returns(self):
        for values in [[], [np.nan], [np.inf], [-1], [-2], [[.1]]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                performance_metrics(values)
        for annual in [0, -1, np.nan]:
            with self.assertRaises(ValueError):
                performance_metrics([.1, -.1], annual)

    def test_input_order_and_nonmutation(self):
        p, prices = toy(); a, b = p.copy(), prices.copy()
        expected = simulate_long_flat(p, prices)
        pd.testing.assert_frame_equal(simulate_long_flat(p.iloc[::-1], prices.iloc[::-1]), expected)
        pd.testing.assert_frame_equal(p, a); pd.testing.assert_frame_equal(prices, b)

    def test_validation_multiple_runs_missing_dates_and_wrong_labels_rejected(self):
        p, prices = toy()
        broken = [p.assign(split='validation'), pd.concat([p, p.assign(experiment='B')]),
                  p.drop(index=1), p.assign(actual=[1, 1, 1]),
                  p.assign(target_date=['2025-10-03', '2025-10-03', '2025-10-06']),
                  p.assign(Date=['2025-07-01', '2025-07-02', '2025-07-03'], target_date=['2025-07-02','2025-07-03','2025-07-04'])]
        for frame in broken:
            with self.assertRaises(ValueError):
                simulate_long_flat(frame, prices)

    def test_missing_duplicate_invalid_prices_rejected(self):
        p, prices = toy()
        broken = [prices.iloc[:-1], prices.drop(index=1), pd.concat([prices, prices.iloc[[0]]]),
                  prices.assign(Close=[100, 0, 99, 108.9]), prices.assign(Close=[100, np.nan, 99, 108.9]),
                  prices.assign(Ticker='MSFT'), prices.drop(columns='Close')]
        for frame in broken:
            with self.assertRaises(ValueError):
                simulate_long_flat(p, frame)

    def test_real_prices_independent_end_to_end(self):
        p = pd.read_csv('results/stage1_predictions.csv')
        prices = pd.read_csv('data/processed/aligned_daily_ohlcv_5y.csv')
        selection, daily, summary = backtest_best(p, prices)
        self.assertEqual((selection['experiment'], selection['model'], selection['seed']), ('B', 'logistic_regression', 42))
        self.assertEqual(selection['n_intervals'], 63)
        self.assertEqual(selection['long_intervals'], 8)
        aapl = prices.loc[prices.Ticker.eq('AAPL')].set_index('Date').Close
        rows = p.loc[p.split.eq('test') & p.experiment.eq('B') & p.model.eq('logistic_regression')].sort_values('Date')
        wealth = 1.; peak = 1.; worst = 0.; returns = []
        for row in rows.itertuples():
            r = (aapl.loc[row.target_date]/aapl.loc[row.Date]-1)*row.predicted
            returns.append(r); wealth *= 1+r; peak = max(peak, wealth); worst = min(worst, wealth/peak-1)
        actual = summary.set_index('strategy').loc['long_flat']
        self.assertAlmostEqual(actual.cumulative_return, wealth-1)
        self.assertAlmostEqual(actual.max_drawdown, worst)
        mean = sum(returns)/63; sd = math.sqrt(sum((r-mean)**2 for r in returns)/62)
        self.assertAlmostEqual(actual.sharpe, math.sqrt(252)*mean/sd)
        expected_bh = aapl.loc[selection['end_date']]/aapl.loc[selection['start_date']]-1
        self.assertAlmostEqual(summary.set_index('strategy').loc['buy_and_hold_aapl', 'cumulative_return'], expected_bh)
        # Test scores/labels in other configurations cannot change selection or returns.
        changed = p.copy(); mask = changed.split.eq('test') & changed.model.eq('random_forest')
        changed.loc[mask, 'predicted'] = changed.loc[mask, 'actual']
        again, other, _ = backtest_best(changed, prices)
        self.assertEqual(again, selection); pd.testing.assert_frame_equal(other, daily)


if __name__ == '__main__':
    unittest.main()
