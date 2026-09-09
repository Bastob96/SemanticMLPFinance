import unittest

import pandas as pd

from src.data import quarterly_split
from src.features import FEATURE_SETS, build_stage1_features


DATA_PATH = "data/processed/aligned_daily_ohlcv_5y.csv"


class Stage1TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(DATA_PATH)
        cls.features = build_stage1_features(cls.raw)

    def test_target_matches_next_aapl_trading_day(self):
        aapl = self.raw.loc[self.raw["Ticker"].eq("AAPL"), ["Date", "Close"]].copy()
        aapl["Date"] = pd.to_datetime(aapl["Date"])
        aapl = aapl.sort_values("Date")
        aapl["expected"] = (aapl["Close"].shift(-1) > aapl["Close"]).astype(int)
        checked = self.features.merge(aapl[["Date", "expected"]], on="Date")
        self.assertTrue(checked["target_next_day_up"].equals(checked["expected"]))

    def test_period_boundaries_do_not_cross(self):
        splits = quarterly_split(self.features, 2025)
        self.assertEqual({name: len(frame) for name, frame in splits.items()}, {"train": 121, "validation": 63, "test": 63})
        partition = {1: "train", 2: "train", 3: "validation", 4: "test"}
        for name, frame in splits.items():
            self.assertTrue((frame["Date"].dt.quarter.map(partition) == frame["target_date"].dt.quarter.map(partition)).all(), name)
        self.assertLess(splits["train"].Date.max(), splits["validation"].Date.min())
        self.assertLess(splits["validation"].Date.max(), splits["test"].Date.min())

    def test_ablation_columns_are_isolated(self):
        self.assertFalse(any(name.startswith(("QQQ_", "MSFT_", "NVDA_")) for name in FEATURE_SETS["A"]))
        self.assertTrue(any(name.startswith("QQQ_") for name in FEATURE_SETS["B"]))
        self.assertFalse(any(name.startswith(("MSFT_", "NVDA_")) for name in FEATURE_SETS["B"]))
        self.assertTrue(any(name.startswith("MSFT_") for name in FEATURE_SETS["C"]))
        self.assertTrue(any(name.startswith("NVDA_") for name in FEATURE_SETS["C"]))


if __name__ == "__main__":
    unittest.main()
