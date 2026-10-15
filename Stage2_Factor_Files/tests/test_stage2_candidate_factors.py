from __future__ import annotations
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("factor_builder", PACKAGE_ROOT / "src/factors/build_stage2_candidate_factors.py")
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class Stage2FactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hp, sp = os.environ.get("STAGE2_FIVE_YEAR"), os.environ.get("STAGE2_SPLIT_2025")
        cls.hp = M.resolve_input("aligned_daily_ohlcv_5y.csv", Path(hp) if hp else None)
        cls.sp = M.resolve_input("aligned_daily_ohlcv_2025_split.csv", Path(sp) if sp else None)
        cls.history, cls.split = M._load_market_data(cls.hp), M._load_market_data(cls.sp)
        cls.wide = M._wide_ohlcv(cls.history)
        cls.table = M.build_factor_table(cls.hp, cls.sp)
        cls.parts = M.partition_tables(cls.table)

    def test_dates_and_schema(self):
        self.assertEqual(self.table.shape, (250, 33))
        self.assertEqual(self.table.Date.min(), pd.Timestamp("2025-01-02"))
        self.assertEqual(self.table.Date.max(), pd.Timestamp("2025-12-31"))
        self.assertEqual(self.table.Date.nunique(), 250)
        self.assertTrue(self.table.Date.is_monotonic_increasing)

    def test_date_split_counts(self):
        self.assertEqual(self.table.Split.value_counts().to_dict(), {"train": 122, "validation": 64, "test": 64})

    def test_complete_finite_output(self):
        self.assertFalse(self.table.isna().any().any())
        self.assertTrue(np.isfinite(self.table[M.RAW_COLUMNS + M.FACTOR_COLUMNS].to_numpy()).all())

    def test_train_factors_nonconstant(self):
        self.assertTrue(self.parts["train"][M.FACTOR_COLUMNS].nunique().gt(1).all())

    def test_all_target_values_and_dates(self):
        close = self.wide.AAPL_Close
        dates = list(close.index)
        for row in self.table.itertuples(index=False):
            pos = dates.index(row.Date)
            self.assertEqual(row.target_date, dates[pos + 1])
            self.assertEqual(row.target_next_day_up, int(close.iloc[pos + 1] > close.iloc[pos]))

    def test_boundary_purge(self):
        self.assertEqual({k: len(v) for k, v in self.parts.items()}, {"train": 121, "validation": 63, "test": 63})
        kept = pd.concat(self.parts.values()).Date
        self.assertEqual(self.table.loc[~self.table.Date.isin(kept), "Date"].dt.strftime("%Y-%m-%d").tolist(), ["2025-06-30", "2025-09-30", "2025-12-31"])
        self.assertLess(self.parts["train"].target_date.max(), self.parts["validation"].Date.min())
        self.assertLess(self.parts["validation"].target_date.max(), self.parts["test"].Date.min())
        self.assertLessEqual(self.parts["test"].target_date.max(), pd.Timestamp("2025-12-31"))

    def test_all_eight_formulas_independently(self):
        eps = 1e-12

        for date in [
            "2025-01-02",
            "2025-06-30",
            "2025-09-30",
            "2025-12-31",
        ]:
            with self.subTest(date=date):
                t = self.wide.index.get_loc(pd.Timestamp(date))

                def price_ret(ticker, n, pos):
                    close = self.wide[f"{ticker}_Close"]
                    return close.iloc[pos] / close.iloc[pos - n] - 1.0

                def volume_ret(ticker, n, pos):
                    volume = self.wide[f"{ticker}_Volume"]
                    return volume.iloc[pos] / volume.iloc[pos - n] - 1.0

                peer_signed = np.mean([
                    price_ret("MSFT", 3, j)
                    + price_ret("NVDA", 3, j)
                    for j in range(t - 4, t + 1)
                ])

                peer_dispersion = np.mean([
                    abs(
                        price_ret("MSFT", 3, j)
                        - price_ret("NVDA", 3, j)
                    )
                    for j in range(t - 4, t + 1)
                ])

                peer_average_5d = (
                    price_ret("MSFT", 5, t)
                    + price_ret("NVDA", 5, t)
                ) / 2.0

                peer_relative = (
                    price_ret("AAPL", 5, t)
                    - peer_average_5d
                )

                aapl_risk = np.std([
                    price_ret("AAPL", 2, j)
                    for j in range(t - 9, t + 1)
                ], ddof=1)

                qqq_risk = np.std([
                    price_ret("QQQ", 2, j)
                    for j in range(t - 9, t + 1)
                ], ddof=1)

                relative_risk = aapl_risk / (qqq_risk + eps)

                peer_volume = np.mean([
                    volume_ret("MSFT", 3, j)
                    + volume_ret("NVDA", 3, j)
                    for j in range(t - 4, t + 1)
                ])

                volume_confirmation = (
                    price_ret("AAPL", 3, t)
                    * peer_volume
                )

                regime_interaction = (
                    price_ret("AAPL", 5, t)
                    - price_ret("QQQ", 5, t)
                ) * qqq_risk

                aapl_range = np.mean([
                    (
                        self.wide["AAPL_High"].iloc[j]
                        - self.wide["AAPL_Low"].iloc[j]
                    )
                    / (self.wide["AAPL_Close"].iloc[j] + eps)
                    for j in range(t - 4, t + 1)
                ])

                qqq_range = np.mean([
                    (
                        self.wide["QQQ_High"].iloc[j]
                        - self.wide["QQQ_Low"].iloc[j]
                    )
                    / (self.wide["QQQ_Close"].iloc[j] + eps)
                    for j in range(t - 4, t + 1)
                ])

                normalized_range_ratio = (
                    aapl_range / (qqq_range + eps)
                )

                lagged_qqq_momentum = (
                    price_ret("AAPL", 3, t)
                    - price_ret("QQQ", 3, t - 2)
                )

                expected = [
                    peer_signed,
                    peer_dispersion,
                    peer_relative,
                    relative_risk,
                    volume_confirmation,
                    regime_interaction,
                    normalized_range_ratio,
                    lagged_qqq_momentum,
                ]

                actual = self.table.loc[
                    self.table.Date.eq(date),
                    M.FACTOR_COLUMNS,
                ].iloc[0].to_numpy(dtype=float)

                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=1e-10,
                    atol=1e-12,
                )

    def test_prefix_invariance(self):
        full = M.calculate_factors(self.wide)
        for cutoff in ["2025-01-02", "2025-06-30", "2025-09-30", "2025-12-31"]:
            pd.testing.assert_frame_equal(M.calculate_factors(self.wide.loc[:cutoff]), full.loc[:cutoff])

    def test_future_mutation_invariance(self):
        cutoff = pd.Timestamp("2025-06-30")
        changed = self.wide.copy()
        changed.loc[changed.index > cutoff, :] *= 7
        pd.testing.assert_frame_equal(M.calculate_factors(changed).loc[:cutoff], M.calculate_factors(self.wide).loc[:cutoff])

    def test_peer_inputs_do_not_enter_a_or_b(self):
        changed = self.wide.copy()
        cols = [c for c in changed if c.startswith(("MSFT_", "NVDA_"))]
        changed[cols] = changed[cols].mul(np.linspace(1, 2, len(changed)), axis=0)
        pd.testing.assert_frame_equal(M.calculate_factors(self.wide)[M.FEATURE_SETS["B"]], M.calculate_factors(changed)[M.FEATURE_SETS["B"]])

    def test_market_inputs_do_not_enter_a(self):
        changed = self.wide.copy()
        cols = [c for c in changed if c.startswith("QQQ_")]
        changed[cols] = changed[cols].mul(np.linspace(1, 2, len(changed)), axis=0)
        pd.testing.assert_frame_equal(M.calculate_factors(self.wide)[M.FEATURE_SETS["A"]], M.calculate_factors(changed)[M.FEATURE_SETS["A"]])

    def test_feature_set_config_and_metadata_exclusion(self):
        config = json.loads((PACKAGE_ROOT / "configs/feature_sets.json").read_text())
        self.assertEqual(config, M.FEATURE_SETS)
        for cols in config.values():
            self.assertTrue(set(cols).issubset(M.FACTOR_COLUMNS))
            self.assertEqual(len(cols), len(set(cols)))

    def test_flat_prices_and_zero_range(self):
        flat = self.wide.iloc[:40].copy()
        for ticker in M.TICKERS:
            for field in ["Open", "High", "Low", "Close"]:
                flat[f"{ticker}_{field}"] = 100.0
            flat[f"{ticker}_Volume"] = 100
        f = M.calculate_factors(flat).iloc[-1]
        self.assertTrue(np.isfinite(f).all())
        np.testing.assert_allclose(f.to_numpy(), 0)

    def assert_invalid(self, frame, message):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, message):
                M._load_market_data(path)

    def test_reject_duplicate(self):
        self.assert_invalid(pd.concat([self.history, self.history.iloc[:1]]), "Duplicate")

    def test_reject_unaligned(self):
        self.assert_invalid(self.history.iloc[1:], "Unaligned")

    def test_reject_missing_or_infinite(self):
        for val in [np.nan, np.inf]:
            frame = self.history.copy()
            frame.loc[0, "Close"] = val
            self.assert_invalid(frame, "Missing|Non-finite")

    def test_reject_bad_ohlc(self):
        frame = self.history.copy()
        frame.loc[0, "High"] = 0.5
        self.assert_invalid(frame, "OHLC")

    def test_reject_wrong_split_and_inconsistent_prices(self):
        for col, value, msg in [("Split", "test", "chronological"), ("Volume", 1, "differ")]:
            bad = self.split.copy()
            bad.loc[0, col] = value
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "bad.csv"
                bad.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, msg):
                    M.build_factor_table(self.hp, path)

    def test_reject_missing_last_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "short.csv"
            self.history.loc[self.history.Date.le("2025-12-31")].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "target"):
                M.build_factor_table(path, self.sp)

    def test_path_resolution_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Cannot find"):
                M.resolve_input("input.csv", repo_root=root)
            a, b = root / "data/processed/input.csv", root / "data/processed_5y_yahoo/input.csv"
            a.parent.mkdir(parents=True)
            a.write_text("a")
            self.assertEqual(M.resolve_input("input.csv", repo_root=root), a)
            b.parent.mkdir(parents=True)
            b.write_text("b")
            with self.assertRaisesRegex(ValueError, "Multiple copies"):
                M.resolve_input("input.csv", repo_root=root)
            self.assertEqual(M.resolve_input("input.csv", explicit=b), b)

    def test_output_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "all.csv"
            M.write_outputs(self.table, out, self.hp, self.sp)
            first = out.read_bytes()
            M.write_outputs(self.table, out, self.hp, self.sp)
            self.assertEqual(first, out.read_bytes())
            for name, part in self.parts.items():
                self.assertEqual(len(pd.read_csv(out.parent / f"stage2_{name}_2025.csv")), len(part))

    def test_prevent_source_overwrite(self):
        with self.assertRaisesRegex(ValueError, "overwrite"):
            M.write_outputs(self.table, self.hp, self.hp, self.sp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
