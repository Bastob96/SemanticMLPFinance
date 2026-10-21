import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.ui import load_dashboard_data


class DashboardDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dashboard_data()

    def test_stage1_bundle_has_every_configuration_and_year(self):
        frame = self.data.stage1
        self.assertEqual(set(frame.year), {2022, 2023, 2024, 2025})
        self.assertEqual(set(frame.split), {"validation", "test"})
        self.assertEqual(len(frame), 64)
        self.assertFalse(frame.duplicated(["year", "split", "experiment", "model"]).any())
        self.assertTrue(frame.balanced_accuracy_mean.between(0, 1).all())
        self.assertTrue(frame.auc_mean.between(0, 1).all())

    def test_stage1_backtests_are_validation_selected_q4_windows(self):
        self.assertEqual(len(self.data.stage1_selections), 4)
        self.assertTrue(
            all(row["selection_split"] == "validation" for row in self.data.stage1_selections)
        )
        daily = self.data.stage1_backtest_daily
        self.assertTrue(daily.Date.dt.quarter.eq(4).all())
        self.assertTrue(daily.target_date.dt.quarter.eq(4).all())
        self.assertTrue(daily.strategy_equity.gt(0).all())

    def test_stage2_pinned_runs_and_keep_decisions(self):
        board = self.data.stage2_scoreboard
        self.assertEqual(len(board.loc[board.proposer.eq("claude")]), 48)
        self.assertEqual(len(board.loc[board.proposer.eq("random")]), 48)
        kept = board.loc[board.status.eq("kept")].set_index("proposer")
        self.assertEqual(kept.loc["claude", "name"], "nvda_range_vs_aapl_range")
        self.assertEqual(kept.loc["random", "name"], "random_665013")
        self.assertTrue(kept.years_helped.eq(3).all())
        self.assertTrue(kept.auc_gain.ge(0.01).all())

    def test_stage2_final_results_are_complete_and_baseline_is_identical(self):
        final = self.data.stage2_final_metrics
        self.assertEqual(len(final), 32)
        self.assertEqual(set(final.year), {2022, 2023, 2024, 2025})
        self.assertFalse(final.isna().any().any())
        baseline = final.loc[final.features.eq("stage1_22")]
        claude = baseline.loc[baseline.proposer.eq("claude")].sort_values(["year", "model"])
        random = baseline.loc[baseline.proposer.eq("random")].sort_values(["year", "model"])
        pd.testing.assert_frame_equal(
            claude.drop(columns=["proposer", "run_id"]).reset_index(drop=True),
            random.drop(columns=["proposer", "run_id"]).reset_index(drop=True),
        )

    def test_bad_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "manifest.json").write_text(json.dumps({"schema_version": 999}))
            with self.assertRaises(ValueError):
                load_dashboard_data(path)


if __name__ == "__main__":
    unittest.main()
