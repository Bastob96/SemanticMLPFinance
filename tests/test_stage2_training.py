import unittest
from pathlib import Path

from src.features import FEATURE_SETS as STAGE1_FEATURE_SETS
from src.stage2 import (
    PRIMARY_EXPERIMENTS,
    build_experiment_specs,
    build_feature_explanation,
    build_predictions,
    load_integrated_splits,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/aligned_daily_ohlcv_5y.csv"
STAGE2 = ROOT / "Stage2_Factor_Files/data/processed/quarterly"
FEATURE_SETS = ROOT / "Stage2_Factor_Files/configs/feature_sets.json"


class Stage2TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits, cls.stage2_sets = load_integrated_splits(
            DATA, STAGE2, FEATURE_SETS, include_test=False
        )

    def test_validation_phase_does_not_load_test(self):
        self.assertEqual(set(self.splits), {"train", "validation"})
        self.assertEqual({k: len(v) for k, v in self.splits.items()}, {"train": 121, "validation": 63})
        self.assertLess(self.splits["train"].target_date.max(), self.splits["validation"].Date.min())

    def test_experiment_feature_counts_and_additive_design(self):
        specs = build_experiment_specs(self.stage2_sets)
        self.assertEqual(tuple(specs), PRIMARY_EXPERIMENTS)
        self.assertEqual({name: len(cols) for name, cols in specs.items()}, {
            "A_stage1": 14, "B_stage1": 17, "B_stage2": 21,
            "C_stage1": 22, "C_stage2": 30,
        })
        self.assertEqual(specs["C_stage2"][:22], STAGE1_FEATURE_SETS["C"])
        self.assertEqual(specs["C_stage2"][22:], self.stage2_sets["C"])

    def test_predictions_match_shared_schema_and_seed_policy(self):
        specs = {"A_stage1": STAGE1_FEATURE_SETS["A"]}
        predictions = build_predictions(self.splits, specs, ("validation",))
        self.assertEqual(predictions.columns.tolist(), [
            "Date", "target_date", "actual", "predicted", "probability_up",
            "split", "experiment", "model", "seed",
        ])
        self.assertEqual(set(predictions.split), {"validation"})
        self.assertEqual(set(predictions.loc[predictions.model.eq("random_forest"), "seed"]), {0, 1, 2})
        self.assertTrue(predictions.loc[predictions.model.eq("logistic_regression"), "seed"].eq(42).all())
        self.assertTrue(predictions.probability_up.between(0, 1).all())

    def test_explanation_is_train_only_and_covers_all_features(self):
        specs = build_experiment_specs(self.stage2_sets)
        explanation = build_feature_explanation(
            self.splits["train"], specs["C_stage2"], self.stage2_sets["C"]
        )
        self.assertEqual(len(explanation), 30)
        self.assertEqual(explanation.source.value_counts().to_dict(), {
            "conventional": 22, "llm_candidate": 8,
        })
        self.assertTrue(explanation.rf_importance_mean.ge(0).all())


if __name__ == "__main__":
    unittest.main()
