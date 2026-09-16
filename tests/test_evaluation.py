import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.evaluation import (
    METRICS, aggregate_runs, calculate_classification_metrics, comparison_table,
    evaluate_predictions, load_predictions, select_best_configuration,
)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = pd.read_csv('results/stage1_predictions.csv')

    def test_hand_calculated_confusion_and_rank_auc(self):
        # TP=2, TN=1, FP=1, FN=1. Four of six positive/negative score pairs win.
        got = calculate_classification_metrics([1, 0, 1, 0, 1], [1, 1, 0, 0, 1], [.9, .8, .2, .1, .7])
        expected = dict(accuracy=3/5, balanced_accuracy=7/12, auc=4/6,
                        precision=2/3, recall=2/3, f1=2/3)
        for name, value in expected.items():
            self.assertAlmostEqual(got[name], value)

    def test_auc_uses_scores_and_counts_ties(self):
        result = calculate_classification_metrics([0, 1, 0, 1], [0, 0, 0, 0], [.1, .4, .4, .9])
        self.assertAlmostEqual(result['auc'], .875)
        self.assertEqual(result['precision'], 0)
        self.assertEqual(result['recall'], 0)
        self.assertEqual(result['f1'], 0)

    def test_undefined_auc_is_explicit(self):
        self.assertTrue(math.isnan(calculate_classification_metrics([0, 1], [0, 1])['auc']))
        result = calculate_classification_metrics([0, 0], [0, 0], [.2, .3])
        self.assertTrue(math.isnan(result['auc']))
        self.assertEqual(result['balanced_accuracy'], 1)

    def test_bad_metric_inputs_rejected(self):
        cases = [([], [], []), ([0], [0, 1], [.2]), ([2], [1], [.2]),
                 ([0], [np.nan], [.2]), ([[0]], [0], [.2]),
                 ([0, 1], [0, 1], [np.inf, .2]), ([0], [0], [1.2]),
                 ([0], [0], []), ([0], [0], [np.nan])]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                calculate_classification_metrics(*args)

    def test_saved_csv_roundtrip_and_no_mutation(self):
        before = self.saved.copy(deep=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'predictions.csv'
            before.to_csv(path, index=False)
            pd.testing.assert_frame_equal(evaluate_predictions(path), evaluate_predictions(before))
        pd.testing.assert_frame_equal(before, self.saved)

    def test_real_saved_metrics_match_committed_results(self):
        got = evaluate_predictions(self.saved)
        expected = pd.read_csv('results/stage1_metrics_provisional_detailed.csv')
        keys = ['split', 'experiment', 'model', 'seed']
        pd.testing.assert_frame_equal(got.set_index(keys)[list(METRICS)].sort_index(),
                                      expected.set_index(keys)[list(METRICS)].sort_index(),
                                      check_exact=False, atol=1e-14, rtol=1e-14)
        self.assertEqual(len(got), 28)
        self.assertTrue(got.n_observations.eq(63).all())

    def test_independent_real_confusion_and_pairwise_auc(self):
        for _, group in self.saved.groupby(['split', 'experiment', 'model', 'seed'], dropna=False):
            y, p, scores = group.actual.to_numpy(), group.predicted.to_numpy(), group.probability_up.to_numpy()
            tp = sum((y == 1) & (p == 1)); tn = sum((y == 0) & (p == 0))
            fp = sum((y == 0) & (p == 1)); fn = sum((y == 1) & (p == 0))
            positives, negatives = scores[y == 1], scores[y == 0]
            auc = sum(float(a > b) + .5*float(a == b) for a in positives for b in negatives)/(len(positives)*len(negatives))
            expected = [(tp+tn)/len(y), .5*(tp/(tp+fn)+tn/(tn+fp)), auc,
                        tp/(tp+fp) if tp+fp else 0, tp/(tp+fn), 2*tp/(2*tp+fp+fn)]
            got = calculate_classification_metrics(y, p, scores)
            for metric, value in zip(METRICS, expected):
                self.assertAlmostEqual(got[metric], value)

    def test_rf_sample_sd_and_all_comparisons(self):
        detailed = evaluate_predictions(self.saved)
        table = aggregate_runs(detailed)
        self.assertEqual(len(table), 16)
        for _, row in table.iterrows():
            subset = detailed.loc[detailed.split.eq(row.split) & detailed.experiment.eq(row.experiment) & detailed.model.eq(row.model)]
            for metric in METRICS:
                values = subset[metric].tolist(); mean = sum(values)/len(values)
                self.assertAlmostEqual(row[f'{metric}_mean'], mean)
                if row.model == 'random_forest':
                    self.assertAlmostEqual(row[f'{metric}_std'], math.sqrt(sum((v-mean)**2 for v in values)/2))
                else:
                    self.assertTrue(math.isnan(row[f'{metric}_std']))
        a = table.loc[table.split.eq('test') & table.experiment.eq('A') & table.model.eq('random_forest')].iloc[0]
        self.assertAlmostEqual(a.accuracy_mean, 89/189)

    def test_incomplete_duplicate_seeds_rejected(self):
        detailed = evaluate_predictions(self.saved)
        for broken in [detailed.loc[~detailed.seed.eq(2)], pd.concat([detailed, detailed.iloc[[0]]])]:
            with self.assertRaises(ValueError):
                aggregate_runs(broken)

    def test_schema_date_and_alignment_errors_rejected(self):
        mutations = []
        mutations.append(self.saved.drop(columns='probability_up'))
        mutations.append(self.saved.iloc[:0])
        mutations.append(pd.concat([self.saved, self.saved.iloc[[0]]]))
        for column, value in [('split', 'train'), ('seed', 99), ('Date', None),
                              ('target_date', '2000-01-01'), ('actual', 2),
                              ('probability_up', np.nan), ('model', 'unknown'), ('experiment', 'A')]:
            broken = self.saved.copy(); broken.loc[0, column] = value; mutations.append(broken)
        broken = self.saved.copy(); broken.loc[0, 'actual'] = 0; mutations.append(broken)
        mutations.append(self.saved.drop(index=0))
        for i, broken in enumerate(mutations):
            with self.subTest(case=i), self.assertRaises(ValueError):
                load_predictions(broken)

    def test_selection_ignores_test_scores_and_fixes_seed(self):
        table = comparison_table(self.saved)
        selected = select_best_configuration(table)
        self.assertEqual((selected['experiment'], selected['model'], selected['seed']), ('B', 'logistic_regression', 42))
        table.loc[table.split.eq('test'), 'balanced_accuracy_mean'] = 999
        self.assertEqual(select_best_configuration(table), selected)
        table.loc[table.split.eq('validation') & table.model.eq('random_forest') & table.experiment.eq('A'), 'balanced_accuracy_mean'] = 1
        self.assertEqual(select_best_configuration(table)['seed'], 0)
        with self.assertRaises(ValueError):
            select_best_configuration(table.loc[table.split.eq('test')])


if __name__ == '__main__':
    unittest.main()
