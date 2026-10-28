"""Checks for Rohan Sai's Michigan browser model and its published evidence."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits

import michigan_boosted_tree as original
import michigan_web_core as web

ROOT = Path(__file__).resolve().parents[1]


class MichiganWebsite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / 'docs/site-data.json').read_text())
        cls.data, cls.features = web.load_browser_data(ROOT / 'docs/data/browser_dataset.csv.gz', cls.manifest)
        cls.events = []
        with patch.object(original, 'metrics', side_effect=AssertionError('Test metrics calculated during training')):
            cls.fitted = web.fit_browser(cls.data, cls.features, lambda n, text: cls.events.append((n, text)))
        cls.result = web.evaluate_browser(cls.fitted)

    def test_real_chronological_sample_and_no_leakage_features(self):
        self.assertEqual(len(self.data), 29000)
        self.assertEqual(len(self.features), 67)
        self.assertFalse(set(self.features) & original.EXCLUDED)
        self.assertEqual(self.data.split.value_counts().to_dict(), {'train':20000, 'test':5000, 'validation':4000})
        self.assertEqual(set(self.data.loc[self.data.split.eq('test')].issue_utc.dt.year), {2024})
        self.assertEqual(self.manifest['owner'], 'Rohan Sai')

    def test_progress_and_separate_test_evaluation(self):
        self.assertEqual(self.events[-1][0], 1)
        self.assertEqual(len([text for _, text in self.events if 'iterations' in text]), 12)
        self.assertNotIn('test_model', self.fitted)
        self.assertEqual(self.result['test_model']['rows'], 5000)
        self.assertEqual(sum(self.result['test_model']['confusion_matrix_tn_fp_fn_tp']), 5000)
        self.assertEqual(self.result['scope'], 'Browser sample experiment')

    def test_warm_start_progress_matches_the_same_180_iteration_estimator(self):
        train = self.data.loc[self.data.split.eq('train')]
        labels = train[web.TARGET].astype(int).to_numpy()
        rate = labels.mean()
        weights = np.where(labels == 1, 0.5 / rate, 0.5 / (1 - rate))
        model = clone(self.fitted['model']).set_params(warm_start=False)
        sample = self.data.loc[self.data.split.eq('test'), self.features].head(100)
        with threadpool_limits(limits=1):
            model.fit(train[self.features], labels, sample_weight=weights)
            np.testing.assert_allclose(model.predict_proba(sample), self.fitted['model'].predict_proba(sample), atol=1e-10)

    def test_prediction_excludes_actual_outcomes(self):
        frame = self.data.loc[self.data.split.eq('test')].head(10)
        result = web.predict_rows(self.fitted, frame)
        changed = frame.assign(target_high_next_3h=1, target_peak_next_3h=999999, county_train_p90=1)
        again = web.predict_rows(self.fitted, changed)
        np.testing.assert_allclose(result.model_score_uncalibrated, again.model_score_uncalibrated)
        self.assertEqual(len(result), 10)

    def test_invalid_inputs_rejected_and_missing_values_supported(self):
        frame = self.data.loc[self.data.split.eq('test')].head(2)
        with self.assertRaisesRegex(ValueError, 'Missing predictor'):
            web.predict_rows(self.fitted, frame.drop(columns=self.features[0]))
        with self.assertRaisesRegex(ValueError, 'finite numbers'):
            web.predict_rows(self.fitted, frame.assign(temperature_c=float('inf')))
        with self.assertRaisesRegex(ValueError, 'finite numbers'):
            web.predict_rows(self.fitted, frame.assign(temperature_c='bad'))
        with self.assertRaisesRegex(ValueError, 'unique'):
            web.read_upload('temperature_c,temperature_c\n1,2')
        result = web.predict_rows(self.fitted, frame.assign(temperature_c=np.nan))
        self.assertEqual(len(result), 2)

    def test_published_evidence_and_code_copies(self):
        saved = json.loads((ROOT / 'docs/evaluation.json').read_text())
        self.assertEqual(saved['test_model']['rows'], 45804)
        self.assertAlmostEqual(saved['test_model']['accuracy'], 0.9380185136669287)
        self.assertEqual(saved['test_model']['confusion_matrix_tn_fp_fn_tp'], [39800,815,2024,3165])
        self.assertTrue(saved['roc'] and saved['pr'])
        for name in ['michigan_boosted_tree.py', 'michigan_web_core.py']:
            self.assertEqual((ROOT/name).read_bytes(), (ROOT/'docs/runtime'/name).read_bytes())


if __name__ == '__main__':
    unittest.main()
