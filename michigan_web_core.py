"""Rohan Sai's staged Michigan browser experiment, sharing the original model rules."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve
from threadpoolctl import threadpool_limits

import michigan_boosted_tree as original

TARGET = 'target_high_next_3h'


def load_browser_data(path, manifest):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != manifest['browser_dataset_sha256']:
        raise ValueError('Browser dataset checksum mismatch. Restore the published sample.')
    raw = pd.read_csv(path, compression='gzip', dtype={'fips': str})
    data, _ = original.clean_dataset(raw)
    features = manifest['features']
    if len(data) != len(raw) or data.duplicated(['issue_utc', 'fips']).any():
        raise ValueError('Invalid or duplicate county/time rows in the browser dataset.')
    if len(features) != 67 or set(features) & original.EXCLUDED:
        raise ValueError('Feature schema is incomplete or contains target/identifier leakage.')
    if not data[TARGET].isin([0, 1]).all():
        raise ValueError('Only complete, binary target windows may enter this experiment.')
    if not data[TARGET].astype(int).eq(data.target_peak_next_3h.ge(data.county_train_p90).astype(int)).all():
        raise ValueError('Labels disagree with the future peak and training county threshold.')
    for split, expected in [('train', set(range(2014, 2023))), ('validation', {2023}), ('test', {2024})]:
        block = data.loc[data.split.eq(split)]
        if set(block.issue_utc.dt.year) != expected or block[TARGET].nunique() != 2:
            raise ValueError(f'{split} must preserve its original years and both labels.')
        if len(block) != manifest['browser_rows'][split]:
            raise ValueError(f'{split} row count differs from the published manifest.')
    return data, features


def fit_browser(data, features, progress=None):
    """Fit on earlier years, then choose the threshold on 2023; never score 2024."""
    report = progress or (lambda fraction, text: None)
    train = data.loc[data.split.eq('train')]
    validation = data.loc[data.split.eq('validation')]
    labels = train[TARGET].astype(int).to_numpy()
    positive_rate = float(labels.mean())
    weights = np.where(labels == 1, 0.5 / positive_rate, 0.5 / (1 - positive_rate))
    model = HistGradientBoostingClassifier(
        max_iter=15, max_leaf_nodes=15, learning_rate=0.05, min_samples_leaf=40,
        l2_regularization=2, early_stopping=False, random_state=42, warm_start=True)
    with threadpool_limits(limits=1):
        for rounds in range(15, 181, 15):
            report(0.05 + 0.8 * (rounds - 15) / 180, f'Training boosted trees: iterations {rounds - 14}–{rounds} of 180')
            model.set_params(max_iter=rounds)
            model.fit(train[features], labels, sample_weight=weights)
        report(0.9, 'Selecting the alert threshold using 2023 validation records only')
        scores = model.predict_proba(validation[features])[:, 1]
    selection = original.choose_threshold(validation[TARGET], scores)
    report(1.0, 'Training and validation complete. The 2024 test has not been scored.')
    return {'model': model, 'data': data, 'features': features, 'threshold_selection': selection}


def curve_points(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    def compact(x, y):
        chosen = np.unique(np.linspace(0, len(x) - 1, min(500, len(x))).astype(int))
        return [[float(x[i]), float(y[i])] for i in chosen]
    return {'roc': compact(fpr, tpr), 'pr': compact(recall[::-1], precision[::-1])}


def evaluate_browser(fitted, progress=None):
    report = progress or (lambda fraction, text: None)
    test = fitted['data'].loc[fitted['data'].split.eq('test')]
    selection = fitted['threshold_selection']
    threshold = selection['threshold'] if selection['feasible'] else float('inf')
    report(0.1, f'Testing the fixed model and threshold on {len(test):,} historical 2024 rows')
    with threadpool_limits(limits=1):
        scores = fitted['model'].predict_proba(test[fitted['features']])[:, 1]
    measured = original.metrics(test[TARGET], scores, threshold)
    measured['threshold_status'] = 'chosen on 2023 only' if selection['feasible'] else 'abstain: validation cap unsupported'
    report(0.65, 'Building ROC/precision-recall curves, confusion matrix and class report')
    result = {
        'scope': 'Browser sample experiment', 'model': 'HistGradientBoostingClassifier',
        'rows': fitted['data'].split.value_counts().to_dict(),
        'threshold_selection': selection, 'test_model': measured,
        'baseline': original.metrics(test[TARGET], np.zeros(len(test)), 0.5),
        'report': classification_report(test[TARGET], scores >= threshold, labels=[0, 1],
                                         target_names=['No high burden', 'High burden'], zero_division=0),
        **curve_points(test[TARGET], scores),
    }
    report(1.0, 'Historical testing complete. Predictions are now available.')
    return result


def predict_rows(fitted, frame):
    features = fitted['features']
    if frame.empty or len(frame) > 1000:
        raise ValueError('Provide between 1 and 1,000 records.')
    if frame.columns.duplicated().any() or len(set(str(c).strip().lower() for c in frame)) != len(frame.columns):
        raise ValueError('CSV headers must be unique, ignoring case and spaces.')
    missing = set(features) - set(frame)
    if missing:
        raise ValueError('Missing predictor columns: ' + ', '.join(sorted(missing)))
    extras = set(frame) - set(features) - original.EXCLUDED
    if extras:
        raise ValueError('Unexpected columns: ' + ', '.join(sorted(extras)))
    inputs = frame[features].apply(pd.to_numeric, errors='coerce')
    if (frame[features].notna() & inputs.isna()).any().any() or np.isinf(inputs.to_numpy(dtype=float)).any():
        raise ValueError('Features must contain finite numbers or blank cells. Missing inputs are kept as missing.')
    with threadpool_limits(limits=1):
        scores = fitted['model'].predict_proba(inputs)[:, 1]
    selection = fitted['threshold_selection']
    threshold = selection['threshold'] if selection['feasible'] else float('inf')
    result = pd.DataFrame({'record': np.arange(1, len(frame) + 1),
                           'model_score_uncalibrated': scores,
                           'threshold_from_2023': selection['threshold'],
                           'high_burden_alert': scores >= threshold,
                           'missing_inputs': inputs.isna().sum(axis=1).to_numpy()})
    for column in ['county', 'fips', 'issue_utc', TARGET]:
        if column in frame:
            result[column] = frame[column].astype(str).to_numpy() if column == 'issue_utc' else frame[column].to_numpy()
    return result


def read_upload(text):
    import csv
    if len(text.encode('utf-8')) > 5 * 1024 * 1024:
        raise ValueError('Use a CSV of at most 5 MB.')
    headers = next(csv.reader(io.StringIO(text.lstrip('\ufeff'))), [])
    if not headers or len(set(h.strip().lower() for h in headers)) != len(headers):
        raise ValueError('CSV headers must be nonempty and unique.')
    return pd.read_csv(io.StringIO(text), nrows=1001)


def records(frame):
    return json.loads(frame.to_json(orient='records', date_format='iso'))
