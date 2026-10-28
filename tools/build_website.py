"""Prepare Rohan Sai's Michigan website from the checksum-verified original data."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import michigan_boosted_tree as original
from michigan_web_core import curve_points, records


def feature_group(name):
    if 'outage' in name: return 'Recent outage history'
    if name in ['latitude', 'longitude', 'nearest_weather_station_km']: return 'Geography & station distance'
    if name in ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month']: return 'Calendar cycles'
    if '_lag_' in name or '_max_' in name or 'change' in name: return 'Weather history'
    if name in ['thunderstorm', 'squall', 'heavy_rain']: return 'Reported storm codes'
    return 'Weather observations'


def build(source):
    original.DATA_PATH = Path(source)
    data, features, validation = original.load_and_validate()
    caps = {'train': 20000, 'validation': 4000, 'test': 5000}
    sample = pd.concat([data.loc[data.split.eq(split)].sample(n=cap, random_state=42) for split, cap in caps.items()])
    sample = sample.sort_values(['issue_utc', 'fips']).reset_index(drop=True)
    docs = ROOT / 'docs'
    (docs / 'data').mkdir(parents=True, exist_ok=True)
    (docs / 'runtime').mkdir(exist_ok=True)
    packed = gzip.compress(sample.to_csv(index=False).encode('utf-8'), mtime=0)
    (docs / 'data/browser_dataset.csv.gz').write_bytes(packed)
    manifest = {
        'owner': 'Rohan Sai', 'account': 'Rohan-Ktksu', 'original_dataset_sha256': validation['dataset_sha256'],
        'original_rows': int(len(data)), 'browser_rows': caps, 'browser_dataset_sha256': hashlib.sha256(packed).hexdigest(),
        'features': features, 'counties': sorted(data.county.unique().tolist()),
        'schema': [{'feature': c, 'group': feature_group(c), 'missing_in_full_data': int(data[c].isna().sum())} for c in features],
        'preview': records(sample.head(6)), 'excluded': sorted(original.EXCLUDED),
        'sampling': 'Fixed seed 42 random samples within the original chronological train/validation/test partitions; no new targets generated.',
        'full_rows': data.split.value_counts().to_dict(),
    }
    (docs / 'site-data.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    saved = json.loads((ROOT / 'outputs/results.json').read_text(encoding='utf-8'))
    predicted = pd.read_csv(ROOT / 'outputs/test_2024_predictions.csv')
    recomputed = original.metrics(predicted.target, predicted.model_score_uncalibrated, saved['threshold_selection']['threshold'])
    for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']:
        if not np.isclose(recomputed[key], saved['test_model'][key], atol=1e-10):
            raise ValueError(f'Saved predictions and report disagree on {key}.')
    published = {
        'scope': 'Published full-dataset research run', 'model': 'HistGradientBoostingClassifier',
        'rows': saved['rows'], 'threshold_selection': saved['threshold_selection'],
        'test_model': saved['test_model'], 'baseline': saved['test_always_no_alert_baseline'],
        'report': 'Confusion-matrix counts and metrics were verified against outputs/test_2024_predictions.csv. This reproduces a previously inspected 2024 historical test, not a new independent evaluation.',
        **curve_points(predicted.target, predicted.model_score_uncalibrated),
    }
    (docs / 'evaluation.json').write_text(json.dumps(published, indent=2), encoding='utf-8')
    for name in ['michigan_boosted_tree.py', 'michigan_web_core.py']:
        shutil.copyfile(ROOT / name, docs / 'runtime' / name)
    (docs / '.nojekyll').touch()
    print(f'Website assets ready: {len(sample):,} browser records, {len(features)} real features, {len(packed):,} compressed bytes.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=original.DATA_PATH)
    build(parser.parse_args().source)
