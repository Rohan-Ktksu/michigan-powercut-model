"""Rohan Sai's browser bridge for the Michigan county outage-burden experiment."""
import json
import platform
from pathlib import Path

import pandas as pd
import sklearn
from js import report_progress
import michigan_web_core as core


def browser_train():
    global fitted
    manifest = json.loads(Path('/michigan/site-data.json').read_text())
    report_progress(0.03, 'Verifying the published subset checksum and chronological partitions')
    data, features = core.load_browser_data('/michigan/data/browser_dataset.csv.gz', manifest)
    fitted = core.fit_browser(data, features, lambda n, text: report_progress(n, text))
    return json.dumps({'threshold_selection': fitted['threshold_selection'],
                       'rows': data.split.value_counts().to_dict(), 'features': len(features),
                       'python': platform.python_version(), 'sklearn': sklearn.__version__})


def browser_test():
    result = core.evaluate_browser(fitted, lambda n, text: report_progress(n, text))
    result['execution'] = {'python': platform.python_version(), 'scikit-learn': sklearn.__version__, 'threads': 1}
    return json.dumps(result)


def browser_predict(request_json):
    request = json.loads(request_json)
    if request.get('csv') is not None:
        frame = core.read_upload(request['csv'])
    else:
        pool = fitted['data'].loc[fitted['data'].split.eq('test')]
        county = request.get('county', 'All counties')
        if county != 'All counties':
            pool = pool.loc[pool.county.eq(county)]
        count = request.get('count', 10)
        if not isinstance(count, int) or not 1 <= count <= 100:
            raise ValueError('Choose 1–100 records.')
        if pool.empty:
            raise ValueError('No test records are available for that county.')
        frame = pool.sample(n=min(count, len(pool)), random_state=42)
    result = core.predict_rows(fitted, frame)
    return json.dumps({'results': core.records(result), 'inputs': core.records(frame[fitted['features']])})


def browser_template():
    return fitted['data'].loc[fitted['data'].split.eq('test'), fitted['features']].head(5).to_csv(index=False)
