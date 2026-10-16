# Michigan Power-Outage Research Model (2014-2024)

This package contains a county-level, three-hour high-outage-burden research model. It uses the EAGLE-I outage archive and historical Michigan ASOS weather observations. It is not a household-level outage forecast or a live alert service. The complete runnable workflow is in one Python file: `michigan_boosted_tree.py`.

## Data and time split

- Prepared dataset: `michigan_short_horizon_dataset.csv` (363,419 county/time windows; 83 counties; 67 model inputs).
- Available source coverage: 2014-2024. 2014 is partial, beginning November 1; it contributes 3,371 usable windows.
- Training: 2014-2022 (272,611 labeled windows).
- Threshold selection: 2023 (45,004 labeled windows).
- Fixed historical test: 2024 (45,804 labeled windows).
- 2025 is excluded because the audited Michigan EAGLE-I source has no explicit zero readings.
- `source_audit.json` records annual EAGLE-I checksums, source links, row counts, weather coverage, and assumptions.

The target is whether any complete hourly county mean in the next three hours reaches that county's 2014-2022 training-period 90th percentile. Missing/incomplete target windows are not converted to negatives. Historical station weather is spatially aggregated and shifted under an assumed two-hour availability delay; that delay is not independently verified. The model uses weather, storm-code, calendar, location, and past-outage features. Historical NWS warning inputs were omitted because verified warning archives were not available consistently for every year.

## Run

From this directory, install the required packages if needed and run:

```powershell
py -3 -m pip install pandas numpy scikit-learn joblib
py -3 michigan_boosted_tree.py
```

The script normalizes timestamps and county identifiers, converts malformed numeric cells to missing without inventing values, validates the bundled data and target, profiles it, fits the classifier, chooses its threshold using 2023 only, scores the fixed 2024 test, and writes:

- `outputs/results.json` - profiling, validation, protocol, and test metrics.
- `outputs/test_2024_predictions.csv` - one historical score per 2024 county/time window.
- `outputs/michigan_boosted_tree_model.joblib` - fitted model, input-column list, threshold, and protocol.

The prepared CSV and model script have a checksum guard so accidental dataset edits fail clearly. The script does not scrape or rebuild source data.

## Current historical test result

On the 2024 test at the threshold selected on 2023: precision 79.5%, recall 61.0%, F1 0.690, average precision 0.733, and ROC AUC 0.889. This is a retrospective research result, not a present-day reliability guarantee. The 2024 period was previously inspected in the broader project, and the EAGLE-I measurement coverage and zero-reporting behavior vary by year. The alert threshold targeted a validation false-positive rate no higher than 2%; the observed 2024 rate was 2.01%.
