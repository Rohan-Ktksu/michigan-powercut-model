# Michigan County Power-Outage Burden Research

**A historical, weather-enriched, three-hour high-burden alert model for Michigan counties (2014-2024)**

This repository contains a reproducible research run that combines county-level outage observations with historical weather, storm, location, calendar, and recent-outage features. The model estimates whether an observed county outage-burden measure will cross a historical county-specific threshold in the next three hours.

> **Scope:** This is a retrospective county-level research model. It does not predict whether a particular home will lose power, identify an outage's cause, estimate restoration time, or provide a live operational forecast. Its score is not a calibrated probability.

## Research journey

The initial project idea was a practical power-cut predictor: identify higher-risk times and areas, relate outages to weather and possible causes, and eventually explore restoration-time prediction. Early work focused on Bengaluru/BESCOM public notices and lab exercises. That material was useful for learning data collection, cleaning, feature engineering, and validation, but public notices did not provide a consistent, verified, long-term record of both outages and confirmed no-outage periods. Transformer and reliability reports were also not equivalent to event-level outage/restoration records.

The research direction then shifted to Michigan because an already downloaded EAGLE-I archive provided a substantially larger, structured, county-level historical dataset, and could be joined with historical weather observations. The task was narrowed to one the available measurements could support: predicting **high county outage burden over a short horizon**, rather than claiming household-level outage occurrence or restoration prediction.

The work proceeded through source auditing, year-by-year coverage checks, preparation of county/time windows, missing-value and target validation, feature construction, chronological model development, threshold selection on a separate year, and a fixed later-year evaluation. A single Python file now packages the modeling, validation, training, scoring, example generation, and output-writing steps.

## Research question and target

**Question:** Given a county's known outage history and available weather/location/calendar information at an issue time, can a model identify whether the county's outage burden will be unusually high during the next three hours?

Each row is a county/time prediction window, issued on a three-hour UTC cadence. The positive target is defined from the maximum of the next three **complete hourly county means**: the target is positive when at least one reaches or exceeds that county's 90th-percentile outage threshold calculated from complete training-period observations in 2014-2022. The threshold has a lower bound of one customer. This is a relative, county-specific high-burden label, not a universal number of customers and not an individual-customer interruption label.

Incomplete future target windows are excluded. Missing readings are not converted to zero, because an absent record is not proof that no outage occurred.

## Data sources and coverage

### Outages

The project uses the EAGLE-I Electric Grid Monitor archive distributed through Figshare. The archive contains estimated outage/customer-impact observations; coverage and zero-reporting behavior vary across years. The annual source links, checksums, row counts, and audit details are documented in [`source_audit.json`](source_audit.json).

- Source record: [EAGLE-I Electric Grid Monitor archive](https://doi.org/10.6084/m9.figshare.24237376)
- Study geography: 83 Michigan counties
- Source years used: 2014-2024
- 2014 is partial, starting November 1
- 2025 is excluded: the audited Michigan source did not provide explicit zero readings needed for the project's labeling rules

### Weather

Historical observations are drawn from Iowa Environmental Mesonet (IEM) ASOS station data, using up to 75 Michigan stations and spatial aggregation to counties. Available inputs include temperature, dew point, humidity, wind, gust, precipitation, pressure, and METAR-reported thunderstorm, squall, and heavy-rain codes.

- Source: [IEM ASOS data download](https://mesonet.agron.iastate.edu/request/download.phtml)
- Weather coverage in the packaged source audit is seasonal (approximately late May through early September), not year-round
- The model assumes a two-hour observation availability delay; that historical reporting latency was not independently verified
- These are historical station observations, not archived operational weather forecasts
- Historical NWS warning features were omitted because year-complete, validated warning records were not available for the study period

### Prepared dataset

[`michigan_short_horizon_dataset.csv`](michigan_short_horizon_dataset.csv) contains **363,419 rows and 74 columns**: 67 candidate model inputs plus target, county/time identifiers, and supporting fields. It spans November 2014 through December 2024 and covers 83 counties. The packaged CSV is a prepared dataset; this repository does not include the raw national EAGLE-I archives or a source-data rebuilding/scraping pipeline.

## What the model uses

The 67 candidate inputs are numeric and fall into these groups:

| Feature group | Examples |
| --- | --- |
| Weather | Temperature, dew point, humidity, wind, gust, precipitation, pressure |
| Storm observations | METAR thunderstorm, squall, and heavy-rain codes |
| Weather history | Lagged observations and rolling maxima over 6, 12, 24, and 48 hours |
| Outage history | Last observed county count, recent means/peaks, and recent reporting coverage |
| Time and geography | County coordinates, distance to nearest weather station, cyclical hour/weekday, and month |

The implementation does **not** report a validated global feature-importance ranking. The feature list alone should not be interpreted as evidence that a specific cause (for example, maintenance or a storm) drove any prediction. The model is associational and the data do not provide a verified causal label for outage causes.

## Cleaning and validation

The one-file workflow in [`michigan_boosted_tree.py`](michigan_boosted_tree.py) performs the packaged run:

1. Checks the prepared dataset checksum against the audited copy.
2. Normalizes county identifiers and timestamps; checks duplicate county/time keys and invalid numeric values.
3. Preserves missing inputs rather than fabricating values. The selected HistGradientBoosting implementation handles missing inputs natively.
4. Validates target and time-split consistency. Only complete next-three-hour windows are eligible; unknown/incomplete targets are not relabeled as negatives.
5. Fits the model using earlier years, chooses the alert threshold on a later validation year, and evaluates the fixed 2024 period only after those choices are fixed.

The packaged validation passed: no duplicate county/time keys were found, no invalid key rows or malformed numeric values required removal in this run, and the dataset checksum matched the audited value. There are many missing feature cells, primarily because weather observations are seasonal and station coverage is incomplete; these missing values were preserved.

## Evaluation protocol

| Period | Rows | Use |
| --- | ---: | --- |
| 2014-2022 | 272,611 | Model fitting and county-specific training thresholds |
| 2023 | 45,004 | Alert-threshold selection only |
| 2024 | 45,804 | Fixed historical evaluation |
| **Total** | **363,419** | Prepared county/time windows |

The classifier is scikit-learn `HistGradientBoostingClassifier` with 180 boosting iterations, a maximum of 15 leaves per tree, learning rate 0.05, minimum leaf size 40, L2 regularization 2, and a fixed random seed. Inverse-frequency sample weights address class imbalance.

The alert threshold was selected using 2023 only: maximize recall subject to a false-positive rate no greater than 2% and at least 25 alerts. The resulting threshold is approximately **0.7743**. This constrains the validation period, not future periods, and the output scores are uncalibrated.

## End result: 2024 historical test

The 2024 test contains 45,804 eligible windows; 11.33% have a positive high-burden label. At the threshold selected on 2023, the model produced:

| Metric | 2024 result | Plain-language meaning |
| --- | ---: | --- |
| Accuracy | 93.80% | Fraction of all windows classified correctly |
| Balanced accuracy | 79.49% | Average recall across positive and negative windows |
| Precision | 79.52% | Fraction of issued alerts that matched the positive label |
| Recall | 60.99% | Fraction of positive windows that were alerted |
| F1 score | 0.690 | Combined precision/recall summary |
| Average precision | 0.733 | Ranking performance across score thresholds |
| ROC AUC | 0.889 | Ranking separation across score thresholds |
| False-positive rate | 2.01% | Fraction of negative windows incorrectly alerted |
| False alerts | 815 | Negative windows flagged as alerts |
| Missed positive windows | 2,024 | Positive windows not flagged |

Confusion matrix (actual rows: negative, positive; predicted columns: no alert, alert):

|  | Predicted no alert | Predicted alert |
| --- | ---: | ---: |
| Actual negative | 39,800 | 815 |
| Actual positive | 2,024 | 3,165 |

The always-no-alert baseline achieves 88.67% accuracy but zero recall and zero precision for alerts. This illustrates why accuracy alone is not a sufficient measure for this imbalanced task. The model's 79.52% precision is a retrospective result for this particular label, dataset, split, and selected threshold; it does not mean an alert has a 79.52% calibrated chance of a real-world outage.

## Historical example outputs

The script saves six illustrative 2024 examples in `outputs/results.json`. They are selected from the predicted-alert and predicted-no-alert groups by model score without using the actual label for selection. Times are UTC, and the score is **uncalibrated**.

| Issue time (UTC) | County | Model score | Output | Historical label |
| --- | --- | ---: | --- | --- |
| 2024-06-26 06:00 | Muskegon County | 0.998 | Alert | High burden |
| 2024-08-28 18:00 | Clare County | 0.998 | Alert | High burden |
| 2024-06-26 09:00 | Muskegon County | 0.998 | Alert | High burden |
| 2024-07-10 09:00 | Allegan County | 0.774 | No alert | No high burden |
| 2024-03-09 03:00 | Wayne County | 0.774 | No alert | No high burden |
| 2024-11-11 15:00 | Marquette County | 0.774 | No alert | High burden |

These are historical demonstrations, not current forecasts. The Marquette example is a missed positive and makes clear that a no-alert output is not a guarantee of no outage burden.

## Run the project

From this repository directory, install the required packages and run the complete packaged workflow:

```powershell
py -3 -m pip install pandas numpy scikit-learn joblib
py -3 michigan_boosted_tree.py
```

The script validates the bundled data, trains the model, evaluates the fixed test, prints historical examples, and writes:

| Output | Contents |
| --- | --- |
| [`outputs/results.json`](outputs/results.json) | Data profile, validation report, protocol, threshold, test metrics, limitations, and examples |
| [`outputs/test_2024_predictions.csv`](outputs/test_2024_predictions.csv) | Historical 2024 score, threshold, alert, county, and target for each eligible window |
| `outputs/michigan_boosted_tree_model.joblib` | Fitted model, feature list, threshold, and protocol |

The script does not download data or reconstruct the prepared dataset. It uses a checksum guard, so editing the CSV without updating and revalidating the audit will stop execution rather than silently train on changed data.

## Repository contents

- [`michigan_boosted_tree.py`](michigan_boosted_tree.py): end-to-end validation, training, threshold selection, evaluation, and output generation
- [`michigan_short_horizon_dataset.csv`](michigan_short_horizon_dataset.csv): prepared county/time modeling table
- [`michigan_counties.csv`](michigan_counties.csv): Michigan county reference data
- [`michigan_weather_stations.csv`](michigan_weather_stations.csv): station reference data
- [`source_audit.json`](source_audit.json): source links, annual counts/checksums, weather coverage, and assumptions
- [`outputs/`](outputs/): saved metrics, historical predictions, and fitted model artifact
- [`deliverables/`](deliverables/): project overview presentation and written report

## Limitations and interpretation

- **Not household prediction:** the unit and target are county-level high burden; the model cannot identify which homes lose service.
- **Not onset or restoration:** it does not predict outage onset, outage duration, remaining restoration time, or restoration time.
- **Incomplete observation and labels:** EAGLE-I coverage and zero reporting vary. Excluding incomplete targets avoids treating unknowns as zeros but may introduce selection bias.
- **Weather coverage and timing:** the packaged weather observations are seasonal; the assumed two-hour data delay is unverified. This is not a real-time weather-forecast integration.
- **Uncalibrated scores:** values such as 0.998 are model scores, not probabilities that can be communicated as calibrated real-world risk.
- **Previously inspected test period:** 2024 was evaluated earlier in the broader project. This packaged run reproduces a fixed historical test, but it is not a fresh independent evaluation.
- **Threshold drift:** the 2% false-positive-rate target was applied on 2023; the 2024 result was 2.01%, and the constraint is not guaranteed in future periods.
- **Partial first year:** 2014 contributes only from November 1.
- **No verified causal explanations:** feature associations cannot establish that weather or another input caused an outage.
- **No live service:** the script scores historical rows and does not ingest current outages or future forecast weather.

## What was achieved and what remains

The project moved from a broad power-cut prediction idea toward a precisely defined, reproducible short-horizon research task supported by the available records. It assembled a large Michigan county/time dataset, incorporated weather and recent-outage context, preserved unknowns during cleaning, separated training, threshold selection, and evaluation chronologically, reported both false alerts and missed events, and packaged the workflow and artifacts for rerunning.

The results are promising as a historical benchmark, especially in precision and ranking metrics, but they are not evidence of operational or household-level performance. The next meaningful research steps are to obtain independently verified, year-round outage and zero-reporting coverage; use weather forecasts that were actually available at each historical issue time; reserve a genuinely untouched future period; test year-by-year and county-by-county stability; calibrate scores on separate data; and compare against strong simple baselines. Causal explanations, restoration prediction, and live alerts should be treated as separate projects requiring appropriate labels and time-aligned data.

## References

1. EAGLE-I Electric Grid Monitor archive, Figshare: [https://doi.org/10.6084/m9.figshare.24237376](https://doi.org/10.6084/m9.figshare.24237376)
2. Iowa Environmental Mesonet, ASOS data download: [https://mesonet.agron.iastate.edu/request/download.phtml](https://mesonet.agron.iastate.edu/request/download.phtml)
3. scikit-learn, `HistGradientBoostingClassifier`: [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
