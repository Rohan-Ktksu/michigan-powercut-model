"""Michigan county-level short-horizon high-outage-burden model, in one file.

Run from this folder with: py -3 michigan_boosted_tree.py
Requires pandas, numpy, scikit-learn and joblib.

The bundled CSV contains the merged EAGLE-I and historical-weather dataset.
This single script cleans and validates that table, profiles it, trains the
boosted tree, selects its threshold on 2023, evaluates the fixed 2024 test,
and saves predictions and model outputs. Raw national source archives are not
required to run this packaged workflow.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "michigan_short_horizon_dataset.csv"
AUDIT_PATH = HERE / "source_audit.json"
OUTPUT_DIR = HERE / "outputs"
HORIZON = 3
SEED = 42
FPR_CAP = 0.02
MIN_VALIDATION_ALERTS = 25

# These fields are identifiers, split labels or future outcomes, never inputs.
EXCLUDED = {
    "issue_utc", "target_peak_next_3h", "target_high_next_3h",
    "county_train_p90", "fips", "county", "split",
}
EXPECTED_DATASET_SHA256 = "d275f21b6489da2348ae1baeb493cef6690dbbf093e7898d76a3f66d4a9ce3c1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile(data):
    """Summarize coverage and missing inputs before model fitting."""
    years = data.issue_utc.dt.year
    labeled_rows = {}
    for year in range(2014, 2025):
        rows = data.loc[years.eq(year)]
        labeled_rows[str(year)] = int(rows[f"target_high_next_{HORIZON}h"].notna().sum())
    feature_columns = [column for column in data.columns if column not in EXCLUDED]
    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "counties": int(data.fips.nunique()),
        "date_range_utc": [data.issue_utc.min().isoformat(), data.issue_utc.max().isoformat()],
        "labeled_3h_rows_by_year": labeled_rows,
        "model_feature_count": len(feature_columns),
        "feature_missing_values": {
            column: int(data[column].isna().sum()) for column in feature_columns
            if data[column].isna().any()
        },
        "feature_columns": feature_columns,
    }


def clean_dataset(data):
    """Normalize identifiers and numeric inputs without inventing observations."""
    report = {"rows_before": int(len(data))}
    data = data.copy()
    data["issue_utc"] = pd.to_datetime(data["issue_utc"], utc=True, errors="coerce")
    for column in ("fips", "county", "split"):
        data[column] = data[column].astype("string").str.strip()
    invalid_keys = (data.issue_utc.isna() | data.fips.isna() | data.fips.eq("")
                    | data.county.isna() | data.county.eq("") | data.split.isna())
    report["invalid_key_rows_removed"] = int(invalid_keys.sum())
    data = data.loc[~invalid_keys].copy()
    data["fips"] = data.fips.str.zfill(5)

    categorical = {"issue_utc", "fips", "county", "split"}
    numeric_columns = [column for column in data.columns if column not in categorical]
    coercions = 0
    for column in numeric_columns:
        original = data[column]
        converted = pd.to_numeric(original, errors="coerce")
        coercions += int((original.notna() & converted.isna()).sum())
        data[column] = converted.replace([np.inf, -np.inf], np.nan)
    report["invalid_numeric_values_converted_to_missing"] = coercions
    report["rows_after"] = int(len(data))
    report["missing_values_preserved"] = int(data.isna().sum().sum())
    return data.reset_index(drop=True), report


def load_and_validate():
    """Load the bundled table and enforce target, time-split and leakage checks."""
    if not DATA_PATH.exists() or not AUDIT_PATH.exists():
        raise FileNotFoundError("Keep the dataset CSV and source_audit.json beside this script.")
    actual_hash = sha256(DATA_PATH)
    if actual_hash != EXPECTED_DATASET_SHA256:
        raise ValueError("The bundled dataset checksum differs from the verified source table.")

    data = pd.read_csv(DATA_PATH, dtype={"fips": str}, low_memory=False)
    required = EXCLUDED | {"temperature_c", "gust_kmh_max_6h", "precip_mm_max_6h",
                           "humidity_pct", "wind_kmh", "pressure_hpa", "last_observed_hour_outages",
                           "outage_mean_24h", "split", "target_high_next_3h",
                           "county_train_p90"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    data, cleaning = clean_dataset(data)
    if data.empty or data.issue_utc.isna().any():
        raise ValueError("Dataset is empty or has invalid issue timestamps.")
    if data.duplicated(["issue_utc", "fips"]).any():
        raise ValueError("Duplicate county/timestamp rows must be reviewed, not silently removed.")
    if data.fips.isna().any() or data.county.isna().any():
        raise ValueError("County identifiers and names must be present.")
    if set(data.split.dropna().unique()) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test splits.")

    expected_years = {"validation": {2023}, "test": {2024}}
    for split, years in expected_years.items():
        block = data.loc[data.split.eq(split)]
        if block.empty or set(block.issue_utc.dt.year.unique()) != years:
            raise ValueError(f"Split {split} must contain only years {sorted(years)}.")
    train_years = set(data.loc[data.split.eq("train"), "issue_utc"].dt.year.unique())
    if train_years != set(range(2014, 2023)):
        raise ValueError("Training split must contain all available years 2014-2022.")

    # Missing target windows stay unknown; do not turn them into negative labels.
    target = f"target_high_next_{HORIZON}h"
    peak = f"target_peak_next_{HORIZON}h"
    labeled = data[target].notna()
    if data.loc[labeled, [peak, "county_train_p90"]].isna().any().any():
        raise ValueError("A labeled target has a missing peak or county cutoff.")
    expected_label = data.loc[labeled, peak].ge(data.loc[labeled, "county_train_p90"]).astype(int)
    if not data.loc[labeled, target].astype(int).equals(expected_label.astype(int)):
        raise ValueError("High-burden labels disagree with peak counts and training county cutoffs.")
    if data.loc[~labeled, peak].notna().any():
        raise ValueError("An incomplete target row unexpectedly has a future peak value.")
    if not data.loc[labeled, target].isin([0, 1]).all():
        raise ValueError("Classification target must contain only 0/1 labels.")

    features = [column for column in data.columns if column not in EXCLUDED]
    numeric = data[features].select_dtypes(include="number")
    if len(numeric.columns) != len(features):
        raise ValueError("All model inputs must be numeric after cleaning.")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Infinite numeric feature values found.")
    if not features or len(features) < 10:
        raise ValueError("Too few usable input features.")

    report = {
        "status": "passed",
        "dataset_sha256": actual_hash,
        "duplicate_county_timestamps": 0,
        "unknown_targets_excluded": int((~labeled).sum()),
        "missing_inputs": int(data[features].isna().sum().sum()),
        "missing_input_policy": "Kept as missing; HistGradientBoosting handles missing values natively.",
        "cleaning": cleaning,
        "target_validation": "Only complete next-3-hour windows were emitted by the builder; labels checked against county p90 from training years 2014-2022.",
        "temporal_validation": "Train 2014-2022, threshold validation 2023, fixed test 2024.",
    }
    return data, features, report


def choose_threshold(labels, scores):
    """Choose 2023 threshold: highest recall under FPR cap and support floor."""
    y = np.asarray(labels, dtype=bool)
    score = np.asarray(scores, dtype=float)
    negatives = int((~y).sum())
    candidates = np.r_[np.unique(score), np.nextafter(score.max(), np.inf)]
    feasible = []
    for threshold in candidates:
        alert = score >= threshold
        tp = int((alert & y).sum())
        fp = int((alert & ~y).sum())
        count = tp + fp
        false_positive_rate = fp / negatives if negatives else 1.0
        if count >= MIN_VALIDATION_ALERTS and false_positive_rate <= FPR_CAP:
            precision = tp / count
            recall = tp / max(int(y.sum()), 1)
            feasible.append((recall, precision, float(threshold), fp, tp))
    if not feasible:
        return {"feasible": False, "threshold": None, "validation_alerts": 0,
                "reason": "No threshold satisfies validation cap and alert support."}
    recall, precision, threshold, fp, tp = max(feasible, key=lambda item: (item[0], item[1], item[2]))
    return {"feasible": True, "threshold": threshold, "validation_alerts": fp + tp,
            "validation_false_positives": fp, "validation_true_positives": tp,
            "validation_false_positive_rate": fp / negatives,
            "validation_precision": precision, "validation_recall": recall,
            "false_positive_rate_cap": FPR_CAP}


def metrics(labels, scores, threshold):
    y = np.asarray(labels, dtype=bool)
    score = np.asarray(scores, dtype=float)
    alert = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, alert, labels=[False, True]).ravel()
    return {
        "rows": int(len(y)), "positive_rate": float(y.mean()), "alerts": int(alert.sum()),
        "accuracy": float(accuracy_score(y, alert)),
        "balanced_accuracy": float(balanced_accuracy_score(y, alert)),
        "precision": float(precision_score(y, alert, zero_division=0)),
        "recall": float(recall_score(y, alert, zero_division=0)),
        "f1": float(f1_score(y, alert, zero_division=0)),
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "false_alerts": int(fp), "missed_events": int(fn),
        "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, feature_columns, validation = load_and_validate()
    profile_data = profile(data)
    target_column = f"target_high_next_{HORIZON}h"
    labeled = data.loc[data[target_column].notna()].copy()
    labeled["target"] = labeled[target_column].astype(int)
    train = labeled.loc[labeled.split.eq("train")]
    val = labeled.loc[labeled.split.eq("validation")]
    test = labeled.loc[labeled.split.eq("test")]
    if min(len(train), len(val), len(test)) < 1000:
        raise ValueError("Each split must provide at least 1,000 complete labeled rows.")

    y_train = train.target.to_numpy(dtype=int)
    positive_rate = max(float(y_train.mean()), 1e-4)
    sample_weight = np.where(y_train == 1, 0.5 / positive_rate,
                             0.5 / (1 - positive_rate))
    model = HistGradientBoostingClassifier(
        max_iter=180, max_leaf_nodes=15, learning_rate=0.05,
        min_samples_leaf=40, l2_regularization=2,
        early_stopping=False, random_state=SEED)

    # Train on all available 2014-2022 rows; class weights address rare positives.
    model.fit(train[feature_columns], y_train, sample_weight=sample_weight)
    validation_scores = model.predict_proba(val[feature_columns])[:, 1]
    threshold_result = choose_threshold(val.target, validation_scores)
    if threshold_result["feasible"]:
        threshold = threshold_result["threshold"]
        threshold_status = "chosen on 2023 only"
    else:
        threshold = float("inf")
        threshold_status = "abstain: validation cap unsupported"

    # 2024 is scored only after the model and threshold are fixed.
    test_scores = model.predict_proba(test[feature_columns])[:, 1]
    test_metrics = metrics(test.target, test_scores, threshold)
    test_metrics["threshold_status"] = threshold_status
    baseline = metrics(test.target, np.zeros(len(test)), 0.5)

    # Select examples by model score only; actual outcomes are shown for context.
    scored_test = test[["issue_utc", "fips", "county", "county_train_p90", "target"]].copy()
    scored_test["model_score_uncalibrated"] = test_scores
    scored_test["alert"] = test_scores >= threshold
    alert_examples = scored_test.loc[scored_test.alert].nlargest(3, "model_score_uncalibrated")
    non_alert_examples = scored_test.loc[~scored_test.alert].nlargest(3, "model_score_uncalibrated")
    selected_examples = pd.concat([alert_examples, non_alert_examples]).sort_values(
        "model_score_uncalibrated", ascending=False)
    historical_examples = [
        {
            "issue_time_utc": row.issue_utc.isoformat(),
            "county": row.county,
            "horizon_hours": HORIZON,
            "alert": bool(row.alert),
            "model_score_uncalibrated": float(row.model_score_uncalibrated),
            "threshold_from_2023": float(threshold),
            "historical_actual_high_burden": bool(row.target),
            "county_training_p90_customers_out": float(row.county_train_p90),
            "example_selection": "Highest model scores within predicted-alert or predicted-no-alert group; selected without using actual label.",
        }
        for row in selected_examples.itertuples(index=False)
    ]

    predictions = test[["issue_utc", "fips", "county", "county_train_p90", "target"]].copy()
    predictions["model_score_uncalibrated"] = test_scores
    predictions["alert_threshold_from_2023"] = threshold
    predictions["alert"] = test_scores >= threshold
    predictions.to_csv(OUTPUT_DIR / "test_2024_predictions.csv", index=False)
    report = {
        "project": "Michigan 3-hour county high-burden alert; selected boosted-tree model",
        "profiling": profile_data,
        "data_validation": validation,
        "protocol": {
            "training": "2014-2022; 2014 is partial (November-December)", "threshold_selection": "2023 only",
            "test": "2024 fixed historical holdout", "horizon_hours": HORIZON,
            "label": "At least one complete hourly county count >= that county's training-period (2014-2022) p90 in the next 3 hours",
            "threshold_rule": "Maximize recall subject to validation FPR <= 2% and >=25 alerts",
            "warning_inputs_included": False,
            "weather_inputs": "Historical ASOS station observations and reported storm codes; NWS warning inputs omitted for consistent all-year coverage.",
            "probabilities_calibrated": False,
        },
        "rows": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "threshold_selection": threshold_result,
        "test_model": test_metrics,
        "test_always_no_alert_baseline": baseline,
        "historical_prediction_examples": historical_examples,
        "limitations": [
            "This is a county-level high-burden alert, not outage onset or household power-cut probability.",
            "EAGLE-I coverage is incomplete and uneven; unknown target windows are excluded, not called negatives.",
            "2025 is excluded because audited Michigan readings have no explicit zero customer counts.",
            "Weather is historical station observation data with assumed, unverified two-hour availability; not archived forecast weather.",
            "The 2024 test was already evaluated in the research project; this packaged run reproduces it and is not a fresh independent test.",
            "The 2% false-positive-rate cap was selected on 2023 and is not guaranteed on 2024.",
            "2014 source coverage starts November 1; 2014 is a partial training year.",
            "Historical NWS warning features were not used because year-complete validated warning data was unavailable.",
        ],
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump({"model": model, "features": feature_columns,
                 "threshold": threshold, "protocol": report["protocol"]},
                OUTPUT_DIR / "michigan_boosted_tree_model.joblib")

    print("1. DATA PROFILING")
    print(json.dumps({key: value for key, value in profile_data.items()
                      if key != "feature_columns"}, indent=2))
    print("\n2. DATA CLEANING AND VALIDATION")
    print(json.dumps(validation, indent=2))
    print("\n3. MODEL: HISTOGRAM GRADIENT BOOSTING + WEATHER AND OUTAGE HISTORY")
    print(f"Features: {len(feature_columns)}; train/validation/test rows: {len(train)}/{len(val)}/{len(test)}")
    print("\n4. 2024 HELD-OUT TEST METRICS")
    print(json.dumps(test_metrics, indent=2))
    print("\n5. SEVERAL HISTORICAL EXAMPLE PREDICTIONS")
    print(json.dumps(historical_examples, indent=2))
    print(f"\nSaved outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
