# Michigan Outage Lab — Project by Rohan Sai

Account: [Rohan-Ktksu](https://github.com/Rohan-Ktksu). Repository: `michigan-powercut-model`.

The GitHub Pages website publishes `docs/` from `guided-model-website`. It uses the original research model and target definition with a browser-sized demonstration, rather than substituting a different classifier or generating synthetic labels.

## Website flow

1. **Dataset:** inspect real sample windows, split sizes and source provenance.
2. **Train Model:** load Pyodide/scikit-learn in a Web Worker, verify the compressed subset checksum, train the boosted tree through 180 iterations with progress, and select the alert threshold on 2023 validation rows.
3. **Test Model:** evaluate the fixed model and threshold on the 2024 subset. Inspect accuracy, balanced accuracy, precision/recall/F1, ROC-AUC, average precision, ROC/PR curves, confusion matrix and a no-alert baseline. Download the result data.
4. **Predict:** classify selected historical county windows or an uploaded CSV through the fitted model. Download predictions and inspect the exact inputs. Uploads are processed within the tab and never enter training.

The clean light theme applies to both pages. The pipeline diagram and scrollable Python source viewer appear only on **Model & Features** (model.html#pipeline).

**Model & Features** explains all 67 retained numeric predictors, exclusions, sampling, parameters and limitations. It initially displays the saved full-research results, checked against `outputs/test_2024_predictions.csv`. Completed browser tests are stored in sessionStorage under `michigan-evaluation`, and that page then labels them as browser sample results. Uploaded prediction records are not placed in that cross-page store.

## Full research run versus browser sample

| Partition | Original research | Browser demonstration |
| --- | ---: | ---: |
| Fit, 2014–2022 | 272,611 | 20,000 |
| Threshold validation, 2023 | 45,004 | 4,000 |
| Historical test, 2024 | 45,804 | 5,000 |

Sampling is deterministic (seed 42) within each original partition. Existing precomputed targets and training-period county cutoffs are retained; no new labels are generated. All 67 predictors are used with the same statistical model settings. Warm-start batches expose iteration progress; regression tests check agreement with a single 180-iteration fit. Browser and desktop package versions may differ, so browser scores and thresholds may differ.

The subset is distributed as `docs/data/browser_dataset.csv.gz`, roughly 3.3 MB. It is not an LFS pointer. The original audited CSV is 168,915,944 bytes and remains stored via Git LFS. The subset manifest records both checksums. The first run also downloads Python/scientific packages; model state resets when reloading or leaving the workspace. Completed evaluation summaries remain available on Model & Features in the same tab.

## Reproduce

Use a Python environment with the packages in `requirements-web.txt`. Obtain the original CSV with Git LFS or download it from the repository's LFS media URL, preserving its audited hash. Never replace the original with a pointer file or silently change the checksum.

```powershell
python -m pip install -r requirements-web.txt
python tools/build_website.py --source path/to/michigan_short_horizon_dataset.csv
python -m pytest -q
python -m http.server 8503 --bind 127.0.0.1
```

Open `http://127.0.0.1:8503/docs/`. The builder verifies the full original dataset, refreshes the deterministic subset, copies shared Python into `docs/runtime/`, and verifies published metrics against the existing full-test predictions before exporting display curves.

The full saved model results remain historical county-burden research, not live forecasts, household interruption probabilities, causal weather attribution or restoration predictions. The already-inspected 2024 period is not a new independent test. Original data-provider attribution remains in the research README.
