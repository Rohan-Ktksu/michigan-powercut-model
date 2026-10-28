# Browser dataset — Project by Rohan Sai

`browser_dataset.csv.gz` contains 29,000 existing historical records sampled with seed 42 within the original periods: 20,000 fit rows (2014–2022), 4,000 validation rows (2023), and 5,000 test rows (2024). It is a demonstration subset of the audited prepared Michigan table, not new observations or synthetic labels.

`../site-data.json` records the original and compressed-subset SHA-256 values. The browser verifies the subset checksum before fitting. Regenerate with `python tools/build_website.py --source <original CSV>` from the repository root. Dataset-provider attribution is retained in the main README.
