# Raw Data

This directory holds the official current-competition input data. Treat every file here as read-only source material.

## Contents

- `ch2025_data_items/`: official sensor parquet files used by the current DACON/ETRI challenge.
- `ch2026_metrics_train.csv`: 450 labeled rows for 10 subjects (`id01`–`id10`) across 2024-06 to 2024-11.
- `ch2026_submission_sample.csv`: 250-row submission schema for the same target columns.
- `ch2026_metrics_description*.md`: metric descriptions converted from the original PDF.
- `dataset-metadata.json`: local Kaggle/DACON packaging metadata copy.

## Rules

- Do not edit, overwrite, normalize, or append to files in this directory.
- Build features into `data/processed/` or memory, not back into `data/raw/`.
- If a raw file must be replaced, document the reason in the experiment log before training.
