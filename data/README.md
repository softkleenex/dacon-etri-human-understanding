# Data Directory Manual

`data/` contains local competition inputs and regenerable intermediate datasets. Data files are ignored by Git; only directory markers and this guide are tracked.

## Directory roles

### `raw/` — current competition source data
Keep the official DACON/ETRI competition inputs here and treat them as read-only. In this workspace that includes:

- `ch2025_data_items/`: official sensor parquet files used by the current challenge.
- `ch2026_metrics_train.csv`: training labels/metrics.
- `ch2026_submission_sample.csv`: required submission schema.
- `ch2026_metrics_description*.md`: converted metric descriptions.

Do not overwrite or edit files in `raw/`. If a source file is replaced, document why before running experiments.

### `external/` — auxiliary same-institution / other-period data
Use this only for data not directly supplied as the current competition input, such as older ETRI lifelog summaries from 2018–2020. These files may be useful for priors or feature ideas, but they must be treated as auxiliary data with leakage risk.

Rules:
- Do not mix external rows into train/test without an explicit validation note.
- Keep bulky extracted external sensor dumps in `artifacts/external_data_archive/` unless a reviewed pipeline needs them.
- Record any external-data feature usage in `submissions/submission_log.csv` notes.

### `processed/` — regenerable caches
This directory should normally contain only `.gitkeep` in a clean workspace. Files such as `df_train_raw.parquet`, `df_test_raw.parquet`, preloaded caches, or embedding caches are generated artifacts. Rebuild them from `raw/` via the training/feature scripts instead of treating them as source data.

### `interim/`
Temporary handoff area for experiments. Move anything worth preserving to `reports/`, `submissions/`, or `artifacts/` with a manifest.

## Clean-room rule

Before starting a new competition session, `data/raw/` may remain populated, `data/external/` should contain only reviewed lightweight auxiliary sources, and `data/processed/` should be empty except `.gitkeep`.
