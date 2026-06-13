# Processed Data

This directory is for regenerable feature caches and intermediate training tables.

In a clean workspace it should contain only `.gitkeep` and this README. Files such as these are generated artifacts, not source data:

- `df_train_raw.parquet`
- `df_test_raw.parquet`
- `df_train_preloaded_cache.parquet`
- `df_test_preloaded_cache.parquet`
- `hf_chronos_embeddings.csv`
- `df_train_processed.csv`

Recreate processed files from `data/raw/` through the training or feature scripts. If a processed file becomes important for reproducing a submission, archive it under `artifacts/` with a manifest rather than treating it as canonical input.
