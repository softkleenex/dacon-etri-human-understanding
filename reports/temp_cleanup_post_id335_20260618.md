# Temporary file/document cleanup post-ID335

## Actions

- Moved preserved Kaggle remote outputs from build/kaggle_outputs to ignored artifacts/kaggle_outputs.
- Deleted build/ including regenerated Kaggle source.zip and transient pull directories.
- Deleted local cache directories and non-venv .DS_Store files.
- Deleted unexecuted temporary promotion script scripts/promote_id178_s1_seed45502_candidate.py.
- Updated sequence-source and anchor-probe reports to point at artifacts/kaggle_outputs instead of build/kaggle_outputs.

## Not touched

- `submissions/*.csv`
- `submissions/submission_log.csv except later audits may normalize/update if explicitly run`
- `artifacts/base_predictions/*.csv`
- `notebooks/history/*.ipynb`
- `DACON submission state`

Preserved remote output directory: `artifacts/kaggle_outputs`
