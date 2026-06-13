# Experiment Operations

This repository is a DACON competition workspace. Optimize for reproducible leaderboard movement, not for accumulating artifacts.

## Before Any Submission

1. Run `make normalize-submissions` to refresh `status_clean`, `risk_reason`, and `safe_for_default_ensemble` in `submissions/submission_log.csv`.
2. Run `make audit`. A critical failure means the workspace is not ready for trusted leaderboard work.
3. Use `make ensemble` only after normalization. The default ensemble excludes blacklisted files and candidates with unsafe CV-LB gaps.

## Submission State Rules

Use `status_clean` as the operational state, not raw `is_submitted`.

- `scored`: real public LB score exists and is plausible.
- `generated_unsubmitted`: local artifact exists but has not been public-scored.
- `submitted_unscored` / `submitted_placeholder_score`: historical automation state; do not use for default ensembling.
- `cv_lb_gap_risk`: local CV and public LB diverged too much; treat as validation failure.
- `invalid_public_score`: penalty-scale or malformed score; never ensemble by default.

## Artifact Rules

Keep raw data, model binaries, OOF files, notebook logs, and training logs out of Git. Track summaries and scripts, not bulk artifacts. Generated experiment logs under `lightning_logs/`, `catboost_info/`, `checkpoints/`, and `scratch/` are ignored; if they were previously tracked, remove them with `git rm --cached` only, never by deleting local files.

## Commit Boundaries

Commit hygiene changes separately from modeling changes. A good cleanup commit includes scripts, docs, Makefile updates, `.gitignore`, and normalized metadata. It should not include new model binaries, raw data, or leaderboard submissions unless explicitly intended.
