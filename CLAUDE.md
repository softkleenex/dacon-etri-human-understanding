# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Workspace for the **5th ETRI Human Understanding AI Paper Challenge** (DACON competition ID `236690`).
Predict 7 binary targets (`Q1`–`Q3` subjective sleep quality / fatigue / stress, `S1`–`S4` objective sleep
metrics) from multimodal smartphone (`m` prefix) and smartwatch (`w` prefix) lifelog data.
Metric: **mean log loss across the 7 targets** (lower is better). 450 labeled person-days from 10 participants.

The paper was accepted to ICTC 2026 (EDAS paper `1571325954`); the confirmed public anchor is submission
**ID342** (local `0.534000`, public `0.5960832519`). Competition submission phase is over — the live concern
is reproducibility of ID342, not new leaderboard gains.

## Commands

```bash
make install          # create .venv, pip install -e ., install local dacon_submit_api wheel from ~/sysp/
make format           # ruff check --fix + ruff format
make lint             # ruff check
make test             # PYTHONPATH=. .venv/bin/pytest tests/ -v
make audit            # full workspace/docs/submissions/artifact/structure integrity sweep
make ensemble         # scripts/ensemble.py — blend top-K safe candidates
make run-local        # scripts/run_full_sota_pipeline.py (heavy local training)
```

Single test: `PYTHONPATH=. .venv/bin/pytest tests/test_features.py::test_name -v`

Always use `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff` — the Makefile never relies on an
activated venv, and neither should new code or docs.

### Lint paths are an explicit allowlist

`make lint`/`make format` only touch `$(LINT_PATHS)` in the Makefile — `src`, `tests`, and a hand-maintained
list of ~150 individual `scripts/*.py` files. **A new script under `scripts/` is not linted until you append
it**, via `LINT_PATHS += scripts/your_script.py` near the bottom of the variable block. Most of `scripts/`
(380+ files) is historical and deliberately unlinted.

## Guarded operations (fail-closed)

Anything that reaches an external service is blocked behind an env-var gate. Do not bypass these.

- **Kaggle remote** (`kaggle-push-*`, `kaggle-pull-*`): requires `ALLOW_KAGGLE_REMOTE=true`.
- **DACON submit** (`make dacon-submit`): requires *all* of `ALLOW_DACON_SUBMIT=true`,
  `DACON_APPROVED_SUB_PATH` exactly equal to `SUB_PATH`, and `DACON_APPROVED_SUB_MEMO` exactly equal to
  `SUB_MEMO`, then passes `scripts/audit_dacon_submit_approval_contract.py`. The triple-match exists so an
  agent cannot submit by reusing a default. Never run it without explicit per-submission user approval
  naming the exact file and memo.
- `scripts/runner_bot.py` and `scripts/orchestrate_sota.py` are auto-loops — do not run them.

Credentials come from the shell environment (`DACON_TOKEN`, `DACON_TEAMNAME`), never from source.
`.env`/`.env.example` hold only public-data API keys (KMA, AirKorea).

## Architecture

### `src/` — reusable library (the only linted, tested modeling code)

- `src/data/` — `data_loader.py` (raw CSV/parquet intake), `make_dataset.py`, `dl_data_module.py`
  (`MultiTaskDataset`, `MultiTaskSequenceDataset`, `DaconDataModule` Lightning module).
- `src/features/` — `build_features.py` is the main entry (`engineer_features`), plus cross-sensor
  interactions, per-participant normalization, rolling/lag windows, and rolling cosinor fits. Domain-specific
  feature families live in their own modules (`morning_recovery.py`, `shutdown.py`, `q_context.py`,
  `demographics.py`).
- `src/models/` — `dl_models.py` (`MultiTaskLSTM` with `TemporalAttention`, `MultiTaskTransformer`,
  `MultiTaskMLP`, `FocalLoss`), `trainer.py` (CatBoost/GBDT + Optuna), `model_builder.py`,
  `kaggle_sota_framework.py` (the Kaggle-side stacking runner).
- `src/utils/validation.py` — **read this before touching any modeling code.** Holds `clip_probabilities`
  (mandatory before writing any submission; unclipped 0/1 gives infinite log loss), `validate_submission`,
  and the two CV protocols: `make_subject_timeseries_folds` and `make_subject_hole_folds`.

### The central problem: CV–LB gap

Local CV improvements routinely fail to transfer. ID287 scored local `0.378934` → public `0.6698787318`.
Local scores below ~`0.50` are evidence of overfitting, not progress. Treat these as high risk regardless of
local score: public-anchor blends, full-OOF residual recombinations, grids selected on full (non-nested) OOF,
and any candidate with `cv_lb_gap >= 0.03`.

### The artifact contract

Every experiment must keep a strict 1:1:1:1 mapping, enforced by `scripts/audit_experiment_artifacts.py`:

```
notebooks/history/submission_<id>_<slug>_cv<score>.ipynb
  -> submissions/submission_id_<id>_<slug>_cv<score>.csv
  -> submissions/submission_log.csv row id=<id>
  -> reports/experiment_<id>_<slug>_cv<score>.json
```

`submissions/submission_log.csv` is the source of truth for experiment history. `make normalize-submissions`
derives the operational columns (`status_clean`, `risk_reason`, `safe_for_default_ensemble`, `cv_lb_gap`,
`artifact_role`) — edit the raw columns, then regenerate; never hand-edit derived columns.

### `scripts/` — the "report-only gate" pattern

Most of the Makefile's hundreds of `post-id34X-*` targets follow one shape: run a
`scripts/diagnose_*.py` / `scripts/audit_*.py` that reads existing artifacts, decides something, and writes a
paired `reports/<name>_<date>.json` + `.md`. These **never** train models, write predictions, touch Kaggle, or
submit — many assert that explicitly in their output (`no_dacon_submission_made`, `no_predictions_created`).
Targets chain via prerequisites to express analysis order. When adding analysis, follow this pattern rather
than inventing a new one: a script, a `reports/` JSON+MD pair, a Makefile target, a `tests/test_*.py`, and a
`LINT_PATHS +=` line.

`scripts/run_strict_*_experiment.py` are the ones that actually produce candidates under the strict
Subject-Hole protocol; `scripts/run_public_*_probe_experiment.py` are public-response probes (higher risk by
construction).

### Data layout

- `data/raw/` — competition data (`ch2026_metrics_train.csv`, `ch2026_submission_sample.csv`,
  `ch2025_data_items/`). Git-ignored.
- `data/processed/` — `df_train_raw.parquet`, `df_test_raw.parquet` plus registered feature parquets.
- `data/external/` — historical ETRI lifelog CSVs and approved public weather/air-quality caches. External
  data use is constrained by competition rules; check `docs/competition/4_rules.md` before adding any.
- `artifacts/base_predictions/` — `base_oof_id_<id>_*.csv` / `base_test_id_<id>_*.csv`, the OOF/test matrices
  later experiments blend. Downstream candidates depend on these, so treat them as immutable once registered.

### Kaggle workflow (base-plus-patch)

Heavy runs go to Kaggle. A large monolithic `source.zip` upload is a known failure mode — do not retry it.
Instead `softkleenex/dacon-etri-human-base` supplies the large raw/processed files and a small patch dataset
(`dacon-etri-upstream-patch-post-id341`, etc.) overlays current `scripts/`, `reports/`, `submissions/`, and
`artifacts/base_predictions/`. The runner notebook stages base first, then the patch. Kaggle consumes
`dataset-metadata.json` as dataset metadata and does not expose it as an input file, so the runner synthesizes
runtime metadata after overlay. See `docs/WORKFLOW_REVIEW.md` for the full incident history.

## Required reading before modeling work

`docs/WORKFLOW_REVIEW.md` (operating rules and current verdict), `docs/EXPERIMENT_OPERATIONS.md`,
`docs/PUBLIC_055_PLAN.md`. `AGENTS.md` is the canonical contributor/agent guide; `docs/ROOT_DOCUMENTS.md`
defines what belongs in each root document. Keep root docs minimal — new notes go in `docs/`.

## Conventions

- Python 3.10+, ruff-authoritative: 120-char lines, double quotes, sorted imports. `snake_case` modules and
  functions, `PascalCase` classes. Type hints and short docstrings on public functions.
- Conventional Commit prefixes (`feat:`, `fix:`, `docs:`, `chore:`).
- Keep feature logic modular under `src/features/`; avoid one-off abstractions.
- Prefer small deterministic fixtures in tests over full competition data.
- If a heavy ML path can't be run locally, say so explicitly and name the skipped verification.
