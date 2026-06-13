# Repository Guidelines

## Project Structure & Module Organization
This Python ML competition repo keeps reusable code in `src/`: `src/data/` for loading and dataset creation, `src/features/` for feature engineering, `src/models/` for model definitions and training helpers, `src/utils/` for validation/logging utilities, and `src/visualization/` for plotting. Automation and experiment entrypoints live in `scripts/`. Tests mirror core modules in `tests/test_*.py`. Config is in `conf/`; documentation and experiment notes are in `docs/`. Treat `data/`, `models/`, `checkpoints/`, `lightning_logs/`, and `submissions/` as generated or large-artifact areas unless a task explicitly requires them.

## Build, Test, and Development Commands
- `make install`: create `.venv`, install the package editable, and install the local DACON API wheel.
- `make format`: run Ruff autofix and formatter across the repo.
- `make lint`: run Ruff checks without modifying files.
- `make test`: run `pytest tests/ -v` with `PYTHONPATH=.`.
- `make jupyter`: start JupyterLab from the project virtualenv.
- `make run-local`: execute the heavy local SOTA training/stacking pipeline.
- `make build-kaggle-dataset` / `make kaggle-push-code`: package and push source/config/data to Kaggle.
- `make ensemble`: combine safe submission candidates via `scripts/ensemble.py`.
- `make audit`: inspect workspace health before trusted submissions.
- `make docs-audit`: verify documentation, command, and submission-record consistency.
- `make normalize-submissions`: refresh submission status/risk columns and blacklist risky files.

## Coding Style & Naming Conventions
Use Python 3.10+. Ruff is authoritative: 120-character line length, double quotes, spaces for indentation, import sorting enabled. Prefer type hints and concise docstrings for public functions. Use `snake_case` for modules, functions, variables, and script names; use `PascalCase` for classes. Keep feature logic modular under `src/features/`; avoid one-off abstractions.

## Testing Guidelines
Use pytest. Add or update `tests/test_*.py` when changing data loading, features, model utilities, or validation. Prefer small deterministic fixtures over full competition data. Before handoff, run `make format`, `make lint`, and `make test`; if a heavy ML path cannot be run locally, document the exact skipped verification and why.

## Commit & Pull Request Guidelines
History uses Conventional Commit-style prefixes: `feat:`, `fix:`, `docs:`, `chore:`. Keep messages imperative and specific, e.g. `feat: add target-specific rolling features`. PRs should describe the change, motivation, verification, and CV/leaderboard impact. Link related issues or experiment notes, and include screenshots only for notebook/report visual changes.

## Security & Configuration Tips
Do not commit `.env`, DACON/Kaggle tokens, private data, or large artifacts. Use environment variables such as `DACON_TOKEN` and `DACON_TEAMNAME` for submissions. Run `scripts/pre_flight_check.py` before Kaggle pushes or leaderboard-facing submissions.

## Root document roles
Keep root documentation minimal. `README.md` is the overview, `AGENTS.md` is the canonical agent/contributor operating guide, `CONTRIBUTING.md` is the human collaboration guide, and `CHANGELOG.md` is release history. See `docs/ROOT_DOCUMENTS.md`.
