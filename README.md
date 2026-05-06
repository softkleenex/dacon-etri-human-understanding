# Dacon Base Template

This is a generic baseline repository for Dacon (and Kaggle) competitions. It provides a standardized directory structure, linting, formatting, and essential data science libraries to quickly start any new competition.

## Directory Structure

- `data/`: Store datasets here. `raw/` for unmodified original data, `processed/` for cleaned/engineered data.
- `docs/`: Project documentation, strategy notes, and planning.
- `models/`: Saved models and weights.
- `notebooks/`: Jupyter notebooks for EDA and experimentation.
- `scripts/`: Executable scripts for training, inference, and submission orchestration.
- `src/`: Reusable source code (data loading, feature engineering, modeling, utilities).
- `submissions/`: Output CSV files for competition submission.
- `tests/`: Unit tests.

## Setup

1. **Install Dependencies:**
   ```bash
   make install
   ```

2. **Environment Variables:**
   Copy `.env.example` to `.env` and fill in necessary keys (like Kaggle token, etc.).
   ```bash
   cp .env.example .env
   ```
   *Note: `DACON_TOKEN`, `DACON_TEAMNAME`, `WANDB_API_KEY`, and `HF_TOKEN` are configured globally in `~/.zshrc` and do not need to be set in the `.env` file per project.*

## Development

- **Format Code:** `make format`
- **Lint Code:** `make lint`
- **Run Tests:** `make test`
