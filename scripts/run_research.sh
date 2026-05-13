#!/bin/bash
set -e

echo "=========================================================="
echo "🔬 Starting E2E Research/EDA Pipeline"
echo "=========================================================="

echo "[1/3] Running Pre-flight checks and packaging codebase..."
make kaggle-push-code

echo "[2/3] Triggering Kaggle Research Notebook (EDA) and waiting for output..."
# Pass AUTO_SUBMIT=false to bypass submission logic and validation
make kaggle-push-notebook KAGGLE_NOTEBOOK_DIR=notebooks/01-EDA KAGGLE_SLUG=softkleenex/dacon-etri-human-eda PULL_DIR=./reports AUTO_SUBMIT=false

echo "[3/3] Research Pipeline Complete! Output is saved in ./reports"
echo "=========================================================="
