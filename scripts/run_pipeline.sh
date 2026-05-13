#!/bin/bash
set -e

# Exit if Kaggle token or Dacon token isn't set
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$DACON_TOKEN" ]; then
    echo "Please ensure KAGGLE_USERNAME, KAGGLE_KEY, DACON_TOKEN, and DACON_TEAMNAME are set in your environment."
    exit 1
fi

echo "=========================================================="
echo "🚀 Starting E2E Pipeline (Foreground Wait & Submit)"
echo "=========================================================="

echo "[1/4] Running Pre-flight checks and packaging codebase..."
make kaggle-push-code

echo "[2/3] Triggering Kaggle GPU/CPU Kernel Execution and Waiting for completion..."
make kaggle-push-notebook

echo "[3/3] Pipeline Complete! Displaying latest submission logs:"
tail -n 5 submissions/submission_log.csv
echo "=========================================================="
