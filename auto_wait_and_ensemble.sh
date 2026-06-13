#!/bin/bash
# Wait for XGB and LGBM to finish
echo "Waiting for XGBoost and LightGBM tuning to finish..."
.venv/bin/python -u scripts/kaggle_manager.py --slug softkleenex/dacon-etri-xgb-tuning-v8 --out_dir submissions/xgb
.venv/bin/python -u scripts/kaggle_manager.py --slug softkleenex/dacon-etri-lgbm-tuning-v8 --out_dir submissions/lgbm

# Once those two are done, push CatBoost (Manually pushed to v9 to bypass v8 api issues)
echo "Waiting for CatBoost tuning kernel (v9) to finish..."
.venv/bin/python -u scripts/kaggle_manager.py --slug softkleenex/dacon-etri-cat-tuning-v9 --out_dir submissions/cat

# Move all CSVs
echo "Consolidating all submissions and OOF files..."
find submissions/xgb submissions/lgbm submissions/cat -name "*.csv" -exec cp {} submissions/ \; || true

# Run ensemble optimization
echo "Running ensemble optimization..."
.venv/bin/python scripts/optimize_ensemble.py

echo "Done!"
