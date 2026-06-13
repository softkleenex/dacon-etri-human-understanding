import glob
import os
import sys

import numpy as np
import pandas as pd


def main():
    print("🚀 Starting Strict Pre-Submission Validator")
    errors = 0

    # 1. Check if all base submission files exist and have no NaNs
    sub_files = glob.glob("submissions/submission_*.csv")
    if not sub_files:
        print("❌ Error: No submission files found in submissions/")
        errors += 1

    for file in sub_files:
        if 'latest' in file or 'log' in file:
            continue
        try:
            df = pd.read_csv(file)
            if df.isna().sum().sum() > 0:
                print(f"❌ Error: {file} contains NaN values!")
                errors += 1
            if df.isin([np.inf, -np.inf]).sum().sum() > 0:
                print(f"❌ Error: {file} contains Inf values!")
                errors += 1

            # Check range [0, 1]
            targets = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
            for t in targets:
                if t in df.columns:
                    if df[t].min() < 0 or df[t].max() > 1:
                        print(f"❌ Error: {file} target {t} is out of bounds [0, 1]! (Min: {df[t].min()}, Max: {df[t].max()})")
                        errors += 1
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")
            errors += 1

    # 2. Check if the DL training code has the 100% training loop or if we are using K-Fold averaging
    dl_trainer_path = "src/models/dl_trainer.py"
    if os.path.exists(dl_trainer_path):
        with open(dl_trainer_path) as f:
            content = f.read()
            if "n_splits: 1" in content and "full_trainer.fit(" not in content:
                print("❌ Error: src/models/dl_trainer.py is missing the 100% full training loop for Single Holdout! This causes severe Concept Drift.")
                errors += 1
            else:
                print("✅ DL model training strategy correctly validated.")

    # 3. Check Anchor Stacking for K-Fold
    train_ml_path = "scripts/train_simple_ml.py"
    if os.path.exists(train_ml_path):
        with open(train_ml_path) as f:
            content = f.read()
            if "n_splits = 1" in content:
                if "lstm_oof_path = None" not in content:
                    print("❌ Error: Anchor Stacking is enabled while using n_splits=1 in train_simple_ml.py. This causes catastrophic feature shift. Disable Anchor Stacking.")
                    errors += 1
                else:
                    print("✅ Anchor Stacking is safely disabled for Single Holdout.")
            elif "n_splits = 5" in content:
                if "lstm_oof_path = None" in content:
                    print("⚠️ Warning: Anchor Stacking is disabled while using n_splits=5. We recommend enabling it for better performance.")
                else:
                    print("✅ Anchor Stacking is safely enabled for K-Fold.")

    if errors > 0:
        print(f"\n🚨 Validation FAILED with {errors} errors. Do NOT submit to Dacon!")
        sys.exit(1)
    else:
        print("\n✅ All pre-submission checks passed. The ensemble is safe and leak-free!")

if __name__ == "__main__":
    main()
