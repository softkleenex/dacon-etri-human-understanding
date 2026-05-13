import os
import glob
import pandas as pd
import numpy as np
import argparse
import re

def simple_average_ensemble(submission_dir='submissions', output_name='submission_ensemble.csv', top_k=3):
    """
    Finds valid CSV files in the directory, parses their log-loss scores from the filename,
    and averages the predictions of the top-K models (lowest scores).
    """
    # 1. Find valid submission files
    csv_files = glob.glob(os.path.join(submission_dir, 'submission_*.csv'))
    valid_files = [f for f in csv_files if 'log' not in f and 'ensemble' not in f and 'latest' not in f]
    
    if len(valid_files) < 2:
        print("Need at least 2 submission files to ensemble.")
        return
        
    # 2. Parse scores and sort
    file_scores = []
    for f in valid_files:
        match = re.search(r'score([0-9]+\.[0-9]+)', f)
        if match:
            score = float(match.group(1))
            file_scores.append((f, score))
            
    if len(file_scores) == 0:
        print("Could not parse scores from filenames. Falling back to averaging all valid files.")
        selected_files = valid_files
    else:
        # Sort by score ascending (lower log-loss is better)
        file_scores.sort(key=lambda x: x[1])
        selected_files = [f[0] for f in file_scores[:top_k]]
        
    print(f"Selected top {len(selected_files)} submission files for ensemble:")
    for f in selected_files:
        print(f" - {f}")
        
    # 3. Read first file to establish structure
    ensemble_df = pd.read_csv(selected_files[0])
    target_cols = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
    
    # Initialize zero sum for targets
    for col in target_cols:
        ensemble_df[col] = 0.0
        
    # 4. Add probabilities
    for file in selected_files:
        df = pd.read_csv(file)
        for col in target_cols:
            ensemble_df[col] += df[col]
            
    # 5. Average probabilities
    for col in target_cols:
        ensemble_df[col] = ensemble_df[col] / len(selected_files)
        
    # 6. Clip probabilities to avoid infinite log-loss
    eps = 1e-15
    for col in target_cols:
        ensemble_df[col] = np.clip(ensemble_df[col], eps, 1 - eps)
        
    # 7. Save
    output_path = os.path.join(submission_dir, output_name)
    ensemble_df.to_csv(output_path, index=False)
    print(f"\\nEnsemble saved to {output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble multiple submission files.")
    parser.add_argument('--dir', type=str, default='submissions', help='Directory containing submission CSVs')
    parser.add_argument('--out', type=str, default='submission_ensemble_latest.csv', help='Output filename')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top models to ensemble')
    args = parser.parse_args()
    
    simple_average_ensemble(args.dir, args.out, args.top_k)