import os
import time
import subprocess
import glob
import argparse
from pathlib import Path
from dacon_submit_api import dacon_submit_api

def wait_and_manage(kernel_slug, output_dir, auto_submit=False):
    print(f"Watching kernel: {kernel_slug}")
    
    while True:
        try:
            result = subprocess.run(['kaggle', 'kernels', 'status', kernel_slug], capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            print(f"Status: {output}")
            if 'complete' in output.lower():
                print("Kernel finished successfully!")
                break
            elif 'error' in output.lower() or 'cancel' in output.lower():
                print("Kernel failed or was cancelled.")
                return
        except subprocess.CalledProcessError as e:
            print(f"Kaggle API error: {e}. Retrying in 60s...")
            
        time.sleep(60)

    print(f"Pulling output to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Retry pulling output up to 3 times
    for attempt in range(3):
        try:
            subprocess.run(['kaggle', 'kernels', 'output', kernel_slug, '-p', output_dir, '--force'], check=True)
            break
        except subprocess.CalledProcessError as e:
            print(f"Failed to pull output (Attempt {attempt+1}/3): {e}")
            if attempt == 2:
                print("Max retries reached for pulling output.")
                return
            time.sleep(30)
            
    if not auto_submit:
        print(f"Outputs pulled to {output_dir}. Skipping auto-submission (Research mode).")
        return
    
    # Auto-submit logic
    # Find the newly generated CSV. Look for any submission_*.csv recursively
    csv_files = glob.glob(f'{output_dir}/**/submission_*.csv', recursive=True)
    
    # Exclude logs and existing ensembles
    valid_csvs = [f for f in csv_files if 'log' not in f and 'latest' not in f and 'ensemble' not in f]
    
    if not valid_csvs:
        print("Could not find any submission files for Dacon.")
        return
        
    # Pick the newest one by modification time
    latest_csv = max(valid_csvs, key=os.path.getmtime)
    filename_to_submit = os.path.basename(latest_csv)
    print(f"Found latest submission file: {latest_csv}")
    
    # ---------------------------------------------
    # 4. Prevent Duplicate Submission Check
    # ---------------------------------------------
    import pandas as pd
    csv_path = 'submissions/submission_log.csv'
    
    if pd.io.common.file_exists(csv_path):
        try:
            log_df = pd.read_csv(csv_path)
            if filename_to_submit in log_df['submission_file'].values:
                print(f"⚠️ Warning: File {filename_to_submit} has already been submitted according to the log. Aborting to save daily limits.")
                return
        except Exception as e:
            print("Error reading log for duplicate check:", e)

    # 5. Submit to Dacon
    dacon_token = os.environ.get('DACON_TOKEN')
    team_name = os.environ.get('DACON_TEAMNAME')
    memo = f"Automated submission from {kernel_slug}"
    competition_id = '236690'
    
    print(f"Submitting {filename_to_submit} to Dacon API...")
    for attempt in range(3):
        try:
            res = dacon_submit_api.post_submission_file(
                file_path=latest_csv,
                token=dacon_token,
                cpt_id=competition_id,
                team_name=team_name,
                memo=memo
            )
            print("Dacon API Result:", res)
            break
        except Exception as e:
            print(f"Error submitting (Attempt {attempt+1}/3): {e}")
            if attempt == 2:
                print("Max retries reached for API submission.")
                return
            time.sleep(30)
        
    # Log to submission_log.csv
    try:
        csv_path = 'submissions/submission_log.csv'
        if not os.path.exists(csv_path):
            df = pd.DataFrame(columns=['id', 'date', 'submission_file', 'model_description', 'local_cv_score', 'public_lb_score', 'is_submitted', 'notes'])
        else:
            df = pd.read_csv(csv_path)

        new_record = {
            'id': len(df) + 1,
            'date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            'submission_file': filename_to_submit,
            'model_description': memo,
            'local_cv_score': 0.0, # Placeholder
            'public_lb_score': None,
            'is_submitted': True,
            'notes': 'Submitted automatically via kaggle_manager.py'
        }
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print("Logged to submission_log.csv")
    except Exception as e:
        print("Error logging submission:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wait for a Kaggle kernel and manage outputs.")
    parser.add_argument("--slug", type=str, required=True, help="Kaggle kernel slug (e.g., username/kernel-name)")
    parser.add_argument("--out_dir", type=str, default="submissions", help="Directory to pull outputs into")
    parser.add_argument("--submit", action="store_true", help="Attempt automatic Dacon submission if CSV is found")
    args = parser.parse_args()
    
    wait_and_manage(args.slug, args.out_dir, auto_submit=args.submit)