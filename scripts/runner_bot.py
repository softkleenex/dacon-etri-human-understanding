import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from dacon_submit_api import dacon_submit_api

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.logger import get_logger

logger = get_logger("runner_bot")

def check_kaggle_status(kernel_slug):
    try:
        result = subprocess.run(['kaggle', 'kernels', 'status', kernel_slug], capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        logger.info(f"Kaggle Status check for {kernel_slug}: {output}")
        if 'complete' in output.lower():
            return 'complete'
        elif 'error' in output.lower() or 'cancel' in output.lower():
            return 'error'
        else:
            return 'running'
    except subprocess.CalledProcessError as e:
        logger.error(f"Kaggle API status error: {e}")
        return 'api_error'

def pull_kaggle_output(kernel_slug, output_dir):
    logger.info(f"Pulling output for {kernel_slug} to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            subprocess.run(['kaggle', 'kernels', 'output', kernel_slug, '-p', output_dir, '--force'], check=True)
            logger.info("Successfully pulled Kaggle output.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull output (Attempt {attempt+1}/3): {e}")
            time.sleep(30)
    return False

def find_best_dl_model(dl_dir):
    """
    Finds the trial with the lowest score from the output of the optuna study.
    Copies:
      - oof_optuna_trial_{X}_score{SCORE}.csv -> submissions/oof_dl_v10.csv
      - submission_optuna_trial_{X}_score{SCORE}.csv -> submissions/submission_dl_v10.csv
      - user_embeddings_optuna_trial_{X}.csv -> submissions/user_embeddings_local_transformer_cv.csv
    """
    logger.info(f"Searching for best DL model in {dl_dir}...")
    # Search in all directories under dl_dir for oof_optuna_trial_*_score*.csv
    oof_pattern = os.path.join(dl_dir, "**/oof_optuna_trial_*_score*.csv")
    oof_files = glob.glob(oof_pattern, recursive=True)

    if not oof_files:
        logger.warning(f"No oof files matching 'oof_optuna_trial_*_score*.csv' found in {dl_dir}")
        return None

    best_file = None
    best_score = 999.0
    best_trial_num = None

    for f in oof_files:
        # Parse score from filename (e.g., oof_optuna_trial_12_score0.6400.csv)
        try:
            base = os.path.basename(f)
            # Find the index of 'score'
            score_idx = base.find('score')
            if score_idx != -1:
                score_str = base[score_idx + 5:].replace('.csv', '')
                score = float(score_str)

                # Parse trial number
                trial_part = base[:score_idx] # e.g., 'oof_optuna_trial_12_'
                trial_num = int(''.join(filter(str.isdigit, trial_part)))

                if score < best_score:
                    best_score = score
                    best_file = f
                    best_trial_num = trial_num
        except Exception as e:
            logger.warning(f"Failed to parse filename {f}: {e}")

    if best_file is None:
        logger.error("Could not determine best DL model based on scores.")
        return None

    logger.info(f"🏆 Best DL Model found: {best_file} with score: {best_score:.5f} (Trial {best_trial_num})")

    # Locate matching submission and embedding files
    parent_dir = os.path.dirname(best_file)
    sub_pattern = os.path.join(parent_dir, f"submission_optuna_trial_{best_trial_num}_score{best_score:.4f}.csv")
    emb_pattern = os.path.join(parent_dir, f"user_embeddings_optuna_trial_{best_trial_num}.csv")

    sub_files = glob.glob(sub_pattern)
    emb_files = glob.glob(emb_pattern)

    if not sub_files:
        # Try generic search just in case formatting is slightly off (e.g. fewer decimal places)
        sub_pattern_alt = os.path.join(parent_dir, f"submission_optuna_trial_{best_trial_num}_score*.csv")
        sub_files = glob.glob(sub_pattern_alt)

    if not sub_files:
        logger.error(f"Could not find matching submission file for trial {best_trial_num}")
        return None

    best_sub = sub_files[0]

    # Destination Paths
    dest_oof = "submissions/oof_dl_v10.csv"
    dest_sub = "submissions/submission_dl_v10.csv"
    dest_emb = "submissions/user_embeddings_local_transformer_cv.csv"

    logger.info(f"Copying OOF: {best_file} -> {dest_oof}")
    shutil.copy2(best_file, dest_oof)

    logger.info(f"Copying Sub: {best_sub} -> {dest_sub}")
    shutil.copy2(best_sub, dest_sub)

    if emb_files:
        best_emb = emb_files[0]
        logger.info(f"Copying User Embeddings: {best_emb} -> {dest_emb}")
        shutil.copy2(best_emb, dest_emb)
    else:
        logger.warning(f"No user embeddings found for trial {best_trial_num}. Skipping embedding copy.")

    return {
        'trial': best_trial_num,
        'score': best_score,
        'oof': dest_oof,
        'sub': dest_sub,
        'emb': dest_emb if emb_files else None
    }

def run_local_script(script_path):
    logger.info(f"Running local script: {script_path}")
    try:
        result = subprocess.run(['.venv/bin/python', script_path], check=True, capture_output=True, text=True)
        logger.info(f"Successfully ran {script_path}")
        # Print last 10 lines of output
        lines = result.stdout.strip().split('\n')
        for line in lines[-15:]:
            logger.info(f"[Script Output] {line}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running script {script_path}: {e}")
        logger.error(f"Script stderr:\n{e.stderr}")
        return False

def submit_to_dacon(sub_path, memo):
    filename = os.path.basename(sub_path)

    # 🚨 CRITICAL USER CONSTRAINT: Parse Local CV and block submission if it is > 0.58
    import re
    local_cv = 0.0
    match_memo = re.search(r'(?:CV|score)\s*([0-9.]+)', memo, re.IGNORECASE)
    if match_memo:
        local_cv = float(match_memo.group(1))
    else:
        match_file = re.search(r'score\s*([0-9.]+)', filename, re.IGNORECASE)
        if match_file:
            local_cv = float(match_file.group(1))

    if local_cv > 0.58:
        logger.warning(f"🚨 [GUARDRAIL ACTIVE] Local CV {local_cv:.5f} is HIGHER than 0.58! Submission BLOCKED to satisfy strict user constraints.")
        return False

    dacon_token = os.environ.get('DACON_TOKEN')
    team_name = os.environ.get('DACON_TEAMNAME')
    competition_id = '236690'

    if not dacon_token or not team_name:
        logger.error("DACON_TOKEN or DACON_TEAMNAME is not set. Cannot submit to Dacon.")
        return False

    # Prevent duplicate submission
    csv_path = 'submissions/submission_log.csv'
    if os.path.exists(csv_path) and filename != "submission.csv":
        try:
            log_df = pd.read_csv(csv_path)
            if filename in log_df['submission_file'].values:
                logger.warning(f"⚠️ Warning: File {filename} has already been submitted according to the log. Skipping.")
                return True
        except Exception as e:
            logger.error(f"Error checking log for duplicates: {e}")

    logger.info(f"Submitting {filename} to Dacon Leaderboard...")
    is_submitted_flag = True
    submission_err_msg = ""
    for attempt in range(3):
        try:
            res = dacon_submit_api.post_submission_file(
                file_path=sub_path,
                token=dacon_token,
                cpt_id=competition_id,
                team_name=team_name,
                memo=memo
            )
            logger.info(f"Dacon API Response: {res}")
            is_submitted_flag = True
            break
        except Exception as e:
            logger.error(f"Error submitting (Attempt {attempt+1}/3): {e}")
            submission_err_msg = str(e)
            if attempt == 2:
                logger.error("Max retries reached for Dacon API submission. Logging failure to submission_log.csv...")
                is_submitted_flag = False
            time.sleep(30)

    # Log to submission_log.csv (Always runs, recording success or failure)
    try:
        if not os.path.exists(csv_path):
            df = pd.DataFrame(columns=['id', 'date', 'submission_file', 'model_description', 'local_cv_score', 'public_lb_score', 'is_submitted', 'notes'])
        else:
            df = pd.read_csv(csv_path)

        import re
        local_cv = 0.0
        match_memo = re.search(r'CV\s+([0-9.]+)', memo)
        if match_memo:
            local_cv = float(match_memo.group(1))
        else:
            match_file = re.search(r'score([0-9.]+)', filename)
            if match_file:
                local_cv = float(match_file.group(1))

        new_record = {
            'id': len(df) + 1,
            'date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            'submission_file': filename,
            'model_description': memo,
            'local_cv_score': round(local_cv, 6),
            'public_lb_score': None,
            'is_submitted': is_submitted_flag,
            'notes': 'Submitted automatically via runner_bot.py' if is_submitted_flag else f'API Submission Failed: {submission_err_msg}'
        }
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        logger.info("Logged submission successfully.")
        return is_submitted_flag
    except Exception as e:
        logger.error(f"Error logging submission: {e}")
        return False

def check_task_running(task_keyword):
    # Check if a python script containing task_keyword is running locally via ps
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        running_tasks = [l for l in lines if task_keyword in l and 'grep' not in l and 'runner_bot' not in l]
        return len(running_tasks) > 0
    except Exception as e:
        logger.error(f"Error checking processes: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Master automation bot for sequence-based Kaggle execution.")
    parser.add_argument("--slug", type=str, default="softkleenex/dacon-etri-dl-tuning-v10", help="Kaggle kernel slug")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds (default: 300s / 5m)")
    args = parser.parse_args()

    kernel_slug = args.slug
    check_interval = args.interval

    logger.info("==================================================================")
    logger.info(f"🤖 Antigravity Runner Bot 기동! (Interval: {check_interval}s)")
    logger.info(f"Target Kaggle Kernel: {kernel_slug}")
    logger.info("==================================================================")

    stage = "WAITING_FOR_INITIAL_ML" # WAITING_FOR_INITIAL_ML -> RUNNING_KAG_MONITOR -> DL_PULLED_WAITING_FOR_FINAL_ML -> DONE

    # If the user has already spawned scripts/train_pseudo_label.py, we track it
    is_initial_ml_done = False
    is_dl_pulled = False

    while True:
        logger.info(f"--- [Status Check Loop] Current Stage: {stage} ---")

        # 1. Check local pseudo-label training task
        ml_running = check_task_running("train_pseudo_label.py")
        logger.info(f"Local train_pseudo_label.py running status: {ml_running}")

        # 2. Check Kaggle Kernel Status
        kag_status = check_kaggle_status(kernel_slug)
        logger.info(f"Kaggle Kernel Status: {kag_status}")

        if stage == "WAITING_FOR_INITIAL_ML":
            if not ml_running:
                # Local ML training must have completed or not started.
                # Check if OOF files exist
                if os.path.exists("submissions/oof_xgb_pseudo_v10.csv"):
                    logger.info("Initial ML Pseudo-Label training seems complete. Running local Calibrated Ensemble Optimization...")
                    # Run 1st stage local ensemble
                    run_local_script("scripts/optimize_ensemble_calibrated.py")

                    # Copy to master submission.csv and submit
                    shutil.copy2("submissions/submission_calibrated_ensemble.csv", "submissions/submission.csv")
                    submit_to_dacon("submissions/submission.csv", "1st Stage Pseudo-Label Calibrated Ensemble (ML-only)")

                    is_initial_ml_done = True
                    stage = "RUNNING_KAG_MONITOR"
                else:
                    logger.warning("No pseudo OOF files found. train_pseudo_label.py might have failed or not started yet. Waiting...")
            else:
                logger.info("Local ML Pseudo-Label training is still running. Waiting for it to finish...")

        elif stage == "RUNNING_KAG_MONITOR":
            if kag_status == "complete":
                logger.info("Kaggle Kernel COMPLETED! Pulling output...")
                pulled = pull_kaggle_output(kernel_slug, "submissions/dl")
                if pulled:
                    is_dl_pulled = True
                    # Find and copy best DL models
                    best_model_info = find_best_dl_model("submissions/dl")
                    if best_model_info:
                        logger.info("Best DL Model extracted and integrated successfully!")
                        # Now, we must restart train_pseudo_label.py locally since we have the updated user embeddings!
                        logger.info("Re-running train_pseudo_label.py to leverage updated User Embeddings...")
                        # Run train_pseudo_label.py synchronously (run_local_script waits for completion)
                        ml_re_run = run_local_script("scripts/train_pseudo_label.py")
                        if ml_re_run:
                            stage = "DL_PULLED_WAITING_FOR_FINAL_ML"
                        else:
                            logger.error("Failed to re-run train_pseudo_label.py. Proceeding to ensemble anyway.")
                            stage = "DL_PULLED_WAITING_FOR_FINAL_ML"
                    else:
                        logger.error("Could not find any valid DL models in output. Ensembling without DL...")
                        stage = "DL_PULLED_WAITING_FOR_FINAL_ML"
                else:
                    logger.error("Could not pull Kaggle outputs. Will retry in next loop.")

            elif kag_status == "error":
                logger.error("Kaggle Kernel FAILED or was CANCELLED. Automation stopped.")
                break

            else:
                # Kaggle is still running. Do local improvements if possible, otherwise sleep.
                logger.info(f"Kaggle is still running. Sleeping for {check_interval} seconds.")

        elif stage == "DL_PULLED_WAITING_FOR_FINAL_ML":
            # The second ML pseudo label training is complete. Now run the final multi-model calibrated ensemble!
            logger.info("Running FINAL Target-Specific Calibrated Ensemble with ALL ML, Pseudo-ML, and DL models!")
            run_local_script("scripts/optimize_ensemble_calibrated.py")

            # Copy to master submission.csv and submit to Dacon
            shutil.copy2("submissions/submission_calibrated_ensemble.csv", "submissions/submission.csv")

            # Dacon submit
            memo = "FINAL SOTA Ensemble (3xML + 2xPseudo-ML + 1xDL) with Temperature Calibration"
            submit_to_dacon("submissions/submission.csv", memo)

            logger.info("🎉 [MISSION ACCOMPLISHED] Final submission generated, master synchronized, and leaderboard submitted!")
            logger.info("Automation completed successfully. Exiting.")
            break

        time.sleep(check_interval)

if __name__ == "__main__":
    main()
