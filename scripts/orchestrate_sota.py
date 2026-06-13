import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Setup directories
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
OOF_PATTERN = str(SUBMISSIONS_DIR / "oof_*_v10.csv")

def get_oof_files():
    return sorted(glob.glob(OOF_PATTERN))

def get_file_mod_times(files):
    return {f: os.path.getmtime(f) for f in files if os.path.exists(f)}

def run_python_script(script_name):
    script_path = PROJECT_ROOT / "scripts" / script_name
    print(f"\n[Orchestrator] 🚀 Running {script_name}...")
    try:
        # Run python script in the project virtualenv python
        python_bin = PROJECT_ROOT / ".venv" / "bin" / "python"
        if not python_bin.exists():
            python_bin = "python"

        result = subprocess.run([str(python_bin), str(script_path)], check=True, capture_output=True, text=True)
        print(f"[Orchestrator] ✅ {script_name} completed successfully.")

        # Log stdout snippets
        stdout_lines = result.stdout.strip().split('\n')
        for line in stdout_lines[-12:]:
            print(f"  [Stdout] {line}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Orchestrator] ❌ {script_name} FAILED with exit code {e.returncode}!")
        print(f"[Orchestrator] Error Output:\n{e.stderr}")
        return False

def execute_pipeline():
    print("\n" + "="*70)
    print(f"[Orchestrator] Starting SOTA-v28 Blending & Submission Pipeline at {pd.Timestamp.now()}")
    print("="*70)

    # 1. SLSQP Stacking & Beta Calibration Optimization (SOTA-v28)
    success = run_python_script("stack_ensemble_v28.py")
    if not success:
        print("[Orchestrator] ❌ Pipeline aborted during stack_ensemble_v28.py")
        return False

    # 2. Strict Pre-Submission Validation
    success = run_python_script("pre_submission_validator.py")
    if not success:
        print("[Orchestrator] ❌ Pipeline aborted during pre_submission_validator.py")
        return False

    # 3. DACON API Submission
    success = run_python_script("submit_calibrated.py")
    if not success:
        print("[Orchestrator] ❌ Pipeline failed during submit_calibrated.py")
        return False

    print("\n" + "="*70)
    print(f"[Orchestrator] 🎉 Pipeline Execution Successfully Completed at {pd.Timestamp.now()}")
    print("="*70)
    return True

def monitor_loop():
    print("==================================================================")
    print("🤖 SOTA-v28 OOF Real-Time Orchestration Daemon Started")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Watching OOF files pattern: {OOF_PATTERN}")
    print("==================================================================")

    # 1. Initial State Retrieval
    oof_files = get_oof_files()
    print(f"[Orchestrator] Found {len(oof_files)} active OOF files on startup:")
    for f in oof_files:
        mtime = pd.Timestamp(os.path.getmtime(f), unit='s').tz_localize('UTC').tz_convert('Asia/Seoul')
        print(f"  - {os.path.basename(f)} (Last Modified: {mtime})")

    tracked_mod_times = get_file_mod_times(oof_files)

    # 2. Immediate Startup Execution to ensure everything is synchronized and submitted
    print("\n[Orchestrator] Performing immediate initial synchronization run...")
    execute_pipeline()

    # 3. Infinite Polling Loop
    print("\n[Orchestrator] Entering real-time monitoring loop (polling every 10s)...")
    while True:
        try:
            time.sleep(10)

            current_files = get_oof_files()
            current_mod_times = get_file_mod_times(current_files)

            triggered = False
            triggered_files = []

            # Check for new files or modified files
            for f in current_files:
                if f not in tracked_mod_times:
                    print(f"\n[Orchestrator] 🔔 New OOF file detected: {os.path.basename(f)}")
                    triggered = True
                    triggered_files.append(f)
                elif current_mod_times[f] > tracked_mod_times[f]:
                    old_time = pd.Timestamp(tracked_mod_times[f], unit='s').tz_localize('UTC').tz_convert('Asia/Seoul')
                    new_time = pd.Timestamp(current_mod_times[f], unit='s').tz_localize('UTC').tz_convert('Asia/Seoul')
                    print(f"\n[Orchestrator] 🔔 OOF file updated: {os.path.basename(f)} ({old_time} -> {new_time})")
                    triggered = True
                    triggered_files.append(f)

            if triggered:
                # Wait 5 seconds to let any active file writes complete fully
                print("[Orchestrator] Waiting 5 seconds to ensure file write operations complete...")
                time.sleep(5)

                # Re-fetch modification times to match post-wait status
                updated_files = get_oof_files()
                tracked_mod_times = get_file_mod_times(updated_files)

                # Execute full pipeline
                execute_pipeline()

        except KeyboardInterrupt:
            print("\n[Orchestrator] 🛑 Monitoring loop terminated by user.")
            break
        except Exception as e:
            print(f"\n[Orchestrator] ⚠️ Error in monitor loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_loop()
