import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# DACON credentials must come from the caller's environment.
if not os.environ.get("DACON_TOKEN") or not os.environ.get("DACON_TEAMNAME"):
    raise RuntimeError("DACON_TOKEN and DACON_TEAMNAME must be set in the environment.")

from scripts.runner_bot import submit_to_dacon


def get_latest_unique_submission():
    sub_dir = PROJECT_ROOT / "submissions"
    unique_subs = sorted(sub_dir.glob("submission_id_*_score*.csv"), key=os.path.getmtime)
    if unique_subs:
        return unique_subs[-1]
    return None

def main():
    parser = argparse.ArgumentParser(description="Submit unique SOTA prediction assets directly to DACON API.")
    parser.add_argument("--file", type=str, default=None, help="Path to the specific unique submission CSV file.")
    args = parser.parse_args()

    if args.file:
        sub_path = Path(args.file)
    else:
        # Fallback to the latest newly generated unique submission file
        latest = get_latest_unique_submission()
        if latest:
            sub_path = latest
        else:
            sub_path = PROJECT_ROOT / "submissions" / "submission_calibrated_ensemble.csv"

    if not sub_path.exists():
        print(f"Error: Submission file {sub_path} does not exist. Verify paths.")
        return

    print(f"🚀 Direct Submission Mode: Preparing to submit {sub_path.name} directly!")

    # Auto-generate dynamic memo matching the specific unique asset
    memo = f"SOTA Calibrated Ensemble - Unique Asset: {sub_path.name}"
    success = submit_to_dacon(str(sub_path), memo)

    if success:
        print(f"🎉 Congratulations! Successfully submitted {sub_path.name} directly to DACON API!")
    else:
        print("❌ Direct submission failed. Verify logs.")

if __name__ == "__main__":
    main()
