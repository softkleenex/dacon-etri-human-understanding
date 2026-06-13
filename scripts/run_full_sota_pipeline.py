import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger("Full_SOTA_Pipeline_Runner")

def run_script(python_bin, script_path):
    logger.info(f"🚀 Executing: {script_path.name}...")
    try:
        res = subprocess.run([str(python_bin), str(script_path)], check=True)
        if res.returncode == 0:
            logger.info(f"✅ {script_path.name} completed successfully.")
            return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {script_path.name} failed with exit code {e.returncode}.")
        return False
    return False

def main():
    logger.info("=========================================================================")
    logger.info("🏁 STARTING FULL END-TO-END SOTA PIPELINE OPTIMIZATION")
    logger.info("=========================================================================")

    python_bin = PROJECT_ROOT / ".venv" / "bin" / "python"
    if not python_bin.exists():
        python_bin = "python"

    # Step 1: Tune DL Transformer model
    tune_dl_script = PROJECT_ROOT / "scripts" / "tune_dl_optuna.py"
    if not run_script(python_bin, tune_dl_script):
        logger.error("Stop execution at Step 1 (DL tuning).")
        sys.exit(1)

    # Step 2: Train SOTA DL sequence anchor with optimized parameters
    train_dl_script = PROJECT_ROOT / "scripts" / "train_dl_sota.py"
    if not run_script(python_bin, train_dl_script):
        logger.error("Stop execution at Step 2 (DL training).")
        sys.exit(1)

    # Step 3: Retrain GBDT base models (incorporating new DL anchor logits)
    train_gbdt_script = PROJECT_ROOT / "scripts" / "train_ultimate_pipeline_clean_v32.py"
    if not run_script(python_bin, train_gbdt_script):
        logger.error("Stop execution at Step 3 (GBDT training).")
        sys.exit(1)

    # Step 4: Run final 4-stage Stacking Ensemble with Optuna Recalibration
    ensemble_script = PROJECT_ROOT / "scripts" / "run_nvidia_stacking_ensemble.py"
    if not run_script(python_bin, ensemble_script):
        logger.error("Stop execution at Step 4 (Stacking Ensemble).")
        sys.exit(1)

    logger.info("=========================================================================")
    logger.info("🎉 SUCCESS! FULL SOTA PIPELINE OPTIMIZATION COMPLETE!")
    logger.info("=========================================================================")

if __name__ == "__main__":
    main()
