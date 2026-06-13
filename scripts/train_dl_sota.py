import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
from pathlib import Path

from omegaconf import OmegaConf

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.dl_trainer import DLTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info("=========================================================================")
    logger.info("⚡ STARTING SOTA-V30 ROBUST DEEP LEARNING SEQUENCE ANCHOR RETRAINING")
    logger.info("=========================================================================")

    # 1. Load configuration from conf/config.yaml
    config_path = PROJECT_ROOT / "conf" / "config.yaml"
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        return

    cfg = OmegaConf.load(config_path)

    # 2. Force settings for highly robust, non-leak sequential DL training
    cfg.env = "local"
    cfg.model.name = "dl_transformer"  # Use Transformer sequence model
    cfg.model.seq_len = 14
    cfg.model.epochs = 20              # Sufficient for small N=451 dataset on CPU
    cfg.training.tune_hyperparameters = False  # Avoid slow hyperparameter search

    cfg.training.run_name = "dl_sota_tuned"

    # Load optimal DL hyperparameters if available
    opt_path = PROJECT_ROOT / "checks" / "dl_model_opt_hyperparams.json"
    if opt_path.exists():
        import json
        try:
            with open(opt_path, "r") as f:
                opt_info = json.load(f)
            best_params = opt_info.get("best_params", {})
            cfg.model.lr = float(best_params.get("lr", 0.00549))
            cfg.model.hidden_dim = int(best_params.get("hidden_dim", 256))
            cfg.model.num_layers = int(best_params.get("num_layers", 3))
            cfg.model.dropout = float(best_params.get("dropout", 0.26))
            cfg.model.batch_size = int(best_params.get("batch_size", 32))
            logger.info("Successfully loaded optimized hyperparameters from checks/dl_model_opt_hyperparams.json")
        except Exception as e:
            logger.error(f"Failed to load optimized DL hyperparameters: {e}")
            cfg.model.lr = 0.00549
            cfg.model.hidden_dim = 256
            cfg.model.num_layers = 3
            cfg.model.dropout = 0.26
            cfg.model.batch_size = 32
    else:
        cfg.model.lr = 0.00549
        cfg.model.hidden_dim = 256
        cfg.model.num_layers = 3
        cfg.model.dropout = 0.26
        cfg.model.batch_size = 32

    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Model Architecture: {cfg.model.name} (Seq Len: {cfg.model.seq_len})")
    logger.info(f"Hyperparameters: LR={cfg.model.lr}, HiddenDim={cfg.model.hidden_dim}, Layers={cfg.model.num_layers}, Epochs={cfg.model.epochs}")

    # 3. Instantiate and run Deep Learning pipeline
    trainer = DLTrainer(cfg)

    hf_emb_path = PROJECT_ROOT / "data" / "processed" / "hf_chronos_embeddings.csv"
    if hf_emb_path.exists():
        logger.info(f"✅ Found existing HF Chronos embeddings at {hf_emb_path}. Skipping extraction to save CPU time.")
    else:
        try:
            logger.info("Extracting HF Chronos Embeddings for temporal signals...")
            import extract_hf_features
            extract_hf_features.main()
        except Exception as e:
            logger.warning(f"Could not extract HF Chronos features: {e}. Proceeding without them...")

    logger.info("Starting DL training loop...")
    avg_score = trainer.run()

    logger.info(f"🎉 DL Training Complete! 5-Fold average Log-Loss: {avg_score:.5f}")

    # 4. Copy the generated optimal checkpoint outputs to baseline v10 OOF path
    import shutil

    logger.info("Mapping and aligning newly trained DL OOF predictions to ensemble anchors...")

    submission_dir = PROJECT_ROOT / "submissions"

    # Find all recently updated DL files matching run_name
    new_oofs = sorted(submission_dir.glob("oof_dl_sota_tuned_score*.csv"), key=os.path.getmtime)
    new_subs = sorted(submission_dir.glob("submission_dl_sota_tuned_score*.csv"), key=os.path.getmtime)

    if new_oofs and new_subs:
        latest_oof = new_oofs[-1]
        latest_sub = new_subs[-1]

        target_oof_path = submission_dir / "oof_dl_v10.csv"
        target_sub_path = submission_dir / "submission_dl_v10.csv"

        shutil.copy2(latest_oof, target_oof_path)
        shutil.copy2(latest_sub, target_sub_path)

        logger.info(f"✅ Successfully updated {target_oof_path.name} from {latest_oof.name}")
        logger.info(f"✅ Successfully updated {target_sub_path.name} from {latest_sub.name}")
    else:
        logger.warning("Could not find exact run_name naming matches. Searching fallback 'dl' files...")
        alternative_oofs = sorted(submission_dir.glob("oof_*_score*.csv"), key=os.path.getmtime)
        alternative_subs = sorted(submission_dir.glob("submission_*_score*.csv"), key=os.path.getmtime)
        
        for f in alternative_oofs:
            if "dl" in f.name:
                shutil.copy2(f, submission_dir / "oof_dl_v10.csv")
                logger.info(f"✅ Aligned alternative OOF: {f.name} -> oof_dl_v10.csv")
                break
        for f in alternative_subs:
            if "dl" in f.name:
                shutil.copy2(f, submission_dir / "submission_dl_v10.csv")
                logger.info(f"✅ Aligned alternative Submission: {f.name} -> submission_dl_v10.csv")
                break

if __name__ == '__main__':
    main()
