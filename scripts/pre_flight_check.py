import os
import sys
import yaml
import json
import argparse
from pathlib import Path

def run_checks(is_research=False, notebook_dir="notebooks/kaggle_runner"):
    print(f"Starting Pre-Flight Checks for Kaggle Deployment (Research Mode: {is_research}, Notebook Dir: {notebook_dir})...")
    errors = 0

    # 1. Check Configuration
    conf_path = Path("conf/config.yaml")
    if not conf_path.exists():
        print("❌ Error: conf/config.yaml not found.")
        errors += 1
    else:
        try:
            with open(conf_path, "r") as f:
                config = yaml.safe_load(f)
            print("✅ Config file loaded successfully.")
            
            # Check metrics only if it's a submission run
            if not is_research:
                if config.get("training", {}).get("metric") != "logloss":
                    print("❌ Error: Competition metric should be 'logloss'.")
                    errors += 1
                
        except Exception as e:
            print(f"❌ Error reading config: {e}")
            errors += 1

    # 2. Check Kaggle Dataset Metadata
    meta_path = Path("dataset-metadata.json")
    if not meta_path.exists():
        print("❌ Error: dataset-metadata.json not found in root. Run `make kaggle-init-code`.")
        errors += 1
    else:
        try:
            with open(meta_path, "r") as f:
                json.load(f)
            print("✅ Dataset metadata is valid JSON.")
        except Exception as e:
            print(f"❌ Error reading dataset metadata: {e}")
            errors += 1

    # 3. Check Notebook Runner Setup
    runner_meta = Path(notebook_dir) / "kernel-metadata.json"
    if not runner_meta.exists():
        print(f"❌ Error: kernel-metadata.json not found in {notebook_dir}.")
        errors += 1
    else:
        try:
            with open(runner_meta, "r") as f:
                meta = json.load(f)
            if not is_research:
                if str(meta.get("enable_gpu", "false")).lower() == "true" and config.get("model", {}).get("name", "").startswith("dl_"):
                    print("❌ Error: GPU is enabled in kernel-metadata.json for a PyTorch DL model. Due to Kaggle's P100 hardware bug, you MUST disable GPU and use CPU. The dataset is small enough for CPU.")
                    errors += 1
            print("✅ Kernel metadata is valid JSON and settings are safe.")
        except Exception as e:
            print(f"❌ Error reading kernel metadata: {e}")
            errors += 1

    # 4. Check for Data Loader robustness
    dl_path = Path("src/data/data_loader.py")
    if dl_path.exists():
        with open(dl_path, "r") as f:
            content = f.read()
            if "(self.data_dir / \"raw\").exists()" not in content:
                print("❌ Error: data_loader.py does not seem to handle Kaggle's flattened directory structure.")
                errors += 1
            else:
                print("✅ Data Loader has path fallback logic.")
    
    # 5. Check Notebook Generation logic
    gen_path = Path("notebooks/generate_notebooks.py")
    if gen_path.exists():
        with open(gen_path, "r") as f:
            content = f.read()
            if "!pip install" not in content or "omegaconf" not in content or "pytorch-lightning" not in content:
                print("❌ Error: generate_notebooks.py is missing !pip install for required dependencies (like omegaconf or pytorch-lightning).")
                errors += 1
            elif "torch.backends.cudnn.enabled = False" not in content:
                print("❌ Error: generate_notebooks.py is missing the P100 GPU cuDNN fix (torch.backends.cudnn.enabled = False).")
                errors += 1
            else:
                print("✅ Notebook generation includes dependency installation and P100 GPU fix.")
                
    # 6. Compute Quota & Runtime Safety Check
    if not is_research:
        is_tuning = config.get("training", {}).get("tune_hyperparameters", False)
        n_splits = config.get("training", {}).get("n_splits", 5)
        
        if is_tuning:
            n_trials = config.get("training", {}).get("n_trials", 10)
            total_models = n_trials * n_splits * 7 # 7 targets
            if total_models > 500:
                print(f"❌ Error: Your configuration will train {total_models} models! This will likely timeout or eat your entire Kaggle GPU weekly quota. Reduce n_trials or disable tuning.")
                errors += 1
            else:
                print(f"✅ Compute quota check passed. Total tuning models to train: {total_models}")
        else:
            print("✅ No hyperparameter tuning. Compute time is safe.")
        
    if errors > 0:
        print(f"\n🚨 Pre-flight checks failed with {errors} errors. Please fix before pushing to Kaggle.")
        sys.exit(1)
    else:
        print("\n🚀 All pre-flight checks passed! Ready for launch.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-flight checks for Kaggle deployment.")
    parser.add_argument("--research", action="store_true", help="Skip competition specific strict checks for research notebooks.")
    parser.add_argument("--notebook_dir", type=str, default="notebooks/kaggle_runner", help="Path to the notebook directory being pushed.")
    args = parser.parse_args()
    
    run_checks(is_research=args.research, notebook_dir=args.notebook_dir)
