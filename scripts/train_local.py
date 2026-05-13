import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import yaml
from pathlib import Path
from omegaconf import OmegaConf

sys.path.append(str(Path.cwd()))
from src.models.dl_trainer import DLTrainer

def main():
    print("Starting Full Local CV Training...")
    
    with open('conf/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # We want to run full epochs locally to check for overfitting
    config['model']['name'] = 'dl_transformer'
    config['model']['epochs'] = 30
    config['model']['seq_len'] = 14
    config['training']['n_splits'] = 5
    config['training']['run_name'] = 'local_transformer_cv'
    
    cfg = OmegaConf.create(config)
    trainer = DLTrainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()
