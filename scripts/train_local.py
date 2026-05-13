import os
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
    config['model']['name'] = 'dl_lstm'
    config['model']['epochs'] = 30
    config['training']['n_splits'] = 5
    config['training']['run_name'] = 'local_full_cv_test'
    
    cfg = OmegaConf.create(config)
    trainer = DLTrainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()
