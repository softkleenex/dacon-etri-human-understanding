import sys
import yaml
from pathlib import Path
from omegaconf import OmegaConf
sys.path.append(str(Path.cwd()))
from src.models.dl_trainer import DLTrainer

# Load Config
with open('conf/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Change to dl_lstm
config['model']['name'] = 'dl_lstm'
config['model']['seq_len'] = 3
config['model']['epochs'] = 1 # very short test
config['training']['n_splits'] = 2

cfg = OmegaConf.create(config)
trainer = DLTrainer(cfg)
trainer.run()
