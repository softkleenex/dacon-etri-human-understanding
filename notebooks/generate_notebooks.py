import os
from pathlib import Path

import nbformat as nbf

KAGGLE_SETUP_BOILERPLATE = [
    nbf.v4.new_code_cell(
        "# Removed autoreload to prevent 'imp' module error in Python 3.12"
    ),
    nbf.v4.new_code_cell(
        "!pip install -q optuna pytorch-lightning omegaconf catboost"
    ),
    nbf.v4.new_code_cell(
        "!ls -laR /kaggle/input || true\n!pwd"
    ),
    nbf.v4.new_code_cell("""import os
import sys
import warnings
from pathlib import Path
import torch

warnings.filterwarnings('ignore')

# Fix for Kaggle P100 GPU PyTorch LSTM bug (CUDA error: no kernel image is available)
torch.backends.cudnn.enabled = False

# Detect if running on Kaggle
is_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') != ''

if is_kaggle:
    # Auto-discover the code dataset path
    input_dir = Path('/kaggle/input')
    code_dataset_path = None
    for p in input_dir.rglob('conf/config.yaml'):
        code_dataset_path = p.parents[1]
        break
        
    if code_dataset_path is None:
        raise FileNotFoundError("Could not find conf/config.yaml inside /kaggle/input")
        
    CODE_DATASET_PATH = code_dataset_path
    if str(CODE_DATASET_PATH) not in sys.path:
        sys.path.append(str(CODE_DATASET_PATH))
    # Kaggle default working directory is /kaggle/working
    PROJECT_ROOT = Path('/kaggle/working')
    print(f"Running on Kaggle. Added {CODE_DATASET_PATH} to sys.path")
else:
    # If running locally, make sure the project root is in sys.path
    PROJECT_ROOT = Path.cwd().resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)
    print(f"Running locally. Changed working directory to {PROJECT_ROOT}")"""),
    nbf.v4.new_code_cell("""import yaml

# Load Config
conf_path = Path('conf/config.yaml')
if is_kaggle:
    conf_path = CODE_DATASET_PATH / 'conf/config.yaml'

with open(conf_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Override environment if running on Kaggle
if is_kaggle:
    config['env'] = 'kaggle'
    
    # Auto-discover data_dir
    data_dir = None
    for p in Path('/kaggle/input').rglob('ch2026_metrics_train.csv'):
        data_dir = p.parent
        break
        
    if data_dir is not None:
        config['paths']['kaggle']['data_dir'] = str(data_dir)
        print(f"Auto-discovered data_dir: {data_dir}")

env = config['env']
paths = config['paths'][env]

print(f"Environment: {env}")
print(f"Data Dir: {paths['data_dir']}")
print(f"Model Dir: {paths['model_dir']}")""")
]

def create_kaggle_runner():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    }

    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Kaggle/Dacon Pipeline Runner\n\nThis notebook serves as the main entry point for running the pipeline on Kaggle or other cloud environments. It automatically detects the environment and adjusts paths via the `config.yaml`."
        )
    ] + KAGGLE_SETUP_BOILERPLATE + [
        nbf.v4.new_code_cell("""# Setup output directories
for dir_name in ['processed_dir', 'model_dir', 'output_dir']:
    if dir_name in paths:
        Path(paths[dir_name]).mkdir(parents=True, exist_ok=True)"""),
        nbf.v4.new_markdown_cell(
            "## Execute Pipeline\n\nNow we can import our modules from `src` and run the pipeline steps."
        ),
        nbf.v4.new_code_cell("""from omegaconf import OmegaConf
from src.models.trainer import Trainer
from src.models.dl_trainer import DLTrainer
import sys
from pathlib import Path

# Since config is a dict, we can convert it to DictConfig
cfg = OmegaConf.create(config)

is_dl = cfg.model.get("name", "").startswith("dl_")
is_tuning = cfg.training.get("tune_hyperparameters", False)

if is_dl and is_tuning:
    print("Running Massive DL Hyperparameter Tuning with Optuna...")
    # Add scripts to path so we can import the tuning script
    scripts_dir = str(Path(CODE_DATASET_PATH) / 'scripts' if is_kaggle else Path(PROJECT_ROOT) / 'scripts')
    if scripts_dir not in sys.path:
        sys.path.append(scripts_dir)
    import tune_dl_optuna
    tune_dl_optuna.main()
elif is_dl:
    trainer = DLTrainer(cfg)
    trainer.run()
else:
    trainer = Trainer(cfg)
    trainer.run()

print("Pipeline executed successfully!")"""),
    ]

    runner_dir = Path("notebooks/kaggle_runner")
    runner_dir.mkdir(parents=True, exist_ok=True)
    with open(runner_dir / "main.ipynb", "w", encoding='utf-8') as f:
        nbf.write(nb, f)


def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    }

    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Exploratory Data Analysis (EDA) / Research\n\nThis notebook is intended for deep data exploration and feature analysis on Kaggle or locally. It uses the shared Kaggle boilerplate to ensure paths and dependencies are correctly loaded."
        )
    ] + KAGGLE_SETUP_BOILERPLATE + [
        nbf.v4.new_code_cell("""import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.data_loader import DataLoader
from src.data.make_dataset import build_training_dataset
from src.features.build_features import engineer_features

# Load your datasets here
print("Loading data via standard DataLoader...")
data_loader = DataLoader(data_dir=paths['data_dir'])
metrics_df = data_loader.load_metrics()
sensor_data_dict = data_loader.load_sensor_data()

print("Building base dataset...")
df_train_raw = build_training_dataset(metrics_df, sensor_data_dict)

print("Engineering features...")
df_train = engineer_features(df_train_raw)
print(f"Data shape: {df_train.shape}")
"""),
        nbf.v4.new_markdown_cell("## Basic Statistics"),
        nbf.v4.new_code_cell("""df_train.info()
display(df_train.describe())"""),
        nbf.v4.new_markdown_cell("## Target Distribution"),
        nbf.v4.new_code_cell("""plt.figure(figsize=(8, 5))
sns.histplot(df_train[config['training']['target_col']], bins=50, kde=True)
plt.title('Target Distribution')
plt.show()"""),
    ]

    eda_dir = Path("notebooks/01-EDA")
    eda_dir.mkdir(parents=True, exist_ok=True)
    with open(eda_dir / "01-basic_eda.ipynb", "w", encoding='utf-8') as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    # Always run from the project root or adjust accordingly
    os.chdir(Path(__file__).resolve().parents[1])
    create_kaggle_runner()
    create_eda_notebook()
    print("Notebooks generated successfully.")
