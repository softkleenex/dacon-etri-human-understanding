import pandas as pd
from pathlib import Path

def load_data(file_path: str | Path) -> pd.DataFrame:
    """Loads dataset from the specified file path."""
    path = Path(file_path)
    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix == '.parquet':
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
