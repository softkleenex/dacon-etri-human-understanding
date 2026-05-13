from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()


class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
        # In Kaggle, the dataset was uploaded directly from data/raw, so 'raw' folder doesn't exist
        if (self.data_dir / "raw").exists():
            self.raw_dir = self.data_dir / "raw"
        else:
            self.raw_dir = self.data_dir
            
        self.processed_dir = self.data_dir / "processed"
        self.items_dir = self.raw_dir / "ch2025_data_items"

    def load_metrics(self) -> pd.DataFrame | None:
        """
        450일분의 학습용 레이블(정답) 데이터(ch2026_metrics_train.csv)를 로드합니다.
        """
        file_path = self.raw_dir / "ch2026_metrics_train.csv"
        if not file_path.exists():
            logger.warning(f"Metrics file does not exist: {file_path}")
            return None

        logger.info(f"Loading metrics file: {file_path.name}")
        return pd.read_csv(file_path)

    def load_sensor_data(self) -> dict[str, pd.DataFrame]:
        """
        ch2025_data_items 디렉토리에 있는 모든 센서 데이터(Parquet) 파일을 읽어 딕셔너리로 반환합니다.
        총 12개의 스마트폰 및 웨어러블 센서(라이프로그) 데이터를 포함합니다.
        """
        data_dict = {}
        if not self.items_dir.exists():
            logger.warning(f"Sensor data directory does not exist: {self.items_dir}")
            return data_dict

        for file_path in self.items_dir.glob("*.parquet"):
            logger.info(f"Loading sensor file: {file_path.name}")
            data_dict[file_path.stem] = pd.read_parquet(file_path)

        if not data_dict:
            logger.info("No sensor files found in the ch2025_data_items directory.")

        return data_dict

    def load_sample_submission(self) -> pd.DataFrame | None:
        """
        제출 양식 파일(ch2026_submission_sample.csv)을 로드합니다.
        """
        file_path = self.raw_dir / "ch2026_submission_sample.csv"
        if not file_path.exists():
            logger.warning(f"Sample submission file does not exist: {file_path}")
            return None

        logger.info(f"Loading sample submission file: {file_path.name}")
        return pd.read_csv(file_path)
