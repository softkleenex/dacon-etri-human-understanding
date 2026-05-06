# 데이터 전처리 및 MLOps 확장 라이브러리 가이드

기본적인 Pandas와 Scikit-Learn만으로 부족할 때, 데이터 처리 속도와 모델 파이프라인의 완성도를 비약적으로 높여주는 핵심 라이브러리들입니다.

## 1. Polars (고속 데이터 처리)
- **개요**: Rust로 작성된 초고속 데이터프레임 라이브러리로, Pandas보다 수 배에서 수십 배 빠릅니다.
- **적용처**: 대규모 시계열 데이터 파싱, 천만 건 단위의 로그 데이터 GroupBy 및 Aggregation 연산, 수백 개의 Rolling Window 피처 생성 시 강력히 권장합니다.
- **예시**:
  ```python
  import polars as pl
  # Pandas보다 훨씬 빠르게 병렬 연산으로 피처를 생성
  df = pl.read_csv('data/raw/train.csv')
  df = df.with_columns(
      pl.col("temperature").rolling_mean(window_size=3).alias("temp_roll_3")
  )
  ```

## 2. PyTorch Lightning (MLOps)
- **개요**: 복잡한 PyTorch 학습 루프(보일러플레이트 코드)를 깔끔한 모듈 형태로 묶어주는 래퍼 라이브러리.
- **적용처**: 딥러닝 훈련 시 분산 학습(Multi-GPU), 16-bit Mixed Precision, 체크포인트 저장, Early Stopping 등을 단 1줄의 설정으로 적용합니다.
- **예시**:
  ```python
  import lightning as L
  
  class MyModel(L.LightningModule):
      def training_step(self, batch, batch_idx): ...
      def configure_optimizers(self): ...
      
  # 트레이너 설정
  trainer = L.Trainer(
      max_epochs=10, 
      accelerator="auto",  # 자동으로 mps나 cuda를 잡아줌
      precision="16-mixed",
      logger=L.pytorch.loggers.WandbLogger() # W&B 연동
  )
  trainer.fit(model, train_loader)
  ```

## 3. DVC (Data Version Control)
- **개요**: Git이 소스 코드를 버전 관리하듯, 대용량 데이터셋과 모델 가중치 파일(`.pt`)의 버전을 관리합니다.
- **적용처**: `data/`와 `models/` 디렉토리 내의 파일은 Git에 올리지 못하므로 DVC를 초기화(`dvc init`)하고 S3나 구글 드라이브를 원격 저장소로 연결하여 재현성을 보장합니다.

## 4. 도메인 특화 라이브러리
- **Geopandas**: 위치(위경도, X/Y/Z) 기반 공간 데이터 분석, 기하학적 펜스 충돌 및 교차 판정.
- **tsaug**: 시계열 데이터에 대한 전문적인 수학적 증강 기법(Time Warping, Magnitude Jittering 등) 적용.
