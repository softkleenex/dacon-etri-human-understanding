# 데이터 전처리 및 MLOps 확장 라이브러리 가이드

기본적인 `pandas`와 순수 `torch` 코드만으로 대회를 치르면 속도 병목(Bottleneck)과 반복적인 보일러플레이트 코드 작성에 지치게 됩니다. 이 문서에서는 파이프라인의 완성도를 비약적으로 높여주는 핵심 라이브러리 활용법을 다룹니다.

---

## 1. Polars (초고속 데이터 처리)
- **개요**: Rust 기반의 데이터프레임 라이브러리로, Pandas보다 수 배~수십 배 빠른 병렬 처리를 지원합니다.
- **적용처**: 수천만 건 이상의 로그 데이터 집계(Aggregation)나, 시계열 데이터에서 수백 개의 Rolling Window 통계량(이동 평균, 분산 등)을 추출할 때 필수적입니다.
- **실전 예시**:
  ```python
  import polars as pl
  
  # Pandas의 groupby-apply 보다 압도적으로 빠름
  df = pl.read_csv('data/raw/train.csv')
  feature_df = df.group_by("user_id").agg([
      pl.col("purchase_amount").sum().alias("total_amount"),
      pl.col("login_time").max().alias("last_login")
  ])
  ```

---

## 2. PyTorch Lightning (딥러닝 MLOps 래퍼)
- **개요**: PyTorch의 복잡한 학습 루프(`for epoch: for batch: loss.backward()...`)를 깔끔하게 지워주고, 핵심 모델 구조에만 집중하게 해주는 라이브러리입니다.
- **강점**: Multi-GPU 설정, 16-bit Mixed Precision(VRAM 절약), W&B 로깅, 조기 종료(Early Stopping) 등을 단 한 줄의 코드로 완벽하게 제어합니다.
- **실전 예시**:
  ```python
  import lightning as L
  from lightning.pytorch.loggers import WandbLogger
  from lightning.pytorch.callbacks import EarlyStopping
  
  # LightningModule은 nn.Module을 상속하며, 훈련 흐름만 정의합니다.
  class MyModel(L.LightningModule):
      def training_step(self, batch, batch_idx):
          x, y = batch
          loss = self.criterion(self(x), y)
          self.log('train_loss', loss)
          return loss
          
  # 단 1줄로 MLOps 적용 완료
  trainer = L.Trainer(
      max_epochs=100, 
      accelerator="auto",               # Mac의 MPS나 클라우드의 CUDA 자동 할당
      precision="16-mixed",             # 16-bit 연산으로 학습 속도 2배, 메모리 절반
      logger=WandbLogger(),             # W&B 실시간 동기화
      callbacks=[EarlyStopping(monitor="val_loss", patience=5)]
  )
  trainer.fit(model, train_loader, val_loader)
  ```

---

## 3. DVC (Data Version Control)
- **개요**: 대용량 데이터셋 파일(`.csv`, `.parquet`)과 모델 가중치 파일(`.pt`)의 변경 이력을 Git처럼 관리해주는 도구입니다.
- **적용처**: `data/`와 `models/` 폴더 내의 무거운 파일들은 GitHub에 올릴 수 없습니다. `dvc init` 후 S3나 Google Drive를 원격 저장소로 연결하여, 코드 버전(Git)과 데이터 버전(DVC)을 1:1로 매핑하여 완벽한 재현성(Reproducibility)을 보장합니다.

---

## 4. 특수 도메인 라이브러리
- **Geopandas**: 모기 비행 궤적, 해운 선박 이동 등 위경도(X, Y) 좌표계 데이터를 다룰 때 사용합니다. 구역 펜스 설정, 두 궤적 간의 교차 판정, 곡률 계산 등 수학적 연산을 매우 직관적으로 처리합니다.
- **tsaug**: 시계열 데이터 파인튜닝 시, 센서 노이즈 추가(Jittering), 시간축 왜곡(Time Warping) 등의 물리적 증강(Augmentation) 기법을 단 1줄로 지원하여 모델의 오버피팅을 방지합니다.
