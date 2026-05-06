# 데이터 전처리 및 MLOps 확장 라이브러리 활용 매뉴얼

기본적인 `pandas`와 순수 `torch` 반복문(for loop) 코드만으로 대회를 치르면 속도 병목(Bottleneck)에 갇혀 실험 횟수가 현저히 줄어듭니다. 연산 속도와 코드 유지보수성을 극한으로 끌어올리는 도구들입니다.

## 🎯 사용 목적 및 방법 (Why & How to use?)

### 1. Polars (데이터 전처리 병목 해결)
Pandas의 `apply`나 `groupby`가 너무 느려 피처 엔지니어링에만 수십 분이 걸릴 때 사용합니다. Rust 기반의 멀티스레딩으로 속도를 수십 배 단축시킵니다.
```python
import polars as pl

# Pandas 대신 Polars 적용 시, 대용량 로그 데이터 그룹화가 초 단위로 끝남
df = pl.read_csv('data/raw/train.csv')
features = df.group_by("user_id").agg([
    pl.col("purchase_amount").sum().alias("total_amount"),
    pl.col("session_time").mean().alias("avg_session")
])
```

### 2. PyTorch Lightning (딥러닝 훈련 코드 단순화)
모델 학습 시 GPU 할당(`to(device)`), 역전파(`loss.backward()`), W&B 로깅 등을 직접 짤 필요 없이, 핵심 로직만 남기고 보일러플레이트를 완전히 지워버립니다.

```python
import lightning as L
from lightning.pytorch.loggers import WandbLogger

class MyModel(L.LightningModule):
    # 모델 껍데기와 Loss만 정의
    def training_step(self, batch, batch_idx):
        loss = self.criterion(self(batch[0]), batch[1])
        return loss

# Multi-GPU, 16-bit 메모리 절약, W&B 연동이 단 1줄에 끝
trainer = L.Trainer(accelerator="auto", precision="16-mixed", logger=WandbLogger())
trainer.fit(MyModel(), train_loader)
```

### 3. DVC (대용량 파일 형상 관리)
수십 GB의 원본 CSV 파일이나, 수 GB에 달하는 모델 가중치 파일(`.pt`)은 GitHub에 올릴 수 없습니다. DVC를 통해 데이터와 가중치 파일의 "변경 이력"을 추적하고 관리하여 실험 재현성을 확보합니다.

```bash
dvc init
dvc add data/raw/train.csv    # 원본 데이터 버전 추적 시작
git add data/raw/train.csv.dvc # Git에는 메타데이터 포인터만 커밋
```

### 4. 도메인 특화 증강 (tsaug, Geopandas)
- **시계열 (tsaug)**: 원본 시계열 데이터에 강제로 노이즈를 섞고 축을 일그러뜨려(Time Warping), 모델이 변동성에 강해지도록(Robust) 학습 데이터를 부풀릴 때 사용합니다.
- **위치/공간 (Geopandas)**: 모기 궤적, 선박 이동 등 (x, y) 좌표 데이터를 다룰 때, 두 궤적이 교차하는지 구역(Polygon) 안에 들어왔는지를 수학적 공식 없이 직관적인 내장 함수로 계산할 때 사용합니다.
