# 데이터 전처리 및 MLOps 확장 라이브러리 매뉴얼

Pandas와 순수 PyTorch 루프의 속도 및 유지보수 병목을 해결하고, 대용량 파일의 형상 관리를 보장하기 위해 도입한 확장 라이브러리 생태계입니다.

## 🎯 사용 목적 (Why use this?)
- **Polars**: Pandas 대비 수 배~수십 배 빠른 병렬 처리로 수백 개의 통계 피처 생성 및 대용량 로그 집계 속도를 단축합니다.
- **PyTorch Lightning**: PyTorch의 복잡한 훈련 반복문(for loop) 보일러플레이트를 제거하고 Multi-GPU, 16-bit Mixed Precision 등을 1줄로 처리합니다.
- **DVC (Data Version Control)**: Git에 올릴 수 없는 수십 GB의 데이터(`.csv`)와 가중치(`.pt`)의 변경 이력을 완벽하게 관리(재현성 확보)합니다.

## 🚀 사용 방법 (How to use?)

### 1. 인증 및 환경 변수 설정
위 라이브러리들은 로컬 연산 및 버전 관리 도구로, 별도의 전역 API 키를 필요로 하지 않습니다. `make install` 수행 시 `requirements.txt`를 통해 자동 설치됩니다.

### 2. Polars 고속 전처리 스니펫
```python
import polars as pl

# Pandas의 groupby-apply 대비 압도적 연산 속도 보장
df = pl.read_csv('data/raw/train.csv')
features = df.group_by("user_id").agg([
    pl.col("purchase_amount").sum().alias("total_amount"),
    pl.col("session_time").mean().alias("avg_session")
])
```

### 3. PyTorch Lightning MLOps 스니펫
훈련 루프를 분리하고 `Trainer`에 핵심 옵션들을 몰아넣습니다.
```python
import lightning as L
from lightning.pytorch.loggers import WandbLogger

class MyModel(L.LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.criterion(self(batch[0]), batch[1])
        return loss

# 16-bit 메모리 절약(precision) 및 W&B 전역 토큰 로거 자동 연동
trainer = L.Trainer(
    accelerator="auto", 
    precision="16-mixed", 
    logger=WandbLogger(project="dacon-comp")
)
trainer.fit(MyModel(), train_loader)
```

## ⚠️ 실전 주의 사항
- **DVC 형상 관리 분리**: `dvc add`로 추가된 원본 파일(예: `data/raw/train.csv`)은 Git에 커밋하면 절대 안 됩니다. `.gitignore`에 원본 파일이 확실히 차단되어 있는지 확인하고, 오직 DVC가 생성한 메타데이터 파일(`train.csv.dvc`)만 Git으로 `commit` 하세요.
- **Polars - Pandas 변환 오버헤드**: Scikit-Learn 등 타 라이브러리는 Polars 데이터프레임을 바로 받지 못할 수 있습니다. 전처리의 맨 마지막 단계에서만 `.to_pandas()` 또는 `.to_numpy()`로 변환하여 메모리 스왑 오버헤드를 최소화하세요.
