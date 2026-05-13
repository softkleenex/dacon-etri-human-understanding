# 데이터 전처리 및 MLOps 확장 라이브러리 매뉴얼

본 대회는 다중 타겟 시계열(Time-Series Multi-Task) 데이터의 특징을 띱니다. 데이터 누수(Data Leakage)를 원천 차단하고 훈련 속도 및 메모리 관리를 최적화하기 위해 도입한 데이터 처리 및 MLOps 방법론 매뉴얼입니다.

## 🎯 핵심 전처리 전략 (Data Leakage Prevention)
미래의 데이터가 과거 모델 학습에 스며들어 평가(Validation)를 오염시키는 것을 막기 위해 다음 두 가지 기법이 필수적으로 강제됩니다.

1. **Expanding Window Normalization (정규화)**:
   - 각 유저별(subject_id)로 피처를 Z-score 정규화할 때, 전체 기간의 평균을 사용(`transform('mean')`)하면 미래 시점의 정보가 누수됩니다.
   - **해결**: `expanding().mean()` 과 `expanding().std()` 를 사용하여 과거부터 현재까지 누적된 데이터만으로 평균과 표준편차를 구합니다.
   
2. **Forward-fill Imputation (결측치 보간)**:
   - 시계열에서 `limit_direction="both"` 로 보간하면 내일의 데이터가 오늘을 채우게 됩니다.
   - **해결**: 반드시 `limit_direction="forward"` 및 `ffill()` 을 사용하여 결측치는 직전 과거 데이터로만 채워지게 설계되었습니다.

3. **K-Fold Loop 내에서의 Global Scaling & KNN Imputation**:
   - `SimpleImputer`나 `StandardScaler`를 교차 검증 루프 밖에 두면 검증 셋(Validation Set)의 통계가 훈련에 반영됩니다.
   - **해결**: `dl_trainer.py` 구조와 같이 `GroupKFold` 분할 직후 훈련 인덱스(train_idx)에만 `fit()` 을 수행하도록 안전하게 격리되었습니다.

## 🚀 딥러닝 훈련 프레임워크 (PyTorch Lightning)
복잡한 시계열 데이터를 7가지 타겟으로 동시에 예측(MultiTaskLSTM)하기 위해 `PyTorch Lightning`을 사용합니다. 복잡한 훈련 반복문(for loop) 보일러플레이트를 제거하고 Multi-GPU 스케일링을 자동화합니다.

### 1. PyTorch Lightning MLOps 스니펫
훈련 루프를 분리하고 `Trainer`에 핵심 옵션들을 몰아넣습니다.
```python
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# val_logloss_mean 모니터링을 통한 자동 저장 및 조기 종료
checkpoint_callback = ModelCheckpoint(
    dirpath="models/fold_0",
    filename="best-checkpoint",
    save_top_k=1,
    monitor="val_logloss_mean",
    mode="min"
)

trainer = pl.Trainer(
    max_epochs=50,
    callbacks=[checkpoint_callback],
    logger=WandbLogger(project="dacon-etri"),
    accelerator="auto", 
    devices=1
)
trainer.fit(model, data_module)
```

## ⚠️ 실전 주의 사항
- **DVC (Data Version Control) 미사용**: 프로젝트 초기 기획과 달리 대규모 로컬 저장소 및 캐글 환경 동기화를 위해 DVC를 사용하지 않습니다. 데이터 원본 관리는 Kaggle Dataset 버전 관리를 통해 대체됩니다.
- **GroupKFold 강제**: 시계열 슬라이딩 윈도우(Sliding Window) 특성상 훈련 셋과 검증 셋에 동일한 유저가 포함되면 절대 안 됩니다. 반드시 `GroupKFold`를 사용하여 유저(`subject_id`) 단위로 완벽하게 쪼개야 합니다.