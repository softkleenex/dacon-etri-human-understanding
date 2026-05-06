# Weights & Biases (`wandb`) 활용 매뉴얼

수많은 실험의 손실(Loss), 평가 지표(Metrics), 하이퍼파라미터를 대시보드에 실시간으로 시각화하고 실험 결과를 비교/분석하기 위한 모니터링 도구입니다. (전역 토큰 로그인은 완료되어 있습니다.)

## 🎯 사용 목적 (Why use this?)
- **실험 추적의 자동화**: 내가 3일 전에 돌렸던 모델의 학습률(LR)이 몇이었고, 점수가 몇이었는지 엑셀에 적을 필요가 없습니다. 모든 히스토리가 클라우드 대시보드에 자동 영구 저장됩니다.
- **실시간 원격 모니터링**: Kaggle 클라우드 워커에서 백그라운드로 돌아가는 학습 곡선을 내 로컬 머신(크롬 브라우저)에서 실시간으로 지켜보며, 오버피팅(Overfitting) 조짐이 보이면 즉시 중단시킬 수 있습니다.

## 🚀 사용 방법 (How to use?)

### 1. 훈련 스크립트 내 지표 로깅
모델 학습 코드(`scripts/train.py`) 최상단에서 `wandb.init()`을 호출하고, 매 에폭마다 `wandb.log()`로 지표를 쏩니다.

```python
import wandb

def train_model(config):
    # 1. 실험 기록 시작
    wandb.init(
        project="dacon-my-competition", # 대시보드 프로젝트명
        name="baseline_lgbm_v1",        # 현재 실험의 이름
        config=config                   # 사용한 하이퍼파라미터 저장
    )
    
    # 2. 학습 루프
    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch()
        val_score = validate()
        
        # 3. 실시간 대시보드 전송
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_score": val_score,
        })
        
    wandb.finish()
```

### 2. PyTorch Lightning 연동
MLOps 프레임워크인 PyTorch Lightning을 쓰면, `wandb.log()`를 일일이 적을 필요 없이 단 1줄로 모든 지표와 모델 그래프까지 로깅됩니다.

```python
from lightning.pytorch.loggers import WandbLogger
import lightning as L

logger = WandbLogger(project="dacon-my-competition", name="transformer_v1")
trainer = L.Trainer(logger=logger, max_epochs=100)
trainer.fit(model) # 알아서 다 로깅됨
```

## ⚠️ 실전 주의 사항
수백 번의 튜닝(Optuna 등)을 돌릴 때 W&B 로깅을 켜두면, 대시보드에 쓰레기(Trash) 실험 로그가 수백 개 쌓이게 됩니다. 
최종 후보 모델을 길게 훈련시킬 때나 유의미한 구조 변경이 있을 때만 로깅을 켜고(enable), 단순 파라미터 탐색 시에는 `wandb.init(mode="disabled")`로 잠시 꺼두는 것이 깔끔한 대시보드 유지의 비결입니다.
