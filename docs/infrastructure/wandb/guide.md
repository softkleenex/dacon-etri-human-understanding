# Weights & Biases (`wandb`) 가이드

Weights & Biases(W&B)는 수많은 백그라운드 실험의 학습 추이, 손실(Loss), 평가 지표 및 하이퍼파라미터를 대시보드에서 시각화하고 비교/분석하기 위한 필수 MLOps 도구입니다.

## 1. 인증 및 환경 변수 설정
이 베이스라인 템플릿 환경에서는 **시스템 전역(`~/.zshrc` 및 `~/.netrc`)에 인증 토큰이 미리 저장**되어 있습니다. 따라서 파이썬 스크립트 실행 시 매번 토큰을 입력하거나 코드에 하드코딩할 필요가 없습니다.

- **전역 토큰 변수**: `WANDB_API_KEY`
- **로그인 상태 확인**: 터미널에서 `wandb login` 입력 시 이미 로그인되어 있다고 표시됩니다.

## 2. 기본 활용법 (Python Script)
모델 학습 스크립트(`scripts/train.py`) 상단에서 `wandb.init()`을 호출하여 실험 로깅을 시작합니다.

```python
import wandb

# 실험 초기화
wandb.init(
    project="dacon-my-competition", # 대회명
    entity="softkleenex",           # 사용자 또는 팀명
    name="exp-001-baseline",        # 실험(Run) 이름
    config={
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32,
        "architecture": "LightGBM"
    }
)

# 학습 루프 내에서 지표 기록
for epoch in range(epochs):
    loss = train(...)
    val_score = validate(...)
    
    # wandb에 기록
    wandb.log({"train_loss": loss, "val_score": val_score, "epoch": epoch})

wandb.finish()
```

## 3. Kaggle 커널/클라우드 환경 연동
로컬 스크립트를 Kaggle로 Push하여 백그라운드에서 학습할 때도(예: `orchestrate_kaggle.py`), Kaggle Secrets 기능 또는 환경 변수 주입을 통해 W&B 토큰을 전달합니다. 이를 통해 클라우드에서 실행되는 모델의 실시간 학습 곡선을 로컬 노트북이나 웹 브라우저에서 실시간으로 모니터링할 수 있습니다.

## 4. Optuna & W&B Sweeps
하이퍼파라미터 탐색 시 Optuna를 단독으로 사용할 수도 있지만, W&B Sweeps 기능을 활용하면 탐색 공간을 YAML 파일로 관리하고 시각화할 수 있습니다. K-Fold 검증 시 폴드별 평균 성능을 최적화의 목적으로 두고 대시보드를 구축하는 것을 권장합니다.
