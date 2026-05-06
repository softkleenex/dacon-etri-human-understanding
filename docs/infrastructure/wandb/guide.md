# Weights & Biases (`wandb`) 가이드

Weights & Biases(W&B)는 수많은 실험의 손실(Loss), 평가 지표(Metrics), 하이퍼파라미터를 대시보드에 시각화하고 실험 결과를 비교/분석하기 위한 필수 MLOps 도구입니다.

---

## 1. 인증 및 환경 변수 설정
이 템플릿 환경에서는 **시스템 전역(`~/.zshrc` 및 `~/.netrc`)에 인증 토큰이 미리 저장**되어 있습니다. 따라서 파이썬 스크립트 실행 시 매번 토큰을 입력하거나 소스 코드에 하드코딩할 필요가 없습니다.

- **전역 변수 확인**: 터미널에서 `echo $WANDB_API_KEY`
- **로그인 상태 확인**: 터미널에서 `wandb login` 입력 시 자동 로그인 완료 메시지가 표시됩니다.

---

## 2. Python 모델 학습 스크립트 연동
학습 코드(`scripts/train.py`) 최상단에서 `wandb.init()`을 호출하여 하나의 Run(실험)을 시작하고, 루프 안에서 `wandb.log()`로 지표를 전송합니다.

```python
import wandb
import os

def train_model(config):
    # 1. W&B 실험 초기화
    wandb.init(
        project=os.getenv("DACON_TEAMNAME", "dacon-competition"), # 프로젝트명
        name="baseline_lgbm_v1",                                  # 실험(Run) 이름
        config=config                                             # 하이퍼파라미터 기록
    )
    
    # 2. 모델 학습 및 평가
    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(...)
        val_score = validate(...)
        
        # 3. W&B 대시보드로 실시간 지표 전송
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_score": val_score,
            "learning_rate": current_lr
        })
        
    # 4. 종료 선언 (스크립트 종료 시 자동 호출되지만 명시적 권장)
    wandb.finish()

if __name__ == "__main__":
    hyperparams = {"epochs": 100, "learning_rate": 0.05, "max_depth": 7}
    train_model(hyperparams)
```

---

## 3. Kaggle 커널(클라우드) 연동 시 주의점
Kaggle Notebook에서 코드를 실행할 때 `WANDB_API_KEY`가 평문으로 노출되는 것을 막아야 합니다.
1. Kaggle 홈페이지 상단 메뉴의 **Add-ons -> Secrets**에 `WANDB_API_KEY`를 등록합니다.
2. 클라우드에서 돌아가는 훈련 스크립트 최상단에 다음과 같은 코드를 추가하여 안전하게 토큰을 가져옵니다.

```python
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    wandb_key = user_secrets.get_secret("WANDB_API_KEY")
    wandb.login(key=wandb_key)
except ImportError:
    # 로컬 환경에서는 이미 ~/.netrc 에 의해 로그인되어 있음
    pass
```

---

## 4. 고급 기능: W&B Sweeps 연동
Optuna를 대신하여 W&B 자체 기능인 **Sweeps**를 사용하면, 어떤 하이퍼파라미터 조합이 성능에 가장 큰 영향을 미쳤는지(Parameter Importance) 대시보드에서 강력한 시각화 툴로 자동 분석해줍니다. K-Fold 평균 점수를 Sweep의 목적 함수(Objective Function) 타겟으로 설정하여 시도해보는 것을 권장합니다.
