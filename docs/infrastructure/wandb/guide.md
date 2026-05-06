# Weights & Biases (`wandb`) 활용 매뉴얼

수많은 실험의 손실(Loss), 평가 지표(Metrics), 하이퍼파라미터를 대시보드에 실시간으로 시각화하고 실험 결과를 비교/분석하기 위해 도입한 필수 MLOps 모니터링 도구입니다.

## 🎯 사용 목적 (Why use this?)
- **실험 추적의 자동화**: 파라미터와 점수 이력을 엑셀에 적을 필요 없이 클라우드 대시보드에 자동 영구 저장됩니다.
- **실시간 원격 모니터링**: Kaggle 클라우드 워커에서 백그라운드로 돌아가는 학습 곡선을 내 크롬 브라우저에서 실시간 모니터링하여, 오버피팅(Overfitting) 조짐 시 즉각 대응할 수 있습니다.

## 🚀 사용 방법 (How to use?)

### 1. 인증 및 전역 환경 변수 설정
**이 템플릿 환경에서는 시스템 전역(`~/.zshrc` 및 `~/.netrc`)에 `WANDB_API_KEY` 인증 토큰이 미리 저장(Set Globally)되어 있습니다.**
따라서 파이썬 스크립트 실행 시 매번 토큰을 입력하거나 소스 코드에 절대 하드코딩할 필요가 없습니다.

```bash
# 전역 설정이 정상적으로 로드되었는지 확인하는 명령어
echo $WANDB_API_KEY
wandb login # (이미 로그인되어 있다고 표기됨)
```

### 2. 실전 파이썬 연동 스니펫
훈련 스크립트 최상단에서 `wandb.init()`을 호출하여 Run(실험)을 시작합니다.

```python
import wandb
import os

def train_model(config):
    # 전역 환경변수를 통해 별도 토큰 입력 없이 자동 연동됨
    wandb.init(
        project=os.getenv("DACON_TEAMNAME", "dacon-competition"),
        name="baseline_lgbm_v1",
        config=config
    )
    
    for epoch in range(config["epochs"]):
        train_loss, val_score = train_and_validate(...)
        
        # 실시간 대시보드 전송
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_score": val_score,
        })
        
    wandb.finish()
```

## ⚠️ 실전 주의 사항
- **Kaggle 클라우드 연동 주의**: 로컬이 아닌 Kaggle 클라우드 워커에서 실행할 때는 `~/.netrc`가 없으므로, Kaggle의 **Secrets 기능**에 `WANDB_API_KEY`를 넣고 코드 안에서 `kaggle_secrets` 모듈로 토큰을 꺼내와야 합니다.
- **로깅 끄기 (Disable)**: 수백 번의 Optuna 파라미터 튜닝 시 W&B 로깅을 그대로 켜두면 대시보드가 쓰레기 로그로 가득 찹니다. 단순 탐색 시에는 `wandb.init(mode="disabled")` 속성으로 잠시 꺼두세요.
