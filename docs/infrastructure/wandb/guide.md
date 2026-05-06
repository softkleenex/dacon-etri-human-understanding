# Weights & Biases (`wandb`) 가이드

수많은 백그라운드 실험의 학습 추이와 하이퍼파라미터를 시각화하고 관리하기 위한 MLOps 툴입니다.

## 설정 방법
1. W&B 계정 생성 및 로그인
2. `WANDB_API_KEY` 환경 변수를 `.env` 또는 `~/.netrc`에 등록

## 파이프라인 연동 기능 [적용 완료]
- **Kaggle 백그라운드 연동**: 로컬에서 `orchestrate_kaggle.py` 실행 시 W&B API 토큰이 안전하게 주입되어, Kaggle에서 실행되는 모든 워커의 로스가 W&B 대시보드에 실시간 기록됩니다.
- **실험 추적**: 어떤 파라미터(예: Learning Rate, Tree Depth 등) 조합이 가장 높은 Hit Rate를 달성했는지 대시보드 상에서 비교/분석합니다.
