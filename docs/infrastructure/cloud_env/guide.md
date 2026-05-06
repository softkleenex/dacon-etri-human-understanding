# 클라우드 환경 (Cloud Environment - Kaggle Notebooks) 가이드

로컬 머신의 자원 한계를 극복하고 모델을 본격적으로 학습시키는 메인 컴퓨팅 환경입니다.

## 하드웨어 스펙
- **GPU**: Kaggle Free GPU (T4 x2, P100)
- **TPU**: Kaggle Free TPU v3-8

## 주요 용도 및 역할
1. **무거운 딥러닝 모델 학습**: Transformer, LSTM, Neural ODE 등 대용량 연산이 필요한 모델 훈련.
2. **K-Fold 교차 검증 (Cross Validation)**: 시간이 오래 걸리는 K-Fold 방식을 클라우드 분산 리소스로 해결.
3. **대규모 하이퍼파라미터 튜닝**: Optuna 등을 활용한 수십~수백 번의 반복 실험.

## 작업 방식 (Workflow)
- 로컬(`src/` 디렉토리)에서 검증된 스크립트를 작성합니다.
- `scripts/push_code.sh`를 통해 Kaggle Dataset에 코드를 배포합니다.
- `scripts/orchestrate_kaggle.py` 스크립트를 통해 다수의 백그라운드 워커를 생성하여 파이프라인을 자동화합니다.
