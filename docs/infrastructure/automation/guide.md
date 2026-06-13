# 파이프라인 자동화 (Automation) 매뉴얼

DACON 환경의 특성(하루 제출 횟수 제한, 평가 점수 민감도)과 Kaggle 클라우드 환경(GPU 시간 제한, 백그라운드 구동)을 완벽하게 이어주기 위해 설계된 자체 자동화 스크립트 가이드입니다.

## 🚀 1. 이륙 전 점검 봇 (`pre_flight_check.py`)
캐글의 주간 GPU 할당량(Quota)을 허무하게 날리거나 커널이 터지는 것을 막기 위해 동작하는 최우선 방어막입니다.
`Makefile`의 `kaggle-push-code` 와 `kaggle-push-notebook` 명령어가 실행되기 직전에 **자동으로 가동**됩니다.

**주요 점검 로직**:
- **Quota Safety Check**: `tune_hyperparameters=True`이고 폴드 수와 시도 횟수를 곱한 총 모델 훈련 개수가 500개를 넘어가면 GPU 시간을 다 먹는다고 판단하여 즉시 Push를 취소합니다.
- **Metric Check**: `config.yaml`의 평가 지표가 `logloss`가 아닌 `rmse` 등으로 잘못 설정되어 있으면 차단합니다.
- **Path, Metadata & Syntax Check**: 데이터 로더의 Kaggle flat-directory fallback, kernel metadata, config, compute quota, Python syntax를 검사합니다.

## 🤖 2. 결과 회수 및 명시적 제출
자동 제출은 기록 무결성을 해치기 쉬우므로 기본 워크플로우에서 제거했습니다. Kaggle 실행 후 `make kaggle-wait-and-pull`로 결과를 받고, `make normalize-submissions`로 기록을 정규화한 뒤, 검토한 파일만 `make dacon-submit`으로 제출합니다.

**주요 점검 로직**:
- **Kaggle Output Pull**: `make kaggle-wait-and-pull`이 커널 종료 후 결과물을 `submissions/`로 받습니다.
- **Submission Normalization**: `normalize_submission_log.py`가 `status_clean`, `risk_reason`, `safe_for_default_ensemble`을 갱신합니다.
- **Deliberate DACON Submit**: `dacon-submit`은 `SUB_PATH`와 `SUB_MEMO`를 명시해 실행합니다.

## 🤝 3. 지능형 앙상블 스크립트 (`ensemble.py`)
여러 번의 실험(CatBoost, LSTM 등)에서 얻어진 결과물들을 섞어 점수를 안정화(Soft Voting)하는 스크립트입니다.
단순히 모든 파일을 합치는 것이 아니라 **public sanity가 확인되고 blacklist에 없는 안전 후보만** 자동으로 골라냅니다.
`make ensemble` 로 가동합니다.

**주요 점검 로직**:
- **Submission Log 기반 필터링**: `submission_log.csv`와 `reports/submission_blacklist.json`을 읽어 placeholder, 미채점, CV-LB gap 위험 후보를 제외합니다.
- **Top-K 선택**: 기본 상위 5개(`--top-k=5`) 안전 후보만 선별합니다.
- **무한대 페널티 클리핑**: 합산된 평균 점수가 무한대 페널티를 받지 않도록 `np.clip(df, 1e-15, 1 - 1e-15)` 안전장치를 거친 뒤 `submission_ensemble_latest.csv`로 저장합니다.