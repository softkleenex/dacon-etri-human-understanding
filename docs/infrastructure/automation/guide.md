# 파이프라인 자동화 (Automation) 매뉴얼

DACON 환경의 특성(하루 제출 횟수 제한, 평가 점수 민감도)과 Kaggle 클라우드 환경(GPU 시간 제한, 백그라운드 구동)을 완벽하게 이어주기 위해 설계된 자체 자동화 스크립트 가이드입니다.

## 🚀 1. 이륙 전 점검 봇 (`pre_flight_check.py`)
캐글의 주간 GPU 할당량(Quota)을 허무하게 날리거나 커널이 터지는 것을 막기 위해 동작하는 최우선 방어막입니다.
`Makefile`의 `kaggle-push-code` 와 `kaggle-push-notebook` 명령어가 실행되기 직전에 **자동으로 가동**됩니다.

**주요 점검 로직**:
- **Quota Safety Check**: `tune_hyperparameters=True`이고 폴드 수와 시도 횟수를 곱한 총 모델 훈련 개수가 500개를 넘어가면 GPU 시간을 다 먹는다고 판단하여 즉시 Push를 취소합니다.
- **Metric Check**: `config.yaml`의 평가 지표가 `logloss`가 아닌 `rmse` 등으로 잘못 설정되어 있으면 차단합니다.
- **Dependency, Path & GPU Check**: 노트북 생성 로직에 패키지 설치(`!pip install`)가 있는지, 데이터 로더에 `Auto-discovery`(동적 경로) 로직이 살아있는지, 그리고 P100 GPU 버그 픽스(`torch.backends.cudnn.enabled = False`)가 존재하는지 검사합니다.

## 🤖 2. 무인 자동 제출 봇 (`auto_wait_submit.py`)
Kaggle 커널이 백그라운드에서 돌아가는 수 시간 동안 대기하는 수고를 덜어주는 무인 자동 제출기입니다.
**이제 `make kaggle-push-notebook` 명령어를 실행하면 캐글 노트북 푸시 직후 터미널 백그라운드에 자동으로 띄워집니다.** 로그는 `submissions/auto_wait_submit.log`에 기록됩니다.

**주요 점검 로직**:
- **Kaggle API 감시 & Retry**: 1분 주기로 커널 상태(`KernelWorkerStatus.RUNNING`)를 감시합니다. Kaggle 서버 오류로 일시적으로 상태를 못 가져오더라도 최대 3번(Backoff Retry) 재시도합니다.
- **최신 CSV 자동 추적**: 하드코딩된 이름이 아니라 `os.path.getmtime`을 활용해 `submissions/` 폴더 내에 방금 훈련이 끝나서 다운로드 된 가장 최신의 CSV를 찾아냅니다. (이때 `log`나 `ensemble` 파일은 자동으로 배제합니다).
- **DACON 무인 제출**: Dacon API를 사용해 타겟 대회(`236690`)에 파일을 쏘고, 이 기록을 로컬 `submissions/submission_log.csv`에 자동으로 누적 저장합니다.

## 🤝 3. 지능형 앙상블 스크립트 (`ensemble.py`)
여러 번의 실험(CatBoost, LSTM 등)에서 얻어진 결과물들을 섞어 점수를 안정화(Soft Voting)하는 스크립트입니다.
단순히 모든 파일을 합치는 것이 아니라 **"가장 점수가 좋은 파일만" 자동으로 골라냅니다**.
`make ensemble` 로 가동합니다.

**주요 점검 로직**:
- **Score 파싱**: 파일명에서 `score0.5843` 과 같은 패턴을 정규식(Regex)으로 파싱하여 점수가 낮은(좋은) 순으로 정렬합니다.
- **Top-K 선택**: 형편없는 점수가 섞여 들어가는 것을 막기 위해 상위 3개(`--top_k=3`)의 파일만 선별합니다.
- **무한대 페널티 클리핑**: 합산된 평균 점수가 무한대 페널티를 받지 않도록 `np.clip(df, 1e-15, 1 - 1e-15)` 안전장치를 거친 뒤 `submission_ensemble_latest.csv`로 저장합니다.