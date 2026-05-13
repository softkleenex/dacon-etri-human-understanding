# 클라우드 환경 (Kaggle Notebooks) 활용 매뉴얼

로컬 머신(Apple Silicon)에서 감당하기 힘든 무거운 딥러닝 모델 학습이나, 장시간의 K-Fold 교차 검증을 무료 클라우드 자원으로 "오프로딩(Off-loading)"하기 위해 도입한 파이프라인입니다.

## 🎯 사용 목적 (Why use this?)
- **무료 GPU/TPU 확보**: 주당 30시간 제공되는 T4 x2/P100 GPU를 활용하여 딥러닝(MultiTaskLSTM) 모델을 훈련시킵니다.
- **백그라운드 병렬 작업**: 로컬에서는 코드 작성과 가벼운 실험(EDA)을 계속 진행하고, 무거운 학습은 클라우드 서버에 던져두어 리소스를 완벽히 분리합니다.

## 🚀 사용 방법 (How to use?)

### 1. 코드 배포 및 훈련 스니펫 (Make 기반)
로컬 코드를 데이터셋으로 묶어 클라우드로 전송하고, 훈련을 트리거하는 파이프라인입니다. 기존 복잡한 명령어 대신 Makefile 명령어를 활용하세요.

```bash
# 1. 로컬 코드를 Kaggle Dataset으로 패키징 및 업데이트 (Push)
# 내부에 src/, conf/, scripts/, pyproject.toml 등을 모아서 자동으로 푸시합니다.
# 주의: 이 명령어를 실행하면 `pre_flight_check.py`가 자동으로 가동되어 로컬 설정의 무결성(GPU 쿼터, 경로 등)을 검증합니다.
make kaggle-push-code

# 2. 백그라운드 훈련 커널 실행 트리거
# (마찬가지로 이륙 전 점검을 수행한 뒤 커널을 띄웁니다)
make kaggle-push-notebook

# 3. 훈련 결과물 대기 및 무인 자동 제출 봇 가동
# 이 명령을 백그라운드로 실행해두면, Kaggle 커널이 종료될 때까지 1분마다 모니터링하다가
# 완료 즉시 결과 CSV를 당겨오고, DACON API를 통해 리더보드에 자동 제출합니다.
python scripts/auto_wait_submit.py &
```

### 2. 동적 경로 탐색 (Auto-Discovery)
Kaggle의 데이터셋 구조는 업로드 방식에 따라 폴더 트리가 평탄화(Flatten)되거나 예측할 수 없게 변할 수 있습니다. 이를 방지하기 위해 로컬 코드베이스는 다음과 같은 **동적 탐색** 로직을 포함합니다.
- `src/data/data_loader.py`: `data/raw` 폴더의 존재 유무를 파악하여 동적으로 raw 데이터 경로를 설정합니다.
- `notebooks/generate_notebooks.py`: `/kaggle/input` 내부를 순회(`rglob`)하며 `conf/config.yaml`의 위치를 스스로 찾아내어 환경 변수에 주입합니다. 하드코딩된 경로는 더 이상 사용하지 않습니다.

### 3. 의존성 자동 설치
노트북 런타임 환경 동기화의 번거로움을 피하기 위해 `generate_notebooks.py` 스크립트가 Kaggle 노트북 생성 시 다음과 같은 의존성 설치 구문을 자동으로 최상단에 주입합니다.
```python
!pip install -q optuna pytorch-lightning omegaconf catboost
```

---

## 🛑 Kaggle 트러블슈팅 체크리스트 (Troubleshooting Log)

과거 Dacon 대회 연동 과정에서 발생했던 치명적인 에러와 그 해결책 모음입니다.

### ❌ 실수 1: 로컬과 Kaggle의 폴더 구조 차이 간과 (`FileNotFoundError`)
- **문제**: 로컬에서는 `data/raw/` 경로에 데이터가 있었지만, Kaggle Dataset으로 업로드 시 루트 디렉토리가 평탄화되면서 에러 발생.
- **해결책**: 위 2번 항목에 명시된 `Auto-Discovery` 로직으로 코드를 리팩토링하여 원천 차단함.

### ❌ 실수 2: Python 3.12 호환성 에러 (Jupyter Magic)
- **문제**: 로컬 버릇대로 노트북 맨 위에 `%load_ext autoreload` 와 `%autoreload 2`를 넣었으나, 현재 Kaggle의 Python 3.12 환경에서는 `imp` 모듈이 삭제되어 `ModuleNotFoundError`를 내며 커널 강제 종료.
- **해결책**: `generate_notebooks.py`에서 `autoreload` 매직 커맨드 삽입 로직을 완전히 제거함.

### ❌ 실수 3: 맹목적인 Kaggle Push (GPU Quota 낭비)
- **문제**: 오타 하나, 패키지 누락 하나 때문에 Kaggle 커널이 10분 뒤에 죽거나, `tune_hyperparameters=True`를 켜놓고 수천 개의 모델을 돌려 쿼터(할당량)를 낭비함.
- **해결책**: **이륙 전 점검(Pre-Flight Check) 스크립트 강제화.** `make kaggle-push-*` 명령어 실행 전 `scripts/pre_flight_check.py`가 가동되어 과도한 모델 학습이나 설정 오류를 사전에 차단함.

### ❌ 실수 4: Kaggle P100 GPU 와 PyTorch LSTM 충돌 (`CUDA error: no kernel image is available`)
- **문제**: Kaggle에 할당된 P100 GPU(Pascal 아키텍처)가 최신 PyTorch의 `cuDNN` RNN 평탄화(Flattening) 로직을 처리하지 못해 훈련 시작 직전 커널이 뻗어버리는 호환성 버그 발생.
- **해결책**: `generate_notebooks.py` 최상단에 `torch.backends.cudnn.enabled = False` 구문을 명시적으로 강제 주입하여 CUDA 호환성 에러를 원천 차단함.