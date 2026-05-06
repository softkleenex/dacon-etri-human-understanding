# 클라우드 환경 (Kaggle Notebooks) 활용 매뉴얼

로컬 머신(Apple Silicon)에서 감당하기 힘든 무거운 딥러닝 모델 학습이나, 장시간의 K-Fold 교차 검증을 클라우드 자원으로 "오프로딩(Off-loading)"하기 위해 사용합니다.

## 🎯 사용 목적 (Why use this?)
- **무료 GPU/TPU 확보**: 주당 30시간 제공되는 T4 x2 GPU를 활용하여 모델 파라미터가 거대한 트랜스포머 등을 훈련시킵니다.
- **백그라운드 병렬 작업**: 로컬에서는 코드 작성과 가벼운 실험(EDA)을 계속 진행하고, 무거운 학습은 클라우드 서버에 던져두어(Save & Run All) 리소스를 분리합니다.

## 🚀 사용 방법 (How to use?)
설정은 이미 전역으로 완료되어 있습니다. 대회 중 코드를 클라우드로 배포하고 결과를 받는 핵심 파이프라인(Kaggle CLI) 사용법입니다.

### 1. 코드 업데이트 (로컬 -> 클라우드)
로컬에서 수정한 코드(`src/`, `scripts/`)를 클라우드 커널이 읽을 수 있도록 데이터셋 형태로 동기화합니다.
```bash
kaggle datasets version -p ./src -m "Add new Time-Series features"
```

### 2. 백그라운드 훈련 시작 (Run)
훈련용으로 세팅해둔 클라우드 노트북(커널)을 백그라운드로 실행시킵니다. 터미널을 종료해도 클라우드에서 계속 훈련이 진행됩니다.
```bash
kaggle kernels push -p ./kaggle_runner_folder
```

### 3. 상태 확인 및 결과물 다운로드 (Pull)
훈련 진행 상황은 W&B 대시보드로 실시간 확인하며, 클라우드 훈련이 끝난 후 최고 성능의 모델 가중치(`.pt`)와 `submission.csv`만 로컬로 가져옵니다.
```bash
# 훈련 진행 상태 확인
kaggle kernels status softkleenex/my-dacon-training-kernel

# 훈련이 완료된 결과물 다운로드
kaggle kernels output softkleenex/my-dacon-training-kernel -p ./models
```

## ⚠️ 실전 주의 사항
- 클라우드 환경(T4 GPU)은 16GB VRAM을 가지므로 로컬(Unified Memory 24GB+)보다 메모리가 작을 수 있습니다. PyTorch Lightning의 `precision="16-mixed"`를 활용해 OOM(Out of Memory)을 방지하세요.
- 클라우드 노트북 코드 내부에 W&B나 HuggingFace 토큰을 평문으로 하드코딩하지 말고, Kaggle의 **Secrets 기능**을 사용해 환경변수로 주입받도록 구성해야 안전합니다.
