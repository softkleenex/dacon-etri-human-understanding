# 클라우드 환경 (Kaggle Notebooks) 활용 매뉴얼

로컬 머신(Apple Silicon)에서 감당하기 힘든 무거운 딥러닝 모델 학습이나, 장시간의 K-Fold 교차 검증을 무료 클라우드 자원으로 "오프로딩(Off-loading)"하기 위해 도입한 파이프라인입니다.

## 🎯 사용 목적 (Why use this?)
- **무료 GPU/TPU 확보**: 주당 30시간 제공되는 T4 x2 GPU를 활용하여 거대한 트랜스포머 모델 등을 훈련시킵니다.
- **백그라운드 병렬 작업**: 로컬에서는 코드 작성과 가벼운 실험(EDA)을 계속 진행하고, 무거운 학습은 클라우드 서버에 던져두어(Save & Run All) 리소스를 완벽히 분리합니다.

## 🚀 사용 방법 (How to use?)

### 1. 인증 및 환경 변수 설정
Kaggle CLI를 통해 클라우드와 통신하기 위해 `kaggle.json` 인증 파일이 필요합니다. (이 파일은 보안상 `.gitignore`에 의해 커밋되지 않습니다.)

```bash
# Kaggle 홈페이지 -> Settings -> Create New Token에서 다운받은 파일을 홈 디렉토리로 이동
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. 코드 배포 및 훈련 스니펫 (CLI)
로컬 코드를 데이터셋으로 묶어 클라우드로 전송하고, 훈련을 트리거하는 파이프라인입니다.

```bash
# 1. 로컬 코드를 Kaggle Dataset으로 업데이트 (Push)
kaggle datasets version -p ./src -m "Update model architecture"

# 2. 백그라운드 훈련 커널 실행 트리거
kaggle kernels push -p ./kaggle_runner_folder

# 3. 훈련 종료 후 결과물(Weights, CSV) 로컬로 다운로드
kaggle kernels output softkleenex/my-dacon-training-kernel -p ./models
```

## ⚠️ 실전 주의 사항
- **Kaggle Secrets 활용**: 클라우드 노트북 코드 내부에 `WANDB_API_KEY`나 `HF_TOKEN`을 평문으로 하드코딩하지 마세요. Kaggle 웹사이트의 **Add-ons -> Secrets** 메뉴에 토큰을 등록하고 `kaggle_secrets` 패키지로 런타임에 주입받아야 계정 해킹을 방지할 수 있습니다.
- **OOM(Out of Memory) 방지**: 클라우드 T4 GPU는 16GB VRAM을 가지므로 로컬(24GB+)보다 메모리가 작습니다. PyTorch Lightning의 `precision="16-mixed"`를 활용하여 메모리를 절약하세요.
