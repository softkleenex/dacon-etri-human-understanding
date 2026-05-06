# 클라우드 환경 (Cloud Environment - Kaggle Notebooks) 가이드

로컬 머신(Apple Silicon)에서 감당하기 힘든 무거운 딥러닝 모델 학습이나, 장시간의 K-Fold 교차 검증을 수행하기 위해 무료 클라우드 자원을 100% 활용하는 "오프로딩(Off-loading)" 가이드입니다.

---

## 1. Kaggle 클라우드 자원의 강점
- **무료 GPU/TPU 제공**: 주당 30시간의 T4 x2(또는 P100) GPU 쿼터를 활용하여, 트랜스포머(Transformer) 등 거대한 파라미터를 가진 모델을 병렬로 훈련시킬 수 있습니다.
- **비동기 백그라운드 워커**: 로컬 머신은 스크립트 작성에만 집중하고, 연산은 클라우드의 "Save & Run All (Commit)" 기능을 통해 백그라운드 서버에 떠넘깁니다. 노트북을 꺼도 클라우드 서버는 훈련을 계속합니다.

---

## 2. 로컬-클라우드 동기화 워크플로우 (Pipeline)
웹 브라우저로 Kaggle 커널에 접속해서 코딩하는 것은 비효율적입니다. 로컬 터미널에서 스크립트를 작성하고 클라우드로 푸시하는 파이프라인을 구성합니다.

### Step 1. 로컬 개발 및 테스트
로컬의 `src/` (피처, 모델 모듈) 및 `scripts/` (훈련 스크립트) 폴더에서 코드를 작성하고, 로컬 환경에서 단 1에폭만 돌려 문법 에러(Syntax Error)나 OOM이 없는지 확인합니다.

### Step 2. Kaggle Dataset으로 코드 동기화
Kaggle CLI를 활용하여 내 로컬의 파이썬 코드 전체를 Kaggle Dataset으로 묶어서 업로드합니다.
```bash
# 로컬 소스 코드를 Dataset으로 업데이트
kaggle datasets version -p ./src -m "Update model architecture"
```

### Step 3. Kaggle 커널 백그라운드 실행 (Push)
미리 설정해둔 훈련용 노트북 템플릿(이 노트북 안에는 방금 푸시한 코드 데이터셋이 마운트되어 있음)을 클라우드로 전송하며 즉시 실행시킵니다.
```bash
kaggle kernels push -p ./kaggle_runner/
```
이때 커널은 `!python /kaggle/input/my-code-dataset/scripts/train.py` 와 같이 내가 짠 파이썬 스크립트를 백그라운드에서 실행하게 됩니다.

### Step 4. 실시간 모니터링 및 결과 수확
- 훈련 중간 지표는 **W&B 대시보드**를 통해 실시간으로 확인합니다.
- 학습이 완전히 종료되면, 결과물(모델 가중치 `.pt` 파일 및 `submission.csv`)을 로컬로 내려받습니다.
```bash
kaggle kernels output softkleenex/my-dacon-training-kernel -p ./models
```

---

## 3. 핵심 주의 사항
- **Secret 관리 철저**: Kaggle Notebook 소스 코드 내에 W&B API 키나 Hugging Face 토큰을 절대 평문(Hardcoding)으로 적지 마세요. 반드시 Kaggle 계정의 **Secrets 기능** 메뉴에 토큰을 등록하고, 코드 내에서는 `kaggle_secrets` 패키지를 통해 동적으로 주입받아 사용해야 보안이 유지됩니다.
- **Out of Memory (OOM) 방지**: 클라우드의 T4 GPU는 16GB VRAM을 가집니다. 배치 사이즈(Batch Size)를 너무 크게 잡지 않도록 유의하고, PyTorch Lightning의 `precision="16-mixed"` 설정을 켜서 메모리를 절약하세요.
