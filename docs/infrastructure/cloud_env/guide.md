# 클라우드 환경 (Cloud Environment - Kaggle Notebooks) 가이드

로컬 Apple Silicon 머신에서 처리하기 버거운 무거운 딥러닝 모델 학습이나, 장시간의 K-Fold 병렬 학습을 수행하기 위해 무료 클라우드 자원을 활용하는 가이드입니다.

## 1. Kaggle 자원 활용의 장점
- **무료 GPU/TPU 제공**: 주당 30시간의 T4 x2(또는 P100) GPU를 활용하여 딥러닝 모델을 병렬로 훈련할 수 있습니다.
- **백그라운드 워커 분리**: 로컬 머신의 자원을 점유하지 않고 클라우드 상에서 "Save & Run All" 기능을 통해 백그라운드 훈련을 던져두고, 로컬에서는 다른 코드 작업에 집중할 수 있습니다.

## 2. 작업 동기화 워크플로우
클라우드 커널 상에서 직접 코드를 짜는 것은 매우 비효율적입니다. 로컬에서 코딩 후 클라우드로 쏘아 보내는(Push) 파이프라인을 구축해야 합니다.

1. **로컬 개발**: `src/`와 `scripts/`에 파이썬 모듈과 훈련 스크립트를 작성합니다.
2. **Kaggle Dataset화**: 현재 프로젝트의 코드(`src/`) 자체를 압축하여 하나의 Kaggle Dataset으로 만듭니다. (Kaggle CLI 사용)
3. **Kaggle Notebook 실행**: 
   - 훈련 전용 노트북(`.ipynb` 또는 `kernel-metadata.json`)을 구성하여 방금 만든 코드 데이터셋을 마운트(Mount)합니다.
   - 노트북 내에서 `!python /kaggle/input/my-code-dataset/scripts/train.py`를 실행하도록 세팅합니다.
4. **결과 다운로드**: 백그라운드 훈련이 끝나면 W&B 대시보드로 로그를 확인하고, `kaggle kernels output` 명령어로 가중치 파일(`.pt`)과 제출용 `submission.csv` 파일을 다운로드 받습니다.

## 3. 주의 사항
- **세션 종료 방지**: Kaggle 커널은 일정 시간 출력이 없거나 브라우저 창이 닫히면 끊길 수 있습니다. 반드시 "Save & Run All (Commit)" 기능을 통해 백그라운드 서버에서 독립적으로 돌도록 해야 합니다.
- **Secret 관리**: Kaggle Notebook 내에 하드코딩된 토큰(W&B 토큰 등)이 퍼블릭으로 공개되지 않도록, Kaggle Secret 설정을 적극 활용하세요.
