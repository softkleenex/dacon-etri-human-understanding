# Kaggle CLI (`kaggle`) 활용 가이드

DACON 제출 및 클라우드 학습 워크플로우를 자동화하는 핵심 도구입니다.

## 설정 방법
1. Kaggle 계정에서 API Token 발급 (`kaggle.json`)
2. `~/.kaggle/kaggle.json` 위치에 배치하고 권한 설정 (`chmod 600 ~/.kaggle/kaggle.json`)

## 파이프라인 연동 기능 [적용 완료]
- **자동 학습 트리거**: `kaggle kernels push` 명령어로 로컬 스크립트를 Kaggle Notebook 환경에 백그라운드 워커로 밀어넣습니다.
- **결과물 다운로드**: 학습이 완료된 후 `kaggle kernels output`으로 서브미션 파일과 메트릭을 자동 로드합니다.
- **데이터셋 동기화**: 프로젝트 코드(`src/`)를 모델의 의존성으로 연결하기 위해 Kaggle Dataset으로 자동 업데이트(`kaggle datasets version`)합니다.

관련 자동화 코드는 `scripts/orchestrate_kaggle.py`에 작성되어 있습니다.
