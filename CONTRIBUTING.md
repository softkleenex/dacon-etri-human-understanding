# DACON ETRI Human Understanding 기여 가이드 (Contributing Guide)

이 프로젝트는 DACON ETRI Human Understanding AI Paper Challenge 제출 및 연구를 위한 워크스페이스입니다. 
팀원 및 협업자(AI 에이전트 포함)가 코드베이스를 일관되고 깨끗하게 유지하기 위한 규칙을 정의합니다.

## 1. 이슈 및 태스크 관리
- 새로운 아이디어, 버그 리포트, 데이터 탐색(EDA) 결과 등은 문서화하여 `docs/` 또는 `references/`에 기록하거나 이슈로 관리합니다.
- 실험 아이디어는 `docs/SUBMISSION_HISTORY.md`의 실험 기록 템플릿에 맞추어 관리합니다.

## 2. 브랜치 전략
- `main` 브랜치는 항상 동작 가능하고 안정적인 코드를 유지합니다.
- 새로운 피처(Feature) 개발이나 실험은 기능/실험 단위의 브랜치를 생성하여 작업합니다.
  - 예: `feat/add-lstm-model`, `exp/try-focal-loss`, `fix/data-loader-bug`

## 3. 코드 스타일 및 린트
이 프로젝트는 `ruff`를 통해 포맷팅 및 린팅을 관리합니다. PR을 올리기 전이나 코드를 커밋하기 전에 반드시 검사를 수행하세요.

```bash
make format  # 코드 포맷팅
make lint    # 코드 린팅
make test    # 유닛 테스트 실행
```

- 타입 힌팅(Type Hinting)을 적극적으로 사용합니다.
- 함수의 목적과 파라미터에 대한 Docstring을 작성합니다.

## 4. 데이터 및 모델 관리
- `data/`, `models/`, `.venv/` 등의 대용량 폴더는 형상 관리(Git)에 포함하지 않습니다 (`.gitignore` 설정 유지).
- 새로운 피처(파생 변수)를 만들었을 경우, `src/features/` 아래에 모듈화하여 저장하고, 추출 파이프라인을 재현 가능하게 구성합니다.

## 5. Pull Request (PR) 가이드
1. 브랜치를 생성하고 작업을 진행합니다.
2. `make format`, `make lint`, `make test`를 모두 통과하는지 확인합니다.
3. PR 본문에는 **변경 사항**, **해결한 문제**, **관련된 실험/이슈**를 명확히 작성합니다.
4. 필요시, Kaggle 커널에서의 테스트 결과나 로컬 CV 점수 변화를 첨부합니다.
