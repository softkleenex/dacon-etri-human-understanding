# Gemini CLI (AI Agent) 가이드

Gemini CLI는 단순한 코드 어시스턴트가 아니라, **프로젝트 전체의 MLOps 파이프라인과 코드베이스를 자율적으로 진화시키는 메인 컨트롤 타워(Master Brain)**입니다.

## 역할 및 책임 (Role & Responsibilities)
1. **가설 수립 및 연구 (Research & Hypothesis)**: Hugging Face 도구 및 웹 검색을 통해 궤적 예측, 물리 기반 신경망(PINN), 시계열 분석 관련 최신 논문을 스스로 탐색하고 가설을 세웁니다.
2. **코드 자율 진화 (Self-Evolving Code)**: 수립한 가설을 바탕으로 `src/features.py`나 `src/models.py`의 코드를 직접 리팩토링 및 고도화합니다.
3. **실험 오케스트레이션 (Experiment Orchestration)**: 수정된 코드를 로컬 환경에서 테스트한 뒤, `make run-kaggle` 명령을 통해 Kaggle 백그라운드 워커에 분산 학습을 지시합니다.
4. **결과 분석 및 피드백 (Analysis & Feedback)**: `experiment_history.json`과 W&B 로그를 분석하여 성능 향상 여부를 판단하고, 실패 시 코드를 롤백하거나 새로운 접근법을 시도합니다.

## 주요 기능 및 명령어
- `/ralph:loop [명령어]`: 특정 목표(예: "모델 성능이 0.62를 넘을 때까지 논문을 검색하고 피처를 추가하여 캐글 학습을 반복해라")를 달성할 때까지 에이전트가 스스로 무한 반복하며 작업하는 자율 에이전틱 워크플로우를 가동합니다.
- `코드베이스 수술`: 에이전트는 전체 파일을 출력하지 않고 `replace` 도구를 사용해 외과 수술처럼 필요한 부분만 빠르고 정확하게 수정합니다.
- `서브 에이전트 위임`: 반복적인 로그 확인이나 대규모 코드 조사는 `generalist`, `codebase_investigator` 등의 서브 에이전트에게 위임하여 메인 컨텍스트를 아낍니다.

## 사용 예시 (Agentic Workflow Trigger)
```bash
/ralph:loop 현재 OOF 점수를 분석하고, 점수가 오르지 않는다면 Hugging Face에서 궤적 분석 논문을 찾아 새로운 피처를 추가한 뒤, 캐글에 제출하고 결과를 확인해. 목표 점수를 달성할 때까지 이 과정을 스스로 반복해.
```