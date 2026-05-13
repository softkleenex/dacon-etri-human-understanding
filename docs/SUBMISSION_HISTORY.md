# 🏆 제출 및 실험 기록부 (Submission & Experiment History)

실제 제출 파일과 K-Fold 로컬 검증 점수(CV Score), DACON 리더보드 점수(Public LB Score) 간의 갭(Gap)을 추적하는 공간입니다.

> **💡 핵심 변경 사항 (CSV 트래킹 도입)**
> 로컬 검증(CV) 점수와 실제 제출(LB) 점수를 명확히 구분하고, 파이썬 스크립트가 결과를 자동으로 기록(Append)할 수 있도록 **마크다운 표 대신 `submissions/submission_log.csv` 파일을 사용**하여 데이터를 추적합니다.

## 📈 리더보드 및 로컬 점수 추적 (CSV Log)
모든 실험 결과는 👉 **[`submissions/submission_log.csv`](../submissions/submission_log.csv)** 에 기록됩니다.

### 📝 CSV 컬럼 명세
- `id`: 실험 고유 번호
- `date`: 추론/제출 날짜
- `submission_file`: 생성된 CSV 파일명 (예: `xgb_fold5.csv`)
- `model_description`: 적용한 모델 및 주요 피처 변경점
- **`local_cv_score` (나 스스로 판단한 점수)**: K-Fold 교차 검증 등을 통해 로컬 머신에서 계산된 점수. 모델 훈련/추론 스크립트(`inference.py`)가 자동으로 이 값을 채워 넣습니다.
- **`public_lb_score` (진짜 제출해서 얻은 점수)**: DACON 웹사이트에 실제로 제출(API 또는 수동)한 뒤 얻은 점수. 제출 전에는 비워두고(Null), 제출 후에 사용자가 점수를 기입합니다.
- `is_submitted`: 실제 DACON에 제출했는지 여부 (True/False). 하루 제출 횟수 제한(Quota) 관리를 위해 사용합니다.
- `notes`: 에이전트나 사용자가 남기는 특이사항.

---

## 🔬 실험 기록 템플릿 (Experiment Tracking Template)
새로운 실험을 진행할 때 아래 템플릿을 복사하여 기록하세요.

### 🧪 실험 명: [실험 이름 또는 주제] (관련 CSV id: [id])
- **가설 (Hypothesis)**: [무엇을 변경하면 어떤 효과가 있을 것이라고 예상하는지 기술]
- **결과 (Result)**: [로컬 CV 및 Public LB 점수 변화, 학습 시간 등 주요 결과]
- **다음 단계 (Next Step)**: [결과를 바탕으로 얻은 통찰 및 다음으로 시도할 작업]

---

## 🛑 튜닝 및 실패 기록 (What Did NOT Work)
(CSV에 담기 힘든 길고 상세한 실패 원인 분석이나 통찰(Insight)은 이 마크다운 문서에 텍스트로 기록합니다.)

- **시도 1**: LightGBM Baseline (CSV id: 1)
  - **결과**: 로컬 CV 0.6022, Public LB 13.3222 (226등)
  - **이유 추정**: DACON 평가 지표가 Log-Loss이므로 점수가 낮을수록 순위가 높음. 현재 모델은 Macro F1 기준으로 최적화되어 있어 Log-Loss 점수가 매우 높게 나옴.
  - **통찰**: 다음 실험부터는 모델의 평가 및 최적화 지표를 Log-Loss로 변경해야 함.

