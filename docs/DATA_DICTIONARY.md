# 📊 데이터 명세서 (Data Dictionary)

대회에서 제공된 원본 데이터의 구조(Schema), 피처(Feature)의 의미, 그리고 탐색적 데이터 분석(EDA) 과정에서 발견한 특이사항을 기록하는 문서입니다.

> **💡 에이전트(Gemini CLI) 활용 팁**
> 에이전트가 피처 엔지니어링(`src/features.py`) 코드를 작성하기 전에, 데이터의 구조를 빠르게 파악할 수 있도록 사용자가 초기 EDA 후 이 문서를 채워두는 것이 좋습니다.

---

## 1. 파일 목록 (File Overview)
- `train.csv`: 모델 학습용 원본 데이터. (Row: N건, Column: M개)
- `test.csv`: 추론 및 제출용 데이터. (Row: N건)
- `sample_submission.csv`: 제출 파일 양식.

## 2. 컬럼 명세 (Column Description)

| 변수명 (Feature) | 데이터 타입 (Type) | 의미 (Description) | 특이사항 (Notes / EDA 발견점) |
| :--- | :---: | :--- | :--- |
| `id` | String | 고유 식별자 | 학습 시 제외(Drop) 필요 |
| `target` | Float | 예측해야 할 타겟 변수 (Y) | 분포가 한쪽으로 치우침(Skewed), Log 변환 필요 |
| `feature_1` | Integer | (예시) 일일 강수량 | 결측치(NaN) 다수 존재 (평균값 대체 보류) |
| `feature_2` | Category | (예시) 지역 코드 | 고유값 15개, One-Hot Encoding 고려 |

---

## 3. 결측치 및 이상치 (Missing Values & Outliers)
- **결측치**: `feature_1` 컬럼에 약 15% 결측치 존재. K-NN Imputer 또는 평균값 대체 고려.
- **이상치**: `feature_3` 변수 중 상위 1% 값이 비정상적으로 높음. 99% 윈조라이징(Winsorization) 또는 제거 테스트 필요.

## 4. 파생 변수 아이디어 (Feature Engineering Ideas)
(대회를 진행하며 떠오른 아이디어를 자유롭게 메모하세요.)
- [ ] 아이디어 1: `feature_1`과 `feature_2`를 곱하여 교호작용(Interaction) 변수 생성.
- [ ] 아이디어 2: 날짜(Date) 컬럼에서 `요일`, `주말 여부`, `공휴일 여부` 추출.
- [ ] 아이디어 3: AI Hub 기상 데이터와 `지역 코드`를 기준으로 병합(Merge)하여 `일평균_습도` 피처 추가.
