# ⚖️ 평가 산식 및 지표 (Evaluation Metric)

모델의 성능을 어떻게 채점하는지에 대한 공식을 반드시 명확히 기재합니다. 에이전트(Gemini CLI)가 이 문서를 읽고 `src/features.py`나 손실 함수(Loss Function)를 목적에 맞게 튜닝(Custom Loss)할 수 있습니다.

## 🎯 공식 평가 지표 (Metric)
- (예: RMSE, MAE, F1-Score, NRMSE 등)

## 🧮 산식 (Formula)
- (대회 홈페이지에 공개된 수식을 텍스트 또는 수식 형태로 기입합니다.)
- 예: $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$

## 💡 에이전트를 위한 튜닝 전략 제안 (Agentic Strategy)
- (이 평가지표를 잘 받기 위해 어떤 접근을 해야 하는지 메모합니다.)
- 예: "MAE가 아니라 RMSE이므로 튀는 값(Outlier) 하나를 잘못 맞추면 페널티가 엄청나게 큽니다. 이상치 처리에 각별히 신경 쓰고, 손실 함수에 L2 Loss(MSE)를 최우선으로 적용하세요."
