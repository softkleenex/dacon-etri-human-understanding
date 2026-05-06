# Optuna 튜닝 가이드

모델 성능을 극대화하기 위한 하이퍼파라미터 탐색기입니다. 단순한 Random Search나 Grid Search와 달리, **베이지안 최적화(Bayesian Optimization)**를 사용하여 과거의 실험(Trial) 결과를 바탕으로 가장 정답에 가까운 공간(Search Space)을 영리하게 찾아갑니다.

---

## 1. 최적화 우선순위 (Tuning Strategy)
대회에서 점수에 가장 큰 영향을 미치는 핵심 파라미터부터 탐색해야 합니다. 탐색 범위를 너무 넓게 잡으면 무의미한 연산만 반복하게 됩니다.

- **LightGBM / XGBoost**: `learning_rate` (0.01 ~ 0.1), `max_depth` (3 ~ 10), `num_leaves` (depth에 비례), `colsample_bytree` (피처 샘플링 비율).
- **Deep Learning (PyTorch)**: `learning_rate` (Log-uniform), `weight_decay`, `dropout_rate`, `batch_size`.

---

## 2. K-Fold 기반 목적 함수 (Objective Function)
단일 검증 세트(Hold-out)만으로 튜닝하면 오버피팅(Overfitting)될 위험이 큽니다. 반드시 **K-Fold 교차 검증의 평균 점수**를 반환하도록 설계해야 실제 리더보드 점수와 동기화됩니다.

```python
import optuna
import numpy as np

def objective(trial):
    # 1. 탐색할 파라미터 제안 (Log 스케일 적극 활용)
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50)
    }
    
    cv_scores = []
    # 2. 5-Fold 검증 수행
    for fold in range(5):
        # 훈련 로직 호출
        score = train_fold_and_evaluate(fold_idx=fold, params=params)
        cv_scores.append(score)
        
        # 3. Early Pruning (조기 종료)
        # 만약 첫 1~2개 폴드 점수가 너무 안 좋다면, 나머지 폴드를 돌릴 필요 없이 해당 Trial을 즉시 버립니다.
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return np.mean(cv_scores)
```

---

## 3. 중단 및 재시작 (Continuous Tuning) 지원
Optuna는 기본적으로 메모리(Memory)에 실험 히스토리를 저장하지만, `storage` 파라미터에 SQLite 파일 경로를 지정하면 실험이 컴퓨터 재부팅 등으로 도중에 끊겨도 언제든 다시 이어서 튜닝할 수 있습니다.

```python
# DB 파일로 저장하여 끊임없는 탐색 지속
study = optuna.create_study(
    direction='maximize',           # 점수를 높이는 것이 목표라면 maximize, Loss라면 minimize
    study_name='dacon_lgbm_tuning', # 스터디 이름 고정
    storage='sqlite:///optuna_study.db', # 로컬 DB 파일 지정
    load_if_exists=True             # 기존 DB가 있으면 이어서 탐색
)

# 50번의 탐색 실행
study.optimize(objective, n_trials=50)

print(f"🥇 Best Score: {study.best_value}")
print(f"🛠 Best Params: {study.best_params}")
```

잠을 자거나 외출할 때 로컬 환경에서 백그라운드로 이 코드를 던져두고, `storage` 기능을 활용해 성능 한계점을 돌파하세요.
