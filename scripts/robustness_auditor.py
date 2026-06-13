import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.data_loader import DataLoader
from src.features.build_features import engineer_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_adversarial_validation(df_train, df_test, id_col, targets):
    """
    Performs Adversarial Validation to identify features that cause Domain Shift 
    between Train (2020) and Test (2023) sets.
    """
    logger.info("⚡ [Step 1] Running Adversarial Validation to Detect Domain Shift...")

    # 1. Prepare adversarial labels
    df_train_adv = df_train.copy()
    df_test_adv = df_test.copy()

    df_train_adv['is_test'] = 0
    df_test_adv['is_test'] = 1

    df_adv = pd.concat([df_train_adv, df_test_adv], ignore_index=True)

    # Drop labels, date, and IDs which are trivial separators
    drop_cols = targets + [id_col, "date", "timestamp", "sleep_date", "lifelog_date", "is_test"]
    for t in targets:
        drop_cols.extend([f"{t}_zscore", f"{t}_is_missing", f"{t}_base", f"base_{t}_prob", f"base_{t}_logit"])

    feature_cols = [c for c in df_adv.columns if c not in drop_cols]

    # Select only numeric features for ExtraTrees
    numeric_cols = df_adv[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = df_adv[numeric_cols].fillna(df_adv[numeric_cols].median())
    y = df_adv['is_test']

    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df_adv))
    feature_importances = np.zeros(len(numeric_cols))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        clf = ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])

        oof_preds[val_idx] = clf.predict_proba(X.iloc[val_idx])[:, 1]
        feature_importances += clf.feature_importances_ / cv.n_splits

    auc_score = roc_auc_score(y, oof_preds)
    logger.info(f"🛡️ Adversarial Validation AUC: {auc_score:.4f}")

    importance_df = pd.DataFrame({
        'feature': numeric_cols,
        'importance': feature_importances
    }).sort_values(by='importance', ascending=False)

    # Identify extremely risky features (Train/Test separators)
    risky_features = importance_df.head(15).to_dict(orient='records')

    return auc_score, risky_features

def run_target_coherence_audit(submission_path, train_metrics):
    """
    Audits the coherence and logical consistency of predicted targets in the ensemble submission.
    Checks for impossible target joint distributions (e.g. Extreme stress yet perfect sleep quality).
    """
    logger.info("⚡ [Step 2] Auditing Target Prediction Coherence and Logical Consistency...")

    if not os.path.exists(submission_path):
        logger.warning(f"Submission file {submission_path} not found. Skipping coherence audit.")
        return None

    sub_df = pd.read_csv(submission_path)

    # Required targets
    targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    # Calculate Correlation Matrix of Predictions
    pred_corr = sub_df[targets].corr()
    train_corr = train_metrics[targets].corr()

    # Measure Cosine Similarity of the two correlation matrices
    p_flat = pred_corr.values.flatten()
    t_flat = train_corr.values.flatten()
    cos_sim = np.dot(p_flat, t_flat) / (np.linalg.norm(p_flat) * np.linalg.norm(t_flat))

    # Log inconsistencies
    # Inconsistency 1: High Stress (Q3 > 0.7) and High Sleep Quality / Satisfaction (Q1 > 0.7)
    inc_1 = sub_df[(sub_df['Q3'] > 0.7) & (sub_df['Q1'] > 0.7)]
    inc_1_ratio = len(inc_1) / len(sub_df)

    # Inconsistency 2: High Fatigue (Q2 > 0.7) and High Sleep Quality / Satisfaction (Q1 > 0.7)
    inc_2 = sub_df[(sub_df['Q2'] > 0.7) & (sub_df['Q1'] > 0.7)]
    inc_2_ratio = len(inc_2) / len(sub_df)

    # Inconsistency 3: High Awakening WASO (S4 > 0.7) and High Deep Sleep (S3 > 0.7)
    inc_3 = sub_df[(sub_df['S4'] > 0.7) & (sub_df['S3'] > 0.7)]
    inc_3_ratio = len(inc_3) / len(sub_df)

    logger.info(f"Target Correlation Alignment Cosine Similarity: {cos_sim:.4f}")
    logger.info("⚠️ Logical Inconsistencies Detected in Submission:")
    logger.info(f"  - High Stress (Q3) & High Satisfaction (Q1): {inc_1_ratio:.2%} ({len(inc_1)} samples)")
    logger.info(f"  - High Fatigue (Q2) & High Satisfaction (Q1): {inc_2_ratio:.2%} ({len(inc_2)} samples)")
    logger.info(f"  - High WASO (S4) & High Deep Sleep (S3): {inc_3_ratio:.2%} ({len(inc_3)} samples)")

    return {
        'correlation_alignment': cos_sim,
        'pred_corr': pred_corr.to_dict(),
        'train_corr': train_corr.to_dict(),
        'inconsistencies': {
            'stress_vs_satisfaction': inc_1_ratio,
            'fatigue_vs_satisfaction': inc_2_ratio,
            'waso_vs_deep_sleep': inc_3_ratio
        }
    }

def run_oof_model_consensus_audit(submission_dir='submissions'):
    """
    Audits the consensus between the different models trained during OOF cross-validation.
    Finds which targets have high ensemble variance (disagreement) and where the predictions are most unstable.
    """
    logger.info("⚡ [Step 3] Auditing OOF Model Consensus and Variance Dynamics...")

    clean_model_names = ['catboost', 'lgbm', 'xgb', 'et', 'rf', 'dl', 'lr']
    targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    model_oof_dfs = {}
    for m in clean_model_names:
        path = os.path.join(submission_dir, f'oof_{m}_v10.csv')
        if os.path.exists(path):
            model_oof_dfs[m] = pd.read_csv(path)

    if len(model_oof_dfs) < 2:
        logger.warning("Not enough OOF files found for consensus audit.")
        return None

    # Analyze target-by-target model agreement
    consensus_metrics = {}

    for t in targets:
        pred_matrix = []
        for m, df in model_oof_dfs.items():
            # Safely fill NaNs with 0.5 to avoid np.corrcoef returning all NaNs
            pred_vals = df[t].fillna(0.5).values
            pred_matrix.append(pred_vals)

        pred_matrix = np.array(pred_matrix) # shape: (num_models, num_samples)

        # Calculate pairwise correlation between models for this target
        model_corr = np.corrcoef(pred_matrix)
        avg_model_corr = (np.sum(model_corr) - len(model_oof_dfs)) / (len(model_oof_dfs) * (len(model_oof_dfs) - 1))

        # Calculate sample-wise prediction variance
        sample_vars = np.var(pred_matrix, axis=0)
        avg_sample_var = np.mean(sample_vars)
        max_sample_var = np.max(sample_vars)

        logger.info(f"Target {t} consensus: Avg Model Correlation = {avg_model_corr:.4f}, Avg Prediction Variance = {avg_sample_var:.5f}")

        consensus_metrics[t] = {
            'avg_correlation': avg_model_corr,
            'avg_variance': avg_sample_var,
            'max_variance': max_sample_var
        }

    return consensus_metrics

def main():
    logger.info("=========================================================================")
    logger.info("🔍 STARTING CRITICAL ROBUSTNESS & SHAKE-OUT PREVENTION AUDIT")
    logger.info("=========================================================================")

    # 1. Initialize data loading
    data_loader = DataLoader(data_dir="data")
    metrics_df = data_loader.load_metrics()
    sample_sub_df = data_loader.load_sample_submission()

    if metrics_df is None or sample_sub_df is None:
        logger.error("Could not load metrics or submission sample. Aborting.")
        return

    # 2. Get cached datasets or build them
    processed_train_path = Path("data/processed/df_train_raw.parquet")
    processed_test_path = Path("data/processed/df_test_raw.parquet")

    if processed_train_path.exists() and processed_test_path.exists():
        logger.info("Loading cached raw datasets from data/processed/...")
        df_train_raw = pd.read_parquet(processed_train_path)
        df_test_raw = pd.read_parquet(processed_test_path)
    else:
        logger.info("Rebuilding from raw sensor parquets...")
        sensor_data_dict = data_loader.load_sensor_data()
        df_train_raw = build_training_dataset(metrics_df, sensor_data_dict)
        df_test_raw = build_training_dataset(sample_sub_df, sensor_data_dict)

    df_train_raw['is_test'] = 0
    df_test_raw['is_test'] = 1
    df_all_raw = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

    logger.info("Running complete feature engineering...")
    df_all = engineer_features(df_all_raw)

    df_train = df_all[df_all['is_test'] == 0].drop(columns=['is_test']).copy()
    df_test = df_all[df_all['is_test'] == 1].drop(columns=['is_test']).copy()

    id_col = "subject_id" if "subject_id" in df_train.columns else "user_id"
    targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    # Run Audits
    adv_auc, risky_features = run_adversarial_validation(df_train, df_test, id_col, targets)

    sub_path = 'submissions/submission_calibrated_ensemble.csv'
    coherence_results = run_target_coherence_audit(sub_path, metrics_df)

    consensus_results = run_oof_model_consensus_audit()

    # Save audit report to Markdown artifact
    artifact_path = Path("/Users/isangjae/.gemini/antigravity-cli/brain/54217bfb-9893-4a8d-9c7d-63812d5c7d6e/robustness_shakeout_audit_report.md")

    logger.info(f"Generating Critical Robustness and Shake-out Audit Report at {artifact_path}...")

    report_content = f"""# 🔍 리더보드 대폭락(Shake-out) 방지 및 강건성(Robustness) 심층 감사 보고서

이 보고서는 Train (2020년 수집)과 Test (2023년 수집) 데이터셋 간의 **도메인 시프트(Domain Shift)**, 타겟 예측 확률의 **논리적 모순율(Logical Inconsistency)**, 그리고 베이스 모델 간 **예측 합의도(Consensus & Variance)**를 엄격하게 추적하여 리더보드 점수 대폭락을 예방하고 로컬 CV의 강건성을 다차원적으로 검증하기 위해 자동으로 생성되었습니다.

---

## 🛡️ Part 1. 적대적 검증 (Adversarial Validation) 결과

적대적 검증 분류기(ExtraTrees)를 사용하여 Train 데이터와 Test 데이터를 구별하는 성능을 측정했습니다. 이 AUC 점수가 `0.50`에 가까울수록 두 데이터셋의 분포가 동일하며, `1.0`에 가까울수록 특정 피처들이 Train과 Test를 쉽게 가르고 있어 리더보드 폭락 위험성이 큼을 경고합니다.

*   **Adversarial Validation AUC Score**: `{adv_auc:.5f}`
"""

    if adv_auc > 0.85:
        report_content += f"""
> [!CAUTION]
> **위험 경보 (High Shake-out Risk)!** Adversarial AUC가 **{adv_auc:.4f}**로 매우 높습니다. 
> 특정 스마트폰/웨어러블 센서 피처가 2020년(Train)과 2023년(Test) 간 기기 사양 및 체크 주기의 차이로 인해 극심한 분포 격차를 보이고 있습니다. 모델이 해당 피처에 강하게 의존하고 있다면, 리더보드 제출 시 심각한 Shake-out(점수 폭락)이 발생할 것입니다. 아래의 고위험 피처 목록을 검토하십시오.
"""
    elif adv_auc > 0.60:
        report_content += f"""
> [!WARNING]
> **주의 경보 (Moderate Shift Detected)!** Adversarial AUC가 **{adv_auc:.4f}**입니다. 
> Train과 Test 간에 완만한 공변량 시프트(Covariate Shift)가 탐지되었습니다. 로컬 CV 개선이 리더보드 개선으로 직결되지 않을 수 있으므로, 아래의 기기 종속적 피처들의 영향력을 제어해야 합니다.
"""
    else:
        report_content += f"""
> [!NOTE]
> **양호 (Robust Feature Alignment)!** Adversarial AUC가 **{adv_auc:.4f}**로 안전 수준입니다. 
> Z-score 상대 정규화 및 피처 가공이 Train/Test 간 기기 수집 오차를 매우 효과적으로 제거하고 있습니다. 리더보드 폭락 위험이 비교적 낮습니다.
"""

    report_content += """

### 🚨 Train/Test를 쉽게 가르는 15대 고위험 피처 (Top 15 Risky Features)
아래 피처들은 Train과 Test 간 분포 변화가 가장 격심하여 분류기가 데이터셋을 쉽게 구분하도록 돕는 주범들입니다. 이 피처들이 예측 모델의 주요 변수로 채택되지 않도록 배제하거나, Z-score를 통한 상대 정규화를 한층 더 엄격히 보정해야 합니다.

| 순위 | 피처 이름 | 적대적 중요도 (Adversarial Importance) | 위험 요인 및 교정 가이드라인 |
| :--- | :--- | :--- | :--- |
"""

    for rank, feat in enumerate(risky_features, 1):
        feat_name = feat['feature']
        feat_imp = feat['importance']

        # Logic to explain potential reason of shift
        guide = " raw 센서 통계값. Z-score로 환산되지 않았거나 기기 주기 차이가 직접 노출됨. 피처 제거 권장."
        if "_zscore" in feat_name:
            guide = " Z-score 환산 피처임에도 분포 격차 잔존. 윈도우 스팬(Span) 튜닝 혹은 스shrinkage 추가 보정 필요."
        elif "_roll" in feat_name:
            guide = " 롤링 통계 피처. 공백 일수로 인한 결측 ffill 전파 왜곡 가능성 있음."
        elif "is_missing" in feat_name:
            guide = " 센서 유실 여부 지표. Train과 Test의 스마트폰 백그라운드 OS 킬링 정책 및 센서 유실 패턴 상이."

        report_content += f"| {rank} | `{feat_name}` | `{feat_imp:.5f}` |{guide} |\n"

    report_content += "\n---\n\n## ⚖️ Part 2. 타겟 간 일관성 및 논리성 감사 (Target Coherence Audit)\n"

    if coherence_results:
        report_content += f"""
앙상블 블렌더가 최종 예측한 확률 결과물(`submission_calibrated_ensemble.csv`)의 7대 정서/수면 상태 타겟 간의 논리적 모순성(Inconsistency) 비율을 계산했습니다.

*   **실제 Train 레이블 vs 예측 확률 상관 매트릭스 유사도 (Cosine Similarity)**: `{coherence_results['correlation_alignment']:.5f}`
    *   *분석*: 이 값이 `1.0`에 수 가까울수록 모델들이 예측한 타겟 간의 유기적 상관관계가 실제 데이터셋 내재적 도메인 지식(Stress-Fatigue 양의 상관, Satisfaction-Fatigue 음의 상관 등)과 일치함을 뜻합니다.

### ⚠️ 예측 결과 내 논리적 모순 검출 비율
인간 정서 및 수면의 생리학적 모순율을 스캔한 결과입니다:

1.  **극심한 스트레스(Q3 > 0.7)인데 기상 후 만족도 최상(Q1 > 0.7)인 모순 비율**: `{coherence_results['inconsistencies']['stress_vs_satisfaction']:.2%}`
2.  **극심한 피로감(Q2 > 0.7)인데 기상 후 만족도 최상(Q1 > 0.7)인 모순 비율**: `{coherence_results['inconsistencies']['fatigue_vs_satisfaction']:.2%}`
3.  **수면 중 잦은 각성(S4 > 0.7)인데 깊은 수면 최상(S3 > 0.7)인 생리학적 모순 비율**: `{coherence_results['inconsistencies']['waso_vs_deep_sleep']:.2%}`
"""

        # Highlight recommendations based on inconsistency
        any_high = any(v > 0.05 for v in coherence_results['inconsistencies'].values())
        if any_high:
            report_content += """
> [!WARNING]
> **모순 비율 경고!** 특정 타겟 쌍의 모순적 예측 비율이 5%를 초과하고 있습니다. 
> 7개 타겟을 독립된 체인으로 돌리는 과정에서 모델이 개별 타겟에만 편향되어 생리학적 도메인 제약조건을 깨뜨리고 있습니다. 
> **교정안**: 예측 확률에 대해 클래스 체인의 연결 강도(Logits) 가중치를 조정하거나, 모순을 유발하는 Outlier 예측 샘플에 대해 기하평균을 취하는 후처리 보정이 권장됩니다.
"""
        else:
            report_content += """
> [!NOTE]
> **타겟 일관성 안전 확인!** 모든 생리학적 모순율이 5% 미만으로 극히 낮습니다. 
> Sequential Classifier Chain Stacking 파이프라인이 타겟 간의 시간적/정서적 조건부 확률을 매우 조화롭고 논리적으로 보존하며 학습을 달성했습니다.
"""
    else:
        report_content += "\n*예측 결과 파일이 존재하지 않아 타겟 일관성 감사를 수행할 수 없습니다.*\n"

    report_content += "\n---\n\n## 🤝 Part 3. 베이스 모델 간 합의 및 변동성 감사 (OOF Consensus Audit)\n"

    if consensus_results:
        report_content += """
7개 베이스 모델(CatBoost, LGBM, XGBoost, ExtraTrees, RandomForest, DeepLearning, LogisticRegression)의 Out-of-Fold 예측 결과에 대한 상호 상관계수 및 샘플별 분산(Variance) 통계입니다. 분산이 지나치게 높은 타겟은 특정 모델의 과적합이나 데이터 결손에 민감하게 반응하고 있음을 뜻합니다.

| 타겟 | 모델 간 평균 상관계수 (Consensus Corr) | 평균 예측 분산 (Avg Pred Var) | 최대 예측 분산 (Max Pred Var) | 강건성 위험도 및 진단 결과 |
| :---: | :---: | :---: | :---: | :--- |
"""
        for t, metrics in consensus_results.items():
            avg_c = metrics['avg_correlation']
            avg_v = metrics['avg_variance']
            max_v = metrics['max_variance']

            risk = "안정 (Consensus 훌륭함)"
            if avg_c < 0.60:
                risk = "🔴 고위험 (모델 간 괴리 큼; 특정 모델이 과적합/튀고 있음)"
            elif avg_v > 0.02:
                risk = "🟡 주의 (평균 분산 다소 높음; 소프트 블렌딩 가중치 튜닝 필요)"

            report_content += f"| **{t}** | `{avg_c:.4f}` | `{avg_v:.5f}` | `{max_v:.5f}` | {risk} |\n"

        report_content += """
### 💡 종합 액션 아이템 및 피드백 루프 (Robustness Improvement Roadmap)
1.  **적대적 검증 극복**: `Part 1`에서 탐지된 상위 5대 고위험 피처(특히 `_zscore`가 아닌 원본 생 센서 데이터 컬럼들)를 `src/features/build_features.py`에서 제거하거나 `LeakFreeSubjectZScaler` 정규화 대상에 반드시 강제 포함시키십시오.
2.  **타겟 간 정합성 극대화**: 모순율이 높은 타겟(예: Q1 수면 만족도 vs Q2 피로도)에 대해 블렌딩 가중치에서 일반화 능력이 우수한 `ExtraTrees`와 `LogisticRegression`의 비중을 인위적으로 높여주어 극단적인 앙상블 오버피팅 확률을 억제하십시오.
3.  **리더보드 폭락(Shake-out) 대비 강건한 최종 선택**: 리더보드 제출 시 단순히 CV가 가장 낮았던 특정 트리 모델의 단일 예측 대신, Z-score 피처만으로 학습하고 `Beta Calibration`을 적용한 **Personalized Calibrated Stacking Ensemble**을 최종 제출물로 선택하여 Test 셋의 도메인 변동에 대응하십시오.
"""
    else:
        report_content += "\n*OOF 예측 결과가 충분하지 않아 모델 간 합의 감사를 수행할 수 없습니다.*\n"

    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"🎉 Robustness and Shake-out Audit Report successfully generated at {artifact_path}!")
    print("🎯 Audit Complete. Please inspect the generated artifact.")

if __name__ == '__main__':
    main()
