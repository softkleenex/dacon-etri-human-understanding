# Submissions Integrity Audit

Found **2 findings**. Safe default ensemble rows: **14**.

## Status Counts

- `submitted_placeholder_score`: 33
- `generated_unsubmitted`: 29
- `scored`: 14
- `submitted_unscored`: 11
- `cv_lb_gap_risk`: 5
- `invalid_public_score`: 1

## Findings

### 1. [MEDIUM] Submission CSV files are not referenced by submission_log.csv

- Area: `artifacts`
- Detail: ['submission_calibrated_ensemble.csv', 'submission_calibrated_ensemble_v11.csv', 'submission_dl_sota_tuned_score0.6338.csv', 'submission_dl_v10.csv', 'submission_id_055_score0.5822.csv', 'submission_id_058_score0.5820.csv', 'submission_id_060_score0.5776.csv', 'submission_kaggle_xgb_optuna_v10_score0.6367.csv', 'submission_optuna_trial_0_score0.6583.csv', 'submission_optuna_trial_37_score0.6533.csv', 'submission_optuna_trial_40_score0.6461.csv', 'submission_optuna_trial_42_score0.6488.csv', 'submission_optuna_trial_45_score0.6537.csv', 'submission_optuna_trial_48_score0.6531.csv', 'submission_optuna_trial_50_score0.6495.csv', 'submission_optuna_trial_52_score0.6485.csv', 'submission_optuna_trial_54_score0.6462.csv', 'submission_optuna_trial_55_score0.6487.csv', 'submission_optuna_trial_56_score0.6478.csv', 'submission_optuna_trial_57_score0.6467.csv', 'submission_optuna_trial_58_score0.6436.csv', 'submission_optuna_trial_59_score0.6458.csv']
- Recommendation: Add log rows for important artifacts, or quarantine obsolete files under artifacts/.

### 2. [MEDIUM] Risk-labelled rows exist

- Area: `log`
- Detail: {'submitted_placeholder_score': 33, 'submitted_unscored': 11, 'cv_lb_gap_risk': 5, 'invalid_public_score': 1}
- Recommendation: Keep rows for history, but exclude them from default ensembling and score claims.
