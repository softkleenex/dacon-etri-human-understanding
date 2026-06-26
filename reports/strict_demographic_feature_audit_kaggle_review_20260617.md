# Strict demographic Kaggle audit review — post ID329

Created: `2026-06-17 10:20:30 UTC+09:00`

## Verdict

No DACON submission was made, and no submission candidate is promoted. The demographic/static feature branch is **not Public 0.58-plausible** under the current gate. It remains useful only as a target-specific hypothesis.

## Why

The Kaggle full audit used `240` ExtraTrees estimators and reproduced the local smoke pattern:

| split | best candidate | score | interpretation |
|---|---|---:|---|
| subject-hole temporal | `compact_plus_demo__extratrees_d5_l8` | `0.616853` | demographics/static features look helpful when subjects are partially shared through the split structure |
| leave-one-subject-out stress | `compact_covariates__logreg_l2_c020` | `0.669770` | the robust stress split prefers compact covariates without demographic expansion |

Subject-hole improvement is real but suspect: `compact_plus_demo__extratrees_d5_l8` improves over `base_no_demo__extratrees_d5_l8` by `-0.007042` and over `compact_covariates__extratrees_d5_l8` by `-0.017906`. But LOSO does not support global promotion: `base_plus_demo__extratrees_d5_l8` is `0.002157` worse than `base_no_demo__extratrees_d5_l8`, and `compact_plus_demo__extratrees_d5_l8` is `0.027497` worse than the LOSO best.

## Target-level signal

- Subject-hole demo wins: `Q1, Q2, S1, S2, S3, S4`
- LOSO demo wins: `Q2, Q3, S2, S4`

This means demographic features may be tested later as per-target, fold-safe gates, especially where LOSO agrees, but they should not become a broad overlay or blend input yet.

## Decision

- Public 0.58 plausible: **no**
- Promote CSV/notebook/submission-log candidate: **no**
- DACON submission: **no**
- Next use: target-specific hypothesis only, with anchor geometry against ID178/ID329/ID287 before any artifact promotion.

## Related artifacts

- `reports/strict_demographic_feature_audit_kaggle_20260617.json`
- `reports/strict_demographic_feature_audit_smoke_20260617.json`
- `reports/strict_demographic_feature_audit_kaggle_20260617.md`
- `reports/remote_demographic_audit_manifest_20260617.json`
- `reports/dacon-etri-demographic-audit_20260617.log`
