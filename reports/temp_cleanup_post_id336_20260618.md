# Temporary file/document cleanup post-ID336

Created: `2026-06-18 01:02:12 UTC+09:00`

## Actions

- Deleted local cache/build debris: .ruff_cache, .pytest_cache, .mypy_cache, non-venv __pycache__, non-venv .DS_Store, build/.
- Formatted and import-sorted scripts/diagnose_id336_s1_weight_sensitivity.py, but did not execute the diagnostic.
- Updated scripts/workspace_cleanup_manifest.py so ID336 decision/diagnostic files are classified intentionally.
- Regenerated workspace cleanup manifest and audit reports.
- Did not delete, move, or submit any DACON-facing submission artifact.
- Did not bulk-archive reports/scripts because many are experiment evidence and require explicit promotion/archive decisions.

## Preserved submission-contract artifacts

- `submission_csv`: `submissions/submission_id_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.csv`
- `history_notebook`: `notebooks/history/submission_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.ipynb`
- `decision_report`: `reports/decision_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.md`
- `metadata_report`: `reports/experiment_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.json`
- `base_oof`: `artifacts/base_predictions/base_oof_id_336_id178_s1_seed45502_randomconv_overlay_cv0.537663.csv`
- `base_test`: `artifacts/base_predictions/base_test_id_336_id178_s1_seed45502_randomconv_overlay_cv0.537663.csv`
- `submission_log`: `submissions/submission_log.csv`

## Current authoritative docs/reports

- `docs/WORKFLOW_REVIEW.md`
- `docs/PUBLIC_055_PLAN.md`
- `docs/VALIDATION_REDESIGN_POST_ID329_20260617.md`
- `docs/EXPERIMENT_OPERATIONS.md`
- `reports/decision_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.md`
- `reports/id336_s1_only_public_risk_gate_20260618.md`
- `reports/post_id336_public_anchor_rerank_20260618.md`
- `reports/sequence_anchor_probe_decision_post_id335_20260618.md`

## Audit outcomes

- `docs_audit`: pass, no findings in command output (`reports/docs_records_audit.md`)
- `artifact_contract_audit`: pass, strict blockers 0; legacy low findings remain (`reports/experiment_artifacts_audit.md`)
- `submissions_audit`: pass with known risk-labelled rows finding (`reports/submissions_audit.md`)
- `structure_audit`: pass with only local .env low finding after cache deletion (`reports/repository_structure_audit.md`)

## Dirty tree after cleanup

- status entries: `903`
- status kinds: `{' M': 26, '??': 877}`
- by top-level:
  - `reports`: `759`
  - `scripts`: `114`
  - `docs`: `10`
  - `notebooks`: `5`
  - `src`: `5`
  - `tests`: `4`
  - `.gitignore`: `1`
  - `AGENTS.md`: `1`
  - `CHANGELOG.md`: `1`
  - `Makefile`: `1`
  - `README.md`: `1`
  - `submissions`: `1`
- cleanup manifest summary:
  - `intended`: `50`
  - `generated_untracked`: `0`
  - `review_required`: `853`

## Decision

No mass deletion/archive was performed. Bulk reports and scripts remain review-required evidence, not automatic source of truth. No DACON submission was created or sent.
