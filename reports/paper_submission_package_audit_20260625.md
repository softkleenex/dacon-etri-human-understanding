# Paper submission package audit

Created: `2026-09-15 22:05:18 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- Package audit pass: `True`
- pdflatex available: `False`
- tectonic available: `True`
- chktex available: `False`
- Human fill-ins remaining: `False`
- Run Kaggle now: `False`
- DACON submit now: `False`
- Next action: Review submission is complete. Monitor the EDAS email/status and do not use Final manuscript unless acceptance or organizer instructions open the camera-ready stage.

## Required Files

- `docs/PAPER_SUBMISSION_DRAFT_20260626.md`: `True`
- `docs/PAPER_SUBMISSION_CHECKLIST_20260626.md`: `True`
- `docs/paper_submission_20260626/README.md`: `True`
- `docs/paper_submission_20260626/paper.tex`: `True`
- `docs/paper_submission_20260626/paper.pdf`: `True`
- `docs/paper_submission_20260626/review_manuscript_submitted_20260717.pdf`: `True`
- `docs/paper_submission_20260626/references.bib`: `True`
- `docs/paper_submission_20260626/model_description_draft.md`: `True`
- `docs/paper_submission_20260626/id342_lineage_manifest.md`: `True`
- `docs/paper_submission_20260626/lineage_attribution_evidence.json`: `True`
- `docs/paper_submission_20260626/reviewer_response.md`: `True`
- `docs/paper_submission_20260626/reproducibility_runbook.md`: `True`
- `docs/paper_submission_20260626/edas_submission_handoff.md`: `True`
- `docs/paper_submission_20260626/edas_receipt.json`: `True`
- `docs/paper_submission_20260626/edas_metadata.json`: `True`
- `docs/paper_submission_20260626/edas_metadata.md`: `True`
- `docs/paper_submission_20260626/paper_length_budget.json`: `True`
- `docs/paper_submission_20260626/paper_length_budget.md`: `True`
- `docs/paper_submission_20260626/overleaf_compile_note.md`: `True`
- `docs/paper_submission_20260626/paper_submission_final_handoff.md`: `True`

## Required Claim Patterns

```json
{
  "alpha_sensitivity": true,
  "bibliography": true,
  "current_anchor_score": true,
  "current_title": true,
  "failed_frontier_controls": true,
  "id218_predecessor": true,
  "id342_anchor": true,
  "ieee_conference_class": true,
  "leaderboard_adaptation_boundary": true,
  "overlapping_similarity_families": true,
  "posthoc_validation_boundary": true,
  "processed_feature_count": true,
  "retrospective_rank_boundary": true,
  "similarity_attribution_boundary": true,
  "similarity_fold_dependence": true,
  "similarity_search_size": true,
  "transductive_deployment_check": true
}
```

## Checklist Patterns

```json
{
  "four_page_rule": true,
  "human_fillins": true,
  "no_dacon_spend_for_paper": true
}
```

## Lineage Patterns

```json
{
  "notebook_csv_log_public_chain": true,
  "public_055_boundary": true,
  "public_score": true
}
```

## Submission Lineage

```json
{
  "artifact_status": {
    "notebooks/history/submission_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.ipynb": true,
    "reports/dacon_three_slot_submission_result_20260623.md": true,
    "reports/decision_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.md": true,
    "submissions/submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv": true
  },
  "exists": true,
  "field_matches": {
    "is_submitted_bool": true,
    "local_cv_score": true,
    "notebook_file": true,
    "public_lb_score": true,
    "submission_file": true
  },
  "matches_expected": true,
  "matching_row_count": 1,
  "path": "submissions/submission_log.csv",
  "row_found": true
}
```

## EDAS Handoff Patterns

```json
{
  "dacon_rules_page": true,
  "edas_submission_link": true,
  "ictc_submission_page": true,
  "no_credentials": true,
  "review_upload_completed": true
}
```

## EDAS Metadata Patterns

```json
{
  "abstract_score_boundary": true,
  "active_with_manuscript": true,
  "affiliation": true,
  "author_count": true,
  "edas_link": true,
  "edas_paper_id": true,
  "keyword_count": true,
  "receipt_source": true,
  "registered_title_matches_source": true,
  "review_manuscript_uploaded": true,
  "title": true,
  "topic": true
}
```

## EDAS Receipt Patterns

```json
{
  "active_with_manuscript": true,
  "observation_source": true,
  "paper_id": true,
  "review_manuscript_uploaded": true,
  "submitted_pdf": true
}
```

## Length Budget Patterns

```json
{
  "actual_pdf_pages": true,
  "final_pdf_verified": true,
  "length_pass": true,
  "max_pages": true
}
```

## Overleaf Compile Note Patterns

```json
{
  "four_page_check": true,
  "local_pdf_verified": true,
  "main_file": true,
  "source_zip": true
}
```

## Final Handoff Patterns

```json
{
  "edas_link": true,
  "id342_score": true,
  "no_dacon_now": true,
  "public055_boundary": true,
  "source_zip": true
}
```

## Source Archive

```json
{
  "matches_expected": true,
  "report_archive_ready": true,
  "report_exists": true,
  "report_parse_ok": true,
  "report_path": "reports/paper_submission_source_archive_20260625.json",
  "required_names_present": {
    "README.md": true,
    "edas_metadata.json": true,
    "edas_metadata.md": true,
    "edas_receipt.json": true,
    "edas_submission_handoff.md": true,
    "id342_lineage_manifest.md": true,
    "lineage_attribution_evidence.json": true,
    "manifest.json": true,
    "model_description_draft.md": true,
    "overleaf_compile_note.md": true,
    "paper.tex": true,
    "paper_length_budget.json": true,
    "paper_length_budget.md": true,
    "paper_submission_final_handoff.md": true,
    "references.bib": true,
    "reproducibility_runbook.md": true,
    "reviewer_response.md": true,
    "support/PAPER_SUBMISSION_CHECKLIST_20260626.md": true
  },
  "zip_exists": true,
  "zip_path": "build/paper_submission_20260626/paper_submission_20260626_source.zip",
  "zip_read_ok": true
}
```

## PDF Status

```json
{
  "compiled_pdf": "build/paper_submission_20260626/compile/paper.pdf",
  "compiled_sha256": "73836b7caee3a595670bba367763630cd91393b8f9ab7748e6a44c58b3b3c428",
  "current_matches_compiled": true,
  "current_pdf": "docs/paper_submission_20260626/paper.pdf",
  "current_sha256": "73836b7caee3a595670bba367763630cd91393b8f9ab7748e6a44c58b3b3c428",
  "matches_expected": true,
  "receipt_parse_ok": true,
  "receipt_submitted_pdf": "docs/paper_submission_20260626/review_manuscript_submitted_20260717.pdf",
  "receipt_submitted_pdf_sha256": "1a91e5b1b5d3803822c89ec42d92f3224a34586badbd88d050d4d96ac1bec3a6",
  "submitted_matches_receipt": true,
  "submitted_pdf": "docs/paper_submission_20260626/review_manuscript_submitted_20260717.pdf",
  "submitted_sha256": "1a91e5b1b5d3803822c89ec42d92f3224a34586badbd88d050d4d96ac1bec3a6"
}
```

## Lineage Attribution

```json
{
  "alpha_sensitivity_checks": {
    "fold_wins_vs_id178": true,
    "mean_log_loss": true
  },
  "evidence_exists": true,
  "evidence_path": "docs/paper_submission_20260626/lineage_attribution_evidence.json",
  "local_hash_checks": {
    "feature_similarity_report": true,
    "id178_oof": true,
    "id218_oof": true,
    "id221_oof": true,
    "id342_oof": true,
    "metrics_train": true
  },
  "local_recomputation": {
    "available": true,
    "checks": {
      "id178_mean_log_loss": true,
      "id218_mean_log_loss": true,
      "id221_mean_log_loss": true,
      "id342_mean_log_loss": true,
      "id342_reconstruction": true,
      "midpoint_id178_id218_mean_log_loss": true,
      "midpoint_id178_id221_mean_log_loss": true,
      "similarity_increment": true,
      "similarity_share": true,
      "total_id178_to_id342_gain": true
    },
    "id221_predecessor_id": 218,
    "matches_evidence": true,
    "max_id342_reconstruction_error": 2.220446049250313e-16,
    "metrics": {
      "id178_mean_log_loss": 0.5386377373903545,
      "id218_mean_log_loss": 0.53410316496098,
      "id221_mean_log_loss": 0.5329901873426955,
      "id342_mean_log_loss": 0.5340000057700849,
      "midpoint_id178_id218_mean_log_loss": 0.5345081742255229,
      "midpoint_id178_id221_mean_log_loss": 0.5340000057700849,
      "similarity_increment": 0.0005081684554379962,
      "similarity_share": 0.10957263098558866,
      "total_id178_to_id342_gain": 0.004637731620269592
    },
    "missing_files": [],
    "row_keys_aligned": true
  },
  "matches_expected": true,
  "method_checks": {
    "id221_predecessor_id": true,
    "id342_formula": true,
    "midpoint_ablation_formula": true,
    "row_count": true,
    "similarity_candidates_per_target": true,
    "similarity_target_configuration_evaluations": true,
    "target_count": true
  },
  "metric_checks": {
    "id178_mean_log_loss": true,
    "id218_mean_log_loss": true,
    "id221_mean_log_loss": true,
    "id342_mean_log_loss": true,
    "midpoint_id178_id218_mean_log_loss": true,
    "midpoint_id178_id221_mean_log_loss": true,
    "similarity_increment": true,
    "similarity_share": true,
    "similarity_target_fold_comparisons": true,
    "similarity_target_fold_wins_vs_id218": true,
    "total_id178_to_id342_gain": true,
    "train_only_preprocessing_delta": true,
    "train_only_preprocessing_id221_mean_log_loss": true
  },
  "metric_consistency_checks": {
    "similarity_increment": true,
    "similarity_share": true,
    "total_gain": true
  },
  "paper_claims": {
    "midpoint_score": true,
    "public_attribution_boundary": true,
    "similarity_increment": true,
    "similarity_share": true,
    "total_gain": true
  },
  "source_records_valid": {
    "feature_similarity_report": true,
    "id178_oof": true,
    "id218_oof": true,
    "id221_oof": true,
    "id342_oof": true,
    "metrics_train": true
  }
}
```

## Forbidden Hits

```json
[]
```

## Human Fill-Ins

```json
[]
```
