# Paper submission package audit

Created: `2026-07-31 15:23:50 UTC+09:00`

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
- `docs/paper_submission_20260626/references.bib`: `True`
- `docs/paper_submission_20260626/model_description_draft.md`: `True`
- `docs/paper_submission_20260626/id342_lineage_manifest.md`: `True`
- `docs/paper_submission_20260626/reproducibility_runbook.md`: `True`
- `docs/paper_submission_20260626/edas_submission_handoff.md`: `True`
- `docs/paper_submission_20260626/edas_metadata.json`: `True`
- `docs/paper_submission_20260626/edas_metadata.md`: `True`
- `docs/paper_submission_20260626/paper_length_budget.json`: `True`
- `docs/paper_submission_20260626/paper_length_budget.md`: `True`
- `docs/paper_submission_20260626/overleaf_compile_note.md`: `True`
- `docs/paper_submission_20260626/paper_submission_final_handoff.md`: `True`

## Required Claim Patterns

```json
{
  "bibliography": true,
  "current_anchor_score": true,
  "current_title": true,
  "failed_frontier_controls": true,
  "id342_anchor": true,
  "ieee_conference_class": true
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
  "review_manuscript_uploaded": true,
  "title": true,
  "topic": true
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
    "edas_submission_handoff.md": true,
    "id342_lineage_manifest.md": true,
    "manifest.json": true,
    "model_description_draft.md": true,
    "overleaf_compile_note.md": true,
    "paper.tex": true,
    "paper_length_budget.json": true,
    "paper_length_budget.md": true,
    "paper_submission_final_handoff.md": true,
    "references.bib": true,
    "reproducibility_runbook.md": true,
    "support/PAPER_SUBMISSION_CHECKLIST_20260626.md": true
  },
  "zip_exists": true,
  "zip_path": "build/paper_submission_20260626/paper_submission_20260626_source.zip",
  "zip_read_ok": true
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
