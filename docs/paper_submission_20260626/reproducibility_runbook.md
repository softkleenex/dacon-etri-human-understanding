# Reproducibility Runbook

Status: handoff runbook for the paper/model-description package. This document
does not create predictions, feature matrices, submission artifacts, Kaggle
runs, submission-log rows, or DACON submissions.

## Safe Checks

Run these commands before handing over the paper package:

```bash
make paper-submission-metadata
make paper-submission-length-audit
make paper-submission-source-archive
make paper-submission-package-audit
make post-id342-deadline-operating-policy
make docs-audit
git diff --check
```

Expected state:

- `make paper-submission-package-audit` passes with
  `package_audit_pass=True`.
- `make paper-submission-metadata` writes EDAS title/abstract/keywords from
  `paper.tex`.
- `make paper-submission-length-audit` passes as a source-level length check.
- `make paper-submission-source-archive` writes an ignored ZIP under
  `build/paper_submission_20260626/`.
- Tectonic compilation writes the verified four-page PDF to
  `build/paper_submission_20260626/compile/paper.pdf`.
- `make docs-audit` reports `Findings: 0`.
- `git diff --check` prints no whitespace errors.
- Deadline policy keeps `run_kaggle_now=False` and `dacon_submit_now=False`.

## Files To Hand Over For Paper Work

- `docs/paper_submission_20260626/paper.tex`
- `docs/paper_submission_20260626/references.bib`
- `docs/paper_submission_20260626/model_description_draft.md`
- `docs/paper_submission_20260626/id342_lineage_manifest.md`
- `docs/paper_submission_20260626/reproducibility_runbook.md`
- `docs/paper_submission_20260626/edas_submission_handoff.md`
- `docs/PAPER_SUBMISSION_CHECKLIST_20260626.md`

## Human Fill-Ins

The following cannot be completed by repository automation:

- EDAS account/session details,
- final rebuild after author metadata is inserted and EDAS upload,
- any required acknowledgments or funding text.

## Do Not Run From This Runbook

These commands are not part of the paper handoff path:

```bash
make dacon-submit
make run-local
make kaggle-push-code
make kaggle-push-upstream-rebuild-runner
```

DACON submissions require a fresh score-bearing candidate and explicit exact
file/memo review. The current deadline policy does not open such a candidate.
