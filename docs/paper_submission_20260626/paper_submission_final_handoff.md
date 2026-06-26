# Paper Submission Final Handoff

Status: deadline handoff for the 2026-06-26 paper-submission path. This is
non-score-bearing and creates no predictions, feature matrices, submission CSVs,
Kaggle runs, submission-log rows, or DACON submissions.

## Current Decision

- Paper submission deadline date: `2026-06-26` (exact time not confirmed in
  the local policy report).
- Algorithm close date: `2026-07-23`.
- Paper package audit: use `reports/paper_submission_package_audit_20260625.md`.
- Source ZIP: `build/paper_submission_20260626/paper_submission_20260626_source.zip`.
- EDAS submission link: `https://edas.info/newPaper.php?c=35149`.
- Working source: `docs/paper_submission_20260626/paper.tex`.
- Metadata to copy: `docs/paper_submission_20260626/edas_metadata.md`.
- Compile note: `docs/paper_submission_20260626/overleaf_compile_note.md`.

## Score Boundary

- Current confirmed public anchor: ID342, Public `0.5960832519`, local
  `0.534000`, DACON submission id `1467937`.
- Public `0.55` is not achieved.
- The paper should present a conservative, reproducible risk-control workflow
  and negative-evidence accounting, not prize-range performance.

## Submit Policy

- Kaggle run now: `False`.
- DACON submit now: `False`.
- Do not spend DACON slots solely to improve the paper story.
- Reopen score-bearing work only if a new candidate clears a fresh local,
  stable-shape, and failed-frontier gate.

## Human Fill-Ins

- Author names, affiliations, email addresses, and final author order.
- EDAS track selection and paper metadata confirmation.
- Final compiled PDF page count and EDAS PDF-checker result.
- EDAS paper ID and final submission timestamp.

## Verification Commands

```bash
make paper-submission-package-audit
make post-id342-deadline-operating-policy
make docs-audit
```
