# Paper Submission Package 2026-07-31

This directory contains the working IEEE conference paper draft for the 5th ETRI
Human Understanding AI Paper Challenge. It was originally created for the
2026-06-26 deadline and updated after the DACON notice extending the paper
deadline to `2026-07-31 23:59`.

Review submission is registered as EDAS paper `1571325954` with status
`Active (has manuscript)`. The four-page PDF has been uploaded.

## Files

- [`paper.pdf`](paper.pdf): EDAS에 제출한 내용과 일치하는 4-page review
  manuscript.
- `paper.tex`: IEEEtran-style 4-page full-paper draft.
- `references.bib`: bibliography entries for DACON/ICTC pages and related
  method context.
- `model_description_draft.md`: draft model-description material for later
  DACON/organizer handoff.
- `id342_lineage_manifest.md`: one-to-one mapping from notebook to CSV to
  submission-log row and DACON public score.
- `reproducibility_runbook.md`: safe handoff commands and non-score-bearing
  reproduction checks.
- `edas_submission_handoff.md`: official EDAS links and human upload checklist.
- `edas_metadata.json` / `edas_metadata.md`: EDAS title, abstract, and keywords
  extracted from `paper.tex`.
- `paper_length_budget.json` / `paper_length_budget.md`: source-level length
  and structure audit for the four-page draft.
- `overleaf_compile_note.md`: source-ZIP and Overleaf compilation handoff.
- `paper_submission_final_handoff.md`: deadline decision, source paths, score
  boundary, and human-only follow-ups.

## Source Archive

Build an Overleaf/TeX-ready source ZIP:

```bash
make paper-submission-metadata
make paper-submission-length-audit
make paper-submission-source-archive
```

The generated archive is ignored under
`build/paper_submission_20260626/paper_submission_20260626_source.zip`.
The tracked manifest report is
`reports/paper_submission_source_archive_20260625.md`.

## Current Claim Boundary

- Current confirmed public anchor: ID342, Public `0.5960832519`, local
  `0.534000`, DACON submission `1467937`.
- Public `0.55` is not achieved and must not be claimed.
- The paper presents anchor-constrained feature-similarity ensembling,
  validation-risk control, and negative-evidence accounting.
- Local `0.534000` is a non-nested model-selection score; the official Public
  `0.5960832519` result is the final comparative evidence.

## Remaining Follow-Up

- The review submission has no remaining human fill-ins.
- Monitor EDAS paper `1571325954` and the registered email for review results.
- Do not use the `Final manuscript` action until acceptance or an organizer
  instruction opens the camera-ready stage.
- If a camera-ready revision is requested, recheck title wording,
  acknowledgments, funding statements, pagination, and PDF compliance.

## Build

The verified local build uses Tectonic:

```bash
cd docs/paper_submission_20260626
tectonic paper.tex \
  --outdir ../../build/paper_submission_20260626/compile \
  --keep-logs --keep-intermediates
```

Verified result: IEEE conference Letter PDF, exactly four pages, with no
overfull boxes or unresolved citations. The tracked review manuscript is
[`paper.pdf`](paper.pdf); the reproducible build output is
`build/paper_submission_20260626/compile/paper.pdf`.
