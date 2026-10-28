# Paper Submission Package 2026-07-31

This directory contains the working IEEE conference paper draft for the 5th ETRI
Human Understanding AI Paper Challenge. It was originally created for the
2026-06-26 deadline and updated after the DACON notice extending the paper
deadline to `2026-07-31 23:59`.

Review submission is registered as EDAS paper `1571325954` with status
`Active (has manuscript)`. The four-page PDF has been uploaded.

## Files

- [`paper.pdf`](paper.pdf): current four-page revision rebuilt from `paper.tex`.
- [`review_manuscript_submitted_20260717.pdf`](review_manuscript_submitted_20260717.pdf):
  immutable snapshot of the review manuscript uploaded to EDAS.
- `paper.tex`: IEEEtran-style 4-page full-paper draft.
- `references.bib`: bibliography entries for DACON/ICTC pages and related
  method context.
- `model_description_draft.md`: draft model-description material for later
  DACON/organizer handoff.
- `id342_lineage_manifest.md`: one-to-one mapping from notebook to CSV to
  submission-log row and DACON public score.
- `lineage_attribution_evidence.json`: tracked aggregate OOF attribution values
  and source hashes; row-level labels and predictions remain excluded.
- `reviewer_response.md`: point-by-point response and the additional robustness
  analyses incorporated after the review.
- `reproducibility_runbook.md`: safe handoff commands and non-score-bearing
  reproduction checks.
- `edas_submission_handoff.md`: official EDAS links and human upload checklist.
- `edas_receipt.json`: author-confirmed EDAS paper ID, status observation, and
  submitted-PDF hash. This is not generated from `paper.tex`.
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
- ID221 is an ID218 multi-signal predecessor plus a small feature-similarity
  correction. OOF decomposition attributes about `11%` of ID342's local gain
  over ID178 to that correction; its public contribution is not isolated.
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

The verified local build uses Tectonic. Prefer the Make target because it also
synchronizes the tracked current PDF:

```bash
make paper-submission-pdf
```

The target writes `build/paper_submission_20260626/compile/paper.pdf` and copies
the same bytes to [`paper.pdf`](paper.pdf). The package audit requires both
files to match while separately preserving the submitted snapshot.
