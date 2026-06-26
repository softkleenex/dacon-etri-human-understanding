# Paper Submission Package 2026-06-26

This directory contains a deadline-oriented IEEE conference paper draft for the
5th ETRI Human Understanding AI Paper Challenge.

## Files

- `paper.tex`: IEEEtran-style 4-page full-paper draft scaffold.
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
- The contribution is a conservative public-anchor risk-control workflow and
  negative-evidence accounting, not a solved Q3/S3/S4 representation method.

## Human-Only Fill-Ins

- Author names, affiliations, and email.
- Final EDAS track selection and upload metadata.
- Final title wording, acknowledgments, and any required funding statements.
- Whether to keep internal report paths in the camera-ready paper or move them
  to an artifact/reproducibility supplement.
- Local PDF compilation was not verified in this workspace because `pdflatex`
  was not installed.

## Build

If a local TeX distribution is available:

```bash
cd docs/paper_submission_20260626
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

No build is required for repository checks; the source draft is the canonical
artifact here.
