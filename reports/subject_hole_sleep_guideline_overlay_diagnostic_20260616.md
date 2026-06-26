# Public sleep-guideline overlay diagnostic

- Fold scheme: `subject_hole`
- Base IDs: `[221, 226, 231]`
- This is report-only; no submission files were generated.

## Summary

- ID221: base `0.532990` -> nested `0.533025` (delta `+0.000035`).
  - S1: base `0.467117`, nested `0.467117`, full-OOF best `base` = `0.467117`.
  - S2: base `0.528881`, nested `0.528881`, full-OOF best `base` = `0.528881`.
  - S3: base `0.489489`, nested `0.489733`, full-OOF best `base` = `0.489489`.
  - S4: base `0.588980`, nested `0.588980`, full-OOF best `base` = `0.588980`.
- ID226: base `0.491771` -> nested `0.491771` (delta `+0.000000`).
  - S1: base `0.433412`, nested `0.433412`, full-OOF best `base` = `0.433412`.
  - S2: base `0.477167`, nested `0.477167`, full-OOF best `base` = `0.477167`.
  - S3: base `0.465115`, nested `0.465115`, full-OOF best `base` = `0.465115`.
  - S4: base `0.561987`, nested `0.561987`, full-OOF best `base` = `0.561987`.
- ID231: base `0.491287` -> nested `0.491287` (delta `+0.000000`).
  - S1: base `0.433412`, nested `0.433412`, full-OOF best `base` = `0.433412`.
  - S2: base `0.477167`, nested `0.477167`, full-OOF best `base` = `0.477167`.
  - S3: base `0.465115`, nested `0.465115`, full-OOF best `base` = `0.465115`.
  - S4: base `0.561987`, nested `0.561987`, full-OOF best `base` = `0.561987`.

## Interpretation guardrail

Full-OOF best rows are optimistic diagnostics. Promotion should require nested improvement, small test movement, and a clean artifact-contract run.
