# Sensor-adherence/dropout smoke — post ID341

Created: `2026-06-18 06:21:40 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.

## Feature coverage

- Rows: `700` total / `450` train / `250` test
- Input numeric columns: `299`
- Feature columns: `94`
- Mean missing rate: train `0.0000`, test `0.0000`
- Family input column counts: `{'screen': 22, 'activity': 26, 'pedo': 31, 'hr': 66, 'light': 47, 'place': 27, 'usage': 31, 'ambience': 17, 'sleep_proxy': 62}`

## Selected target configs

| target | candidate | wins | mean Δ | max Δ |
|---|---|---:|---:|---:|
| _(none)_ |  |  |  |  |

## Decision

- Reviewable low-movement candidate: `False`
- Public `0.58` plausible: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Next action: If this source has a reviewable row, combine it with the existing sleep/evening source gate before any artifact. Otherwise close sensor-adherence as a standalone local lane and prioritize the approved Kaggle upstream rebuild.

