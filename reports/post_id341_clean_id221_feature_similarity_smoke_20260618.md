# Post-ID341 clean ID221 feature-similarity smoke

Created: `2026-06-18 05:23:58 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.

## Read

Replay historical ID221 feature-similarity configs under frozen outer splits with fold-internal preprocessing and outer-train-only labels.

## OOF summary

- Candidate loss: `0.536547`
- Δ vs ID178: `-0.002091`
- Δ vs ID218 base: `+0.002444`
- Δ vs historical ID221: `+0.003557`
- Δ vs ID341: `-0.002037`

## Split gate

| split | loss | Δ178 | Δ218 | Δ221 | Δ341 |
|---|---:|---:|---:|---:|---:|
| `subject_hole_5` | 0.534627 | -0.003865 | +0.000665 | +0.001801 | -0.003810 |
| `subject_timeseries_5` | 0.536872 | -0.002876 | +0.001301 | +0.002353 | -0.002816 |
| `actual_test_band_holdout` | 0.540224 | -0.001380 | +0.001240 | +0.002565 | -0.001241 |
| `test_boundary_neighbor2_holdout` | 0.577675 | -0.008615 | +0.000157 | +0.003467 | -0.008463 |
| `test_boundary_neighbor5_holdout` | 0.556287 | -0.001699 | +0.000263 | +0.001506 | -0.001633 |
| `loso_10` | 0.540000 | -0.002465 | +0.002374 | +0.003329 | -0.002409 |

## Test geometry preview

- Mean/max Δ vs ID178: `0.037919` / `0.203124`
- Mean/max Δ vs ID341: `0.037806` / `0.203124`
- Projection onto ID287 failed direction: `-0.0162`
- Projection onto ID329 gap direction: `+0.3511`

## Decision

- Clean ID221 fixed-config signal: `False`
- Beats ID178 primary: `True`
- Beats ID341 primary: `True`
- Matches/beats historical ID221: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run heavy rebuild now: `False`
- Next action: Use this as a smoke read only. If the fixed historical configs do not survive, the next clean ID221 rebuild must use inner-fold config selection rather than replaying the public-structure ID221 choices.

