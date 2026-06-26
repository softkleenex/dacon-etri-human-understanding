# Test split vs CV fold alignment diagnostic

Created: `2026-06-17 21:46:39 UTC+09:00`

This is a report-only validation diagnostic. No model was trained, no submission CSV was created, and no DACON submission was made.

## Actual train/test split shape

| subject | train | test | test frac | test runs | timeline runs |
|---|---:|---:|---:|---:|---|
| id01 | 41 | 27 | 0.397 | 2 | `train:0-27(28); test:28-41(14); train:42-54(13); test:55-67(13)` |
| id02 | 48 | 32 | 0.400 | 2 | `train:0-31(32); test:32-47(16); train:48-63(16); test:64-79(16)` |
| id03 | 33 | 21 | 0.389 | 2 | `train:0-21(22); test:22-32(11); train:33-43(11); test:44-53(10)` |
| id04 | 57 | 27 | 0.321 | 7 | `train:0-33(34); test:34-34(1); train:35-35(1); test:36-40(5); train:41-42(2); test:43-50(8); train:51-67(17); test:68-73(6); train:74-74(1); test:75-75(1); train:76-76(1); test:77-79(3); train:80-80(1); test:81-83(3)` |
| id05 | 44 | 21 | 0.323 | 5 | `train:0-25(26); test:26-32(7); train:33-33(1); test:34-38(5); train:39-51(13); test:52-53(2); train:54-56(3); test:57-58(2); train:59-59(1); test:60-64(5)` |
| id06 | 48 | 24 | 0.333 | 5 | `train:0-29(30); test:30-37(8); train:38-38(1); test:39-43(5); train:44-57(14); test:58-58(1); train:59-59(1); test:60-63(4); train:64-65(2); test:66-71(6)` |
| id07 | 49 | 30 | 0.380 | 2 | `train:0-32(33); test:33-47(15); train:48-63(16); test:64-78(15)` |
| id08 | 56 | 19 | 0.253 | 7 | `train:0-31(32); test:32-33(2); train:34-35(2); test:36-41(6); train:42-42(1); test:43-44(2); train:45-61(17); test:62-62(1); train:63-63(1); test:64-67(4); train:68-68(1); test:69-70(2); train:71-72(2); test:73-74(2)` |
| id09 | 41 | 27 | 0.397 | 2 | `train:0-27(28); test:28-41(14); train:42-54(13); test:55-67(13)` |
| id10 | 33 | 22 | 0.400 | 2 | `train:0-21(22); test:22-32(11); train:33-43(11); test:44-54(11)` |

## Fold-shape comparison

Actual test relative-position histogram: `[0.0, 0.0, 0.0, 0.0, 0.228, 0.268, 0.024, 0.0, 0.216, 0.264]`

| strategy | folds | mean L1 ↓ | min L1 | max L1 | verdict |
|---|---:|---:|---:|---:|---|
| subject_hole_5 | 5 | 1.8093 | 1.5059 | 1.9770 | less aligned |
| subject_timeseries_5 | 5 | 1.7452 | 1.2968 | 2.0000 | less aligned |
| actual_test_band_holdout_diagnostic | 1 | 1.2173 | 1.2173 | 1.2173 | less aligned |
| test_boundary_neighbor_holdout_diagnostic | 1 | 1.0740 | 1.0740 | 1.0740 | closer to interleaved hidden-test shape |

## Interpretation

- The hidden test rows are not a pure future holdout. Most subjects have two test blocks: one middle block and one later block.
- Some subjects (`id04`, `id05`, `id06`, `id08`) have fragmented later test blocks, so a single chronological split is structurally mismatched.
- Pure histogram alignment can favor late-window validation, but prior public results show that a pure chronological validation story is not enough for this competition.
- The diagnostic holdouts that use the known sample-row date geometry should be used as stress tests, not as a new optimization leaderboard.
- A future candidate should pass subject-hole, test-geometry stress, and LOSO-style robustness, then be checked against ID178/ID329/ID287 geometry before promotion.

## Decision

- Public 0.58 plausible candidate from this diagnostic: **no**
- Submission candidate promoted: **no**
- Recommended use: validation design evidence only.

