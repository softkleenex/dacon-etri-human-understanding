# Post-ID341 Q2/Q3/S3 source-hypothesis matrix

Created: `2026-06-23 18:07:54 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.

## Decision

- Matrix ready: `True`
- All focus targets missing locally: `True`
- All focus targets have historical ID226 signal: `True`
- Local Q2/Q3/S3 artifact promotable: `False`
- External direct concat allowed: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Requires explicit Kaggle approval: `True`
- Approval phrase: `Kaggle upstream rebuild no-submit 실행 승인`

Keep local ID342 closed. Use this matrix as the score-relevant design compass for the explicit-approved no-submit Kaggle upstream rebuild; only pulled reports that beat ID178 and pass Public-0.58 review gates may open a separate artifact-contract review.

## Budget and public-gap stop line

- Strict selected targets today: `Q1, S4`
- Additional sum delta needed to beat ID178: `-0.010472`
- Additional sum delta needed for Public-0.58 review: `-0.090352`
- Local CV needed for Public 0.58 under recent floor: `0.517445`
- Local CV needed for Public 0.58 under recent stress: `0.494927`

## Target-source matrix

| target | local status | ID226 Δ178 | ID226 fold wins | feature hits | first hypothesis | fail-fast rule |
|---|---|---:|---:|---:|---|---|
| Q2 | `missing` | -0.064994 | 23/23 | 73 | evening/late app-load and recovery burden before sleep | stop if Q2 does not win at least 4/5 primary splits with negative mean delta vs ID341 |
| Q3 | `missing` | -0.051075 | 23/23 | 108 | social/routine novelty plus evening app and activity load | stop if the source increases joint-distribution risk or cannot beat ID341 in at least 4/5 primary splits |
| S3 | `missing` | -0.025778 | 23/23 | 53 | pre-sleep quiet/dark transition and physiological settling before sleep | stop if S3 cannot produce a low-movement all-primary improvement; do not borrow Q1/S4 sleep quality movement |

## Target details

### Q2

- Definition: pre-sleep subjective physical fatigue; 1 means lower/better fatigue than the subject mean
- Label risk: individual-mean survey binarization makes subject-baseline leakage and temporal drift especially dangerous
- Paper hint: The reference paper reports mUsageStats+DW as a strong Q2 baseline family.
- Raw sources: `mUsageStats`, `wHr`, `wPedo`, `mActivity`, `mScreenStatus`, `mLight`, `wLight`
- Processed feature examples: `act_presleep_active`, `ch2025_mUsageStats_app_camera_sum`, `ch2025_mUsageStats_app_chrome_sum`, `ch2025_mUsageStats_app_gallery_sum`, `ch2025_mUsageStats_app_game_sum`, `ch2025_mUsageStats_app_instagram_sum`, `ch2025_mUsageStats_app_kakao_sum`, `ch2025_mUsageStats_app_music_sum`, `ch2025_mUsageStats_app_naver_sum`, `ch2025_mUsageStats_app_netflix_sum`
- External-data use: ETRI 2018-2020 survey fatigue scales may help as a proxy task, but direct row concatenation is blocked.
- Route: clean ID221 root -> clean ID226 target-wise backbone -> output-intake gate -> separate artifact review

### Q3

- Definition: pre-sleep subjective stress; 1 means lower/better stress than the subject mean
- Label risk: stress is survey-relative and can look good under full-OOF calibration while failing public transfer
- Paper hint: The reference paper shows wPedo+mUsageStats+DW as a strong Q3 combination.
- Raw sources: `mUsageStats`, `mWifi`, `mBle`, `mGps`, `mAmbience`, `wPedo`, `mActivity`, `wHr`
- Processed feature examples: `act_0_ratio`, `act_3_ratio`, `act_4_ratio`, `act_7_ratio`, `act_8_ratio`, `act_active_ratio`, `act_aftn_active`, `act_aftn_still`, `act_eve_active`, `act_eve_still`
- External-data use: External survey stress/fatigue scales are useful only as priors or pretraining labels, not as direct DACON rows.
- Route: clean ID221 root -> clean ID226 target-wise backbone -> output-intake gate -> separate artifact review

### S3

- Definition: objective sleep onset latency guideline adherence; 1 means acceptable sleep-onset latency
- Label risk: sensor-derived onset labels are likely driven by evening boundary timing, missingness, and sleep-window alignment
- Paper hint: The reference paper reports strong S3 signal from wPedo+DW and mLight+DW families.
- Raw sources: `mScreenStatus`, `mActivity`, `mACStatus`, `mLight`, `wLight`, `wHr`, `wPedo`
- Processed feature examples: `ac_night_charging`, `ac_presleep_charging`, `act_night_active`, `act_night_still`, `act_presleep_active`, `mlight_dark_ratio`, `mlight_night_mean`, `screen_night_on`, `screen_night_ratio`, `screen_presleep_on`
- External-data use: External sleep logs can inform onset/efficiency priors, but the current local gate forbids direct concat.
- Route: clean ID221 root -> clean ID226 target-wise backbone -> output-intake gate -> separate artifact review

## Interpretation

Q2/Q3/S3 are the right next battleground, but not because there is a safe local artifact waiting. They are missing locally, while historical ID226 carries target-wise signal across all primary folds. The clean move is therefore a no-submit upstream rebuild, not another same-family local blend.

