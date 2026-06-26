# Post-ID342 deadline operating policy

Created: `2026-06-25 22:20:55 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- Deadline mode active: `True`
- Now: `2026-06-25 22:20:55 UTC+09:00`
- Paper deadline date: `2026-06-26`
- Paper deadline exact time known: `False`
- Algorithm close date: `2026-07-23`
- Current public anchor: `ID342` / `0.5960832519`
- Gap to Public 0.55: `+0.0460832519`
- Registered routes closed: `32/32`
- No current candidate: `True`
- Paper package audit pass: `True`
- Paper package human fill-ins remaining: `True`
- Run Kaggle now: `False`
- DACON submit now: `False`
- DACON spend policy: Do not spend DACON slots before a candidate clears a fresh local/stable-shape/failed-frontier gate. If a daily slot is expiring, use it only for an explicitly named information probe whose downside is accepted; current evidence does not justify such a probe.

## Paper Handoff Paths

```json
{
  "compile_note": "docs/paper_submission_20260626/overleaf_compile_note.md",
  "final_handoff": "docs/paper_submission_20260626/paper_submission_final_handoff.md",
  "package_audit": "reports/paper_submission_package_audit_20260625.json",
  "source_zip": "build/paper_submission_20260626/paper_submission_20260626_source.zip",
  "working_tex": "docs/paper_submission_20260626/paper.tex"
}
```

## Next Two Hours

- Freeze new long Kaggle/heavy runs.
- Use docs/paper_submission_20260626/paper.tex as the working IEEE source draft.
- Use docs/paper_submission_20260626/paper_submission_final_handoff.md for the exact handoff paths.
- Upload build/paper_submission_20260626/paper_submission_20260626_source.zip to Overleaf if a TeX PDF is needed.
- Fill human-only author/affiliation/EDAS fields and compile the four-page PDF outside this workspace if needed.
- Allow only report-only or local gates expected to finish within 30-60 minutes.
- Do not submit ID343/ID344 or other tiny micro-signal files without a new deadline-specific approval.

## Next Day

- If paper submission is required by 2026-06-26, prioritize method narrative, validation protocol, and negative-evidence appendix.
- Continue leaderboard work only through materially new Q3/S3/S4 hypotheses that can be locally gated.
- Keep ID342 as the best confirmed public anchor unless a new scored result beats 0.5960832519.

## Official Deadline Sources

```json
{
  "algorithm_close": {
    "date": "2026-07-23",
    "note": "DACON overview search result listed 07.23 Close and 04.21 Start Date.",
    "source_url": "https://dacon.io/en/competitions/official/236690/overview/description"
  },
  "paper_submission": {
    "corroborating_source_url": "https://dacon.io/competitions/official/236690/talkboard/416793",
    "date": "2026-06-26",
    "exact_time_known": false,
    "note": "ICTC 2026 paper-submission page lists Paper Submission Deadline: June 26, 2026. DACON talkboard also states the ETRI paper deadline follows ICTC 2026 and is 2026-06-26.",
    "source_url": "https://ictc.org/authors_paperSubmission"
  }
}
```

## Evidence Inputs

```json
{
  "paper_package_audit": {
    "exists": true,
    "human_fillins_remaining": true,
    "package_audit_pass": true,
    "parse_ok": true,
    "path": "reports/paper_submission_package_audit_20260625.json"
  },
  "priority": {
    "exists": true,
    "parse_ok": true,
    "path": "reports/post_id342_external_research_priority_20260625.json"
  },
  "submission_log": {
    "best_public_score": 0.5960832519,
    "best_scored_file": "submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv",
    "best_scored_id": "342",
    "exists": true,
    "path": "submissions/submission_log.csv",
    "recent_rows": [
      {
        "artifact_exists": "True",
        "artifact_role": "submission",
        "cv_lb_gap": "0.0625546286219425",
        "date": "2026-06-18 01:08:03",
        "duplicate_file_count": "1",
        "id": "337",
        "is_submitted": "True",
        "is_submitted_bool": "True",
        "local_cv_score": "0.538066",
        "local_cv_score_num": "0.538066",
        "model_description": "ID178 S1-only random-conv seed45502 overlay",
        "notebook_file": "notebooks/history/submission_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.ipynb",
        "notes": "Generated ID178 S1-only random-conv seed45502 overlay artifact. All non-S1 targets stay at ID178; S1 uses nk512/lr_c0_1/weight0.05 from the post-ID335 Kaggle anchor replay. Not Public 0.58-plausible; no DACON submission was made. | 2026-06-21 submitted via DACON daily slate slot 3/5; API message ok; public score pending. | 2026-06-21 DACON API returned public score 0.6006206286219425 for daily slate slot 3/5.",
        "public_lb_score": "0.6006206286219425",
        "public_lb_score_num": "0.6006206286219425",
        "risk_reason": "Absolute CV-LB gap is >= 0.03.",
        "safe_for_default_ensemble": "False",
        "status_clean": "cv_lb_gap_risk",
        "submission_file": "submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv"
      },
      {
        "artifact_exists": "True",
        "artifact_role": "submission",
        "cv_lb_gap": "0.06135351769999997",
        "date": "2026-06-18 01:14:14",
        "duplicate_file_count": "1",
        "id": "338",
        "is_submitted": "True",
        "is_submitted_bool": "True",
        "local_cv_score": "0.537861",
        "local_cv_score_num": "0.537861",
        "model_description": "ID179 S1-only random-conv seed45502 overlay",
        "notebook_file": "notebooks/history/submission_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.ipynb",
        "notes": "Generated ID179 S1-only random-conv seed45502 overlay artifact. All non-S1 targets stay at ID179; S1 uses nk512/lr_c0_1/weight0.05 from the post-ID335 Kaggle anchor replay. Not Public 0.58-plausible; no DACON submission was made. | 2026-06-23 submitted via three-slot slate slot 1/3; DACON API returned isSubmitted=True HTTP 200; public score pending. | 2026-06-24 authenticated DACON submission history correction: sub_id 1467936 at 2026-06-23 23:30:25 KST scored 0.5992145177; earlier team-best attribution to ID338 was wrong.",
        "public_lb_score": "0.5992145177",
        "public_lb_score_num": "0.5992145177",
        "risk_reason": "Absolute CV-LB gap is >= 0.03.",
        "safe_for_default_ensemble": "False",
        "status_clean": "cv_lb_gap_risk",
        "submission_file": "submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv"
      },
      {
        "artifact_exists": "True",
        "artifact_role": "submission",
        "cv_lb_gap": "0.062077778700000086",
        "date": "2026-06-18 02:17:30",
        "duplicate_file_count": "1",
        "id": "339",
        "is_submitted": "True",
        "is_submitted_bool": "True",
        "local_cv_score": "0.536441",
        "local_cv_score_num": "0.536441",
        "model_description": "ID178 Q123 seq-summary scale0.80 + ID337 S1",
        "notebook_file": "notebooks/history/submission_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.ipynb",
        "notes": "Generated ID178 Q1/Q2/Q3 sequence-summary scale0.80 plus ID337 S1 safety artifact. S2/S3/S4 stay at ID178. Max movement is below 0.10 versus ID178 in the post-A1 cap-sensitivity gate. Not Public 0.58-plausible; no DACON submission was made. | 2026-06-23 submitted via three-slot slate slot 3/3; DACON API returned isSubmitted=True HTTP 200; public score pending. | 2026-06-24 authenticated DACON submission history confirmed sub_id 1467938 at 2026-06-23 23:30:48 KST scored 0.5985187787.",
        "public_lb_score": "0.5985187787",
        "public_lb_score_num": "0.5985187787",
        "risk_reason": "Absolute CV-LB gap is >= 0.03.",
        "safe_for_default_ensemble": "False",
        "status_clean": "cv_lb_gap_risk",
        "submission_file": "submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv"
      },
      {
        "artifact_exists": "True",
        "artifact_role": "submission",
        "cv_lb_gap": "0.06295317525327704",
        "date": "2026-06-18 03:41:08",
        "duplicate_file_count": "1",
        "id": "340",
        "is_submitted": "True",
        "is_submitted_bool": "True",
        "local_cv_score": "0.538426",
        "local_cv_score_num": "0.538426",
        "model_description": "ID178 Q2-only seq-summary lr_c0_3 w0.01",
        "notebook_file": "notebooks/history/submission_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.ipynb",
        "notes": "Generated ID178 Q2-only sequence-summary lr_c0_3 weight 0.01 safety artifact. All non-Q2 targets stay at ID178. This was selected by the ID340 subset geometry scan; not Public 0.58-plausible; no DACON submission was made. | 2026-06-21 submitted via DACON daily slate slot 2/5; API message ok; public score pending. | 2026-06-21 DACON API returned public score 0.601379175253277 for daily slate slot 2/5.",
        "public_lb_score": "0.601379175253277",
        "public_lb_score_num": "0.601379175253277",
        "risk_reason": "Absolute CV-LB gap is >= 0.03.",
        "safe_for_default_ensemble": "False",
        "status_clean": "cv_lb_gap_risk",
        "submission_file": "submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv"
      },
      {
        "artifact_exists": "True",
        "artifact_role": "submission",
        "cv_lb_gap": "",
        "date": "2026-06-18 03:52:48",
        "duplicate_file_count": "1",
        "id": "341",
        "is_submitted": "True",
        "is_submitted_bool": "True",
        "local_cv_score": "0.538584",
        "local_cv_score_num": "0.538584",
        "model_description": "ID178 Q2-only seq-summary lr_c0_3 w0.0025",
        "notebook_file": "notebooks/history/submission_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.ipynb",
        "notes": "Generated ID178 Q2-only sequence-summary lr_c0_3 weight 0.0025 safety artifact. All non-Q2 targets stay at ID178. This was selected by the post-ID340 sequence-summary subset/micro geometry scans; not Public 0.58-plausible; no DACON submission was made. | 2026-06-21 reconciled as submitted from 2026-06-19 daily slate: DACON HTTP 200 observed before local pandas dtype failure; public score pending.",
        "public_lb_score": "",
        "public_lb_score_num": "",
        "risk_reason": "Marked submitted but has no public LB score.",
        "safe_for_default_ensemble": "False",
        "status_clean": "submitted_unscored",
        "submission_file": "submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv"
      },
      {
        "artifact_exists": "True",
        "artifact_role": "submission",
        "cv_lb_gap": "0.06208325189999997",
        "date": "2026-06-23 16:44:59",
        "duplicate_file_count": "1",
        "id": "342",
        "is_submitted": "True",
        "is_submitted_bool": "True",
        "local_cv_score": "0.534000",
        "local_cv_score_num": "0.534",
        "model_description": "ID178/ID221 feature-similarity shrink a0.50",
        "notebook_file": "notebooks/history/submission_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.ipynb",
        "notes": "Generated ID178/ID221 feature-similarity shrink alpha 0.50. All targets are convex blends against ID178 to keep independent strict-feature signal with lower movement. Not Public 0.58-plausible; no DACON submission was made. | 2026-06-23 submitted via three-slot slate slot 2/3; DACON API returned isSubmitted=True HTTP 200; public score pending. | 2026-06-24 authenticated DACON submission history confirmed sub_id 1467937 at 2026-06-23 23:30:37 KST scored 0.5960832519; this is the best confirmed 2026-06-23 slate result and current public anchor.",
        "public_lb_score": "0.5960832519",
        "public_lb_score_num": "0.5960832519",
        "risk_reason": "Absolute CV-LB gap is >= 0.03.",
        "safe_for_default_ensemble": "False",
        "status_clean": "cv_lb_gap_risk",
        "submission_file": "submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv"
      },
      {
        "artifact_exists": "True",
        "artifact_role": "submission",
        "cv_lb_gap": "",
        "date": "2026-06-23 23:55:40",
        "duplicate_file_count": "1",
        "id": "343",
        "is_submitted": "False",
        "is_submitted_bool": "False",
        "local_cv_score": "0.537854",
        "local_cv_score_num": "0.537854",
        "model_description": "ID338 S3 phone/wearable micro overlay",
        "notebook_file": "notebooks/history/submission_343_id338_s3_phone_wearable_micro_w0005_cv0.537854_unsubmitted.ipynb",
        "notes": "Generated ID338 S3-only phone/wearable discordance micro overlay. Q1/Q2/Q3/S1/S2/S4 stay at ID338; S3 blends an independent phone/wearable source at weight 0.005. Geometry gate passed but signal is tiny; no DACON submission was made.",
        "public_lb_score": "",
        "public_lb_score_num": "",
        "risk_reason": "Generated locally but not submitted/scored.",
        "safe_for_default_ensemble": "False",
        "status_clean": "generated_unsubmitted",
        "submission_file": "submission_id_343_id338_s3_phone_wearable_micro_w0005_cv0.537854_unsubmitted.csv"
      },
      {
        "artifact_exists": "True",
        "artifact_role": "submission",
        "cv_lb_gap": "",
        "date": "2026-06-24 00:07:41",
        "duplicate_file_count": "1",
        "id": "344",
        "is_submitted": "False",
        "is_submitted_bool": "False",
        "local_cv_score": "0.537839",
        "local_cv_score_num": "0.537839",
        "model_description": "ID338 S3 phone/wearable micro overlay",
        "notebook_file": "notebooks/history/submission_344_id338_s3_phone_wearable_w0020_cv0.537839_unsubmitted.ipynb",
        "notes": "Generated ID338 S3-only phone/wearable discordance micro overlay. Q1/Q2/Q3/S1/S2/S4 stay at ID338; S3 blends an independent phone/wearable source at weight 0.02. Geometry gate passed but signal is tiny; no DACON submission was made.",
        "public_lb_score": "",
        "public_lb_score_num": "",
        "risk_reason": "Generated locally but not submitted/scored.",
        "safe_for_default_ensemble": "False",
        "status_clean": "generated_unsubmitted",
        "submission_file": "submission_id_344_id338_s3_phone_wearable_w0020_cv0.537839_unsubmitted.csv"
      }
    ],
    "row_count": 344,
    "scored_count": 28,
    "submitted_count": 73
  },
  "synthesis": {
    "exists": true,
    "parse_ok": true,
    "path": "reports/post_id342_remaining_route_synthesis_20260625.json"
  }
}
```
