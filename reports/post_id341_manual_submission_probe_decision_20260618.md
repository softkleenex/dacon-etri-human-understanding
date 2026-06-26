# Post-ID341 manual submission probe decision

Created: `2026-06-21 16:36:54 UTC+09:00`

This is report-only. It does not submit to DACON, push/run Kaggle, create a new CSV, or edit the submission log.

## Decision

- Manual probe decision ready: `True`
- Public 0.58 plausible candidate exists: `False`
- Default action: `do_not_submit`
- Submit now: `False`
- Lowest-risk probe ID: `341`
- Lowest-risk probe file: `submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv`
- Lowest-risk artifact contract ready: `False`
- Lowest proxy focus ID: `340`
- Why not lowest-proxy by default: ID340 has the lowest transfer proxy among the 337-341 focus probes, but ID341 is the safer manual sanity probe because it has much smaller movement from the confirmed ID178 public anchor.
- Lowest-local-CV focus ID: `339`
- Why not lowest-local-CV by default: ID339 has the lowest local CV among the 337-341 focus probes, but it moves much more from ID178 and has a worse transfer proxy; this is exactly the failure mode seen in ID287/ID329.
- Requires explicit DACON approval: `True`
- Recommended primary next action: Use the explicit-approved no-submit Kaggle/Linux upstream rebuild. The previous ID341/340/337 sanity probes have now shown only the 0.60 public band, so do not spend another DACON submission on the same micro-probe family.

## Focus candidates

| id | local | proxy | Δ178 mean | Δ178 max | Δ341 max | joint tier | warn | combined tier | artifact ready | block reasons | file |
|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|---|
| 337 | 0.538066 | 0.603030 | 0.001336 | 0.037585 | 0.037585 | `low_joint_warning` | 0 | `joint_safe_but_transfer_blocked` | `False` | `transfer_proxy_above_public_058_gate,transfer_tier:recent_sequence_probe_unsubmitted` | `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv` |
| 338 | 0.537861 | 0.605302 | 0.004660 | 0.073989 | 0.073989 | `low_joint_warning` | 0 | `joint_safe_but_transfer_blocked` | `True` | `transfer_proxy_above_public_058_gate,transfer_tier:recent_sequence_probe_unsubmitted` | `submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv` |
| 339 | 0.536441 | 0.608259 | 0.012602 | 0.097051 | 0.095026 | `low_joint_warning` | 0 | `joint_safe_but_transfer_blocked` | `True` | `transfer_proxy_above_public_058_gate,transfer_tier:recent_sequence_probe_unsubmitted` | `submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` |
| 340 | 0.538426 | 0.601780 | 0.000494 | 0.008102 | 0.006076 | `low_joint_warning` | 0 | `joint_safe_but_transfer_blocked` | `False` | `transfer_proxy_above_public_058_gate,transfer_tier:recent_sequence_probe_unsubmitted` | `submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv` |
| 341 | 0.538584 | 0.603830 | 0.000124 | 0.002025 | 0.000000 | `low_joint_warning` | 0 | `joint_safe_but_transfer_blocked` | `False` | `transfer_proxy_above_public_058_gate,transfer_tier:current_control_no_submit` | `submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv` |

## Safety-ranked focus candidates

| id | local | proxy | Δ178 mean | Δ178 max | Δ341 max | joint tier | warn | combined tier | artifact ready | block reasons | file |
|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|---|
| 341 | 0.538584 | 0.603830 | 0.000124 | 0.002025 | 0.000000 | `low_joint_warning` | 0 | `joint_safe_but_transfer_blocked` | `False` | `transfer_proxy_above_public_058_gate,transfer_tier:current_control_no_submit` | `submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv` |
| 340 | 0.538426 | 0.601780 | 0.000494 | 0.008102 | 0.006076 | `low_joint_warning` | 0 | `joint_safe_but_transfer_blocked` | `False` | `transfer_proxy_above_public_058_gate,transfer_tier:recent_sequence_probe_unsubmitted` | `submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv` |
| 337 | 0.538066 | 0.603030 | 0.001336 | 0.037585 | 0.037585 | `low_joint_warning` | 0 | `joint_safe_but_transfer_blocked` | `False` | `transfer_proxy_above_public_058_gate,transfer_tier:recent_sequence_probe_unsubmitted` | `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv` |
| 338 | 0.537861 | 0.605302 | 0.004660 | 0.073989 | 0.073989 | `low_joint_warning` | 0 | `joint_safe_but_transfer_blocked` | `True` | `transfer_proxy_above_public_058_gate,transfer_tier:recent_sequence_probe_unsubmitted` | `submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv` |
| 339 | 0.536441 | 0.608259 | 0.012602 | 0.097051 | 0.095026 | `low_joint_warning` | 0 | `joint_safe_but_transfer_blocked` | `True` | `transfer_proxy_above_public_058_gate,transfer_tier:recent_sequence_probe_unsubmitted` | `submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` |

## Proxy-ranked focus candidates

| id | local | proxy | Δ178 mean | Δ178 max | Δ341 max | joint tier | warn | combined tier | artifact ready | block reasons | file |
|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|---|
| 340 | 0.538426 | 0.601780 | 0.000494 | 0.008102 | 0.006076 | `low_joint_warning` | 0 | `joint_safe_but_transfer_blocked` | `False` | `transfer_proxy_above_public_058_gate,transfer_tier:recent_sequence_probe_unsubmitted` | `submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv` |
| 337 | 0.538066 | 0.603030 | 0.001336 | 0.037585 | 0.037585 | `low_joint_warning` | 0 | `joint_safe_but_transfer_blocked` | `False` | `transfer_proxy_above_public_058_gate,transfer_tier:recent_sequence_probe_unsubmitted` | `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv` |
| 341 | 0.538584 | 0.603830 | 0.000124 | 0.002025 | 0.000000 | `low_joint_warning` | 0 | `joint_safe_but_transfer_blocked` | `False` | `transfer_proxy_above_public_058_gate,transfer_tier:current_control_no_submit` | `submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv` |
| 338 | 0.537861 | 0.605302 | 0.004660 | 0.073989 | 0.073989 | `low_joint_warning` | 0 | `joint_safe_but_transfer_blocked` | `True` | `transfer_proxy_above_public_058_gate,transfer_tier:recent_sequence_probe_unsubmitted` | `submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv` |
| 339 | 0.536441 | 0.608259 | 0.012602 | 0.097051 | 0.095026 | `low_joint_warning` | 0 | `joint_safe_but_transfer_blocked` | `True` | `transfer_proxy_above_public_058_gate,transfer_tier:recent_sequence_probe_unsubmitted` | `submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` |

## Anchor public truth

```json
{
  "id178_public": 0.6014113009,
  "id287_public": 0.6698787318,
  "id329_public": 0.6017908577
}
```

