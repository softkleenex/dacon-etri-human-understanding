# Fifth-pass constrained-combo no-promotion decision

Scan: `reports/post333_failed_direction_constrained_combo_fifthpass_scan_20260616.json`

The fifth-pass scan included IDs 329-333 as sources. It did not produce a material new candidate, so no new submission artifacts were promoted.

| role | current id | current local | fifth-pass local | improvement | max Δ ID178 | failed projection | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| practical_cap080 | 329 | 0.516718 | 0.516601 | 0.000118 | 0.079192 | +0.0499 | no promote |
| tightscore_cap120 | 330 | 0.511769 | 0.511764 | 0.000006 | 0.113916 | +0.0499 | no promote |
| scoreguard_cap150 | 331 | 0.509930 | 0.509807 | 0.000123 | 0.149464 | +0.0498 | no promote |
| wideguard_cap240 | 332 | 0.503202 | 0.503149 | 0.000052 | 0.227831 | +0.0498 | no promote |

Best overall remained local `0.466963` with projection `+0.2693`; this matches the existing frontier/stress behavior rather than creating a safer public candidate.

Current recommendation remains ID329 as practical next candidate if a DACON public probe is explicitly approved. No DACON submission was made.
