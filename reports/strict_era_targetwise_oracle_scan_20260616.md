# Strict-era target-wise oracle scan

Scanned `129` strict-era artifacts with base OOF predictions.
Target-wise oracle mean is `0.491771`.

Best source per target:
- Q1: ID171, loss `0.497013`, file `submission_id_171_id163_q40_extreme_public_probe_cv0.493975_unsubmitted.csv`
- Q2: ID171, loss `0.491685`, file `submission_id_171_id163_q40_extreme_public_probe_cv0.493975_unsubmitted.csv`
- Q3: ID171, loss `0.516014`, file `submission_id_171_id163_q40_extreme_public_probe_cv0.493975_unsubmitted.csv`
- S1: ID171, loss `0.433412`, file `submission_id_171_id163_q40_extreme_public_probe_cv0.493975_unsubmitted.csv`
- S2: ID171, loss `0.477167`, file `submission_id_171_id163_q40_extreme_public_probe_cv0.493975_unsubmitted.csv`
- S3: ID172, loss `0.465115`, file `submission_id_172_id163_targetwise_extreme_public_probe_cv0.492762_unsubmitted.csv`
- S4: ID225, loss `0.561987`, file `submission_id_225_id221_id172_cap030_convex_bridge_probe_cv0.496691_unsubmitted.csv`

Interpretation: the target-wise artifact oracle equals ID226, so existing artifact recombination alone does not push local below `0.491771`. Further progress needs genuinely new signal or a different validation/learning scheme, not another static blend of existing submissions.
