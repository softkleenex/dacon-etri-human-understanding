# ID178-delta constrained pair-blend scan

Target-wise pair/single selection among existing candidates with max test delta versus ID178 constrained by cap.

| cap | local CV | max delta vs ID178 | selected ids by target |
|---:|---:|---:|---|
| 0.10 | 0.475057 | 0.099896 | Q1:255/178:logit_pair@0.1, Q2:255/178:logit_pair@0.1, Q3:255/178:logit_pair@0.05, S1:255/178:logit_pair@0.1, S2:255/178:logit_pair@0.1, S3:253/258:logit_pair@0.05, S4:258/178:logit_pair@0.3 |
| 0.12 | 0.463684 | 0.119513 | Q1:255/259:logit_pair@0.2, Q2:255/248:logit_pair@0.05, Q3:255/259:logit_pair@0.2, S1:255/259:logit_pair@0.2, S2:255/259:logit_pair@0.2, S3:258/178:prob_pair@0.2, S4:253/258:logit_pair@0.4 |
| 0.14 | 0.454800 | 0.139101 | Q1:255/252:prob_pair@0.05, Q2:255/252:prob_pair@0.05, Q3:255/252:prob_pair@0.05, S1:255/252:prob_pair@0.05, S2:255/252:prob_pair@0.05, S3:258/253:prob_pair@0.1, S4:258/255:logit_pair@0.05 |
| 0.16 | 0.441919 | 0.159583 | Q1:255/252:prob_pair@0.1, Q2:255/252:prob_pair@0.1, Q3:255/252:prob_pair@0.1, S1:255/252:prob_pair@0.1, S2:255/252:prob_pair@0.1, S3:258/247:logit_pair@0.4, S4:253/252:prob_pair@0.1 |
| 0.18 | 0.440031 | 0.179554 | Q1:259/259:single@0.0, Q2:255/252:prob_pair@0.1, Q3:259/259:single@0.0, S1:255/252:prob_pair@0.1, S2:255/252:prob_pair@0.1, S3:247/258:logit_pair@0.15, S4:259/259:single@0.0 |
