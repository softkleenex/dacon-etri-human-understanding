# Post-ID230 second-stage tempered ladder robustness

Confirmed Public truth remains ID178 `0.6014113009`. IDs 228-230 are unsubmitted and full-OOF second-stage calibrated.

Focused robustness ranking:
- ID227: local `0.464409`, robust `0.495627`, worst `Q3`/`id10`/`q2`
- ID230: local `0.472703`, robust `0.504400`, worst `Q3`/`id10`/`q2`
- ID229: local `0.479973`, robust `0.510142`, worst `S4`/`id10`/`q2`
- ID228: local `0.482245`, robust `0.513849`, worst `S4`/`id10`/`q2`
- ID226: local `0.491771`, robust `0.524870`, worst `S4`/`id10`/`q2`
- ID224: local `0.509110`, robust `0.539895`, worst `S4`/`id10`/`q4`
- ID221: local `0.532990`, robust `0.564811`, worst `S4`/`id10`/`q4`
- ID178: local `0.538638`, robust `0.571839`, worst `S4`/`id10`/`q4`

Prediction shift vs ID226 base:
- ID228: max `0.141433`, mean `0.021671`
- ID229: max `0.292814`, mean `0.028410`
- ID230: max `0.413693`, mean `0.032320`
- ID227: max `0.575022`, mean `0.038031`

Prediction shift vs clean ID221:
- ID228: max `0.582364`, mean `0.113369`
- ID229: max `0.582364`, mean `0.116656`
- ID230: max `0.582364`, mean `0.119213`
- ID227: max `0.650998`, mean `0.123295`

Public-gap projection, not proof:
- ID228: local `0.482245`; if ID178 gap repeats -> `0.545018`; needed gap for Public 0.55 is `+0.067755`
- ID229: local `0.479973`; if ID178 gap repeats -> `0.542747`; needed gap for Public 0.55 is `+0.070027`
- ID230: local `0.472703`; if ID178 gap repeats -> `0.535477`; needed gap for Public 0.55 is `+0.077297`
- ID227: local `0.464409`; if ID178 gap repeats -> `0.527183`; needed gap for Public 0.55 is `+0.085591`

Interpretation: ID228 is the least aggressive second-stage candidate that still has nominal ID178-gap margin. ID230/ID227 are lower but increasingly dominated by high-variance S4 quantile calibration.
