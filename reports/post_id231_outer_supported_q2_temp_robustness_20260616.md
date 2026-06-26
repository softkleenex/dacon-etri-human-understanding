# Post-ID231 outer-supported Q2 temperature robustness

ID231 applies only Q2 `temp_t0.85_s0` on top of ID226 because Q2 was the only target with positive outer calibration support.

Focused robustness ranking:
- ID227: local `0.464409`, robust `0.495627`, worst `Q3`/`id10`/`q2`
- ID230: local `0.472703`, robust `0.504400`, worst `Q3`/`id10`/`q2`
- ID228: local `0.482245`, robust `0.513849`, worst `S4`/`id10`/`q2`
- ID231: local `0.491287`, robust `0.524343`, worst `S4`/`id10`/`q2`
- ID226: local `0.491771`, robust `0.524870`, worst `S4`/`id10`/`q2`
- ID224: local `0.509110`, robust `0.539895`, worst `S4`/`id10`/`q4`
- ID221: local `0.532990`, robust `0.564811`, worst `S4`/`id10`/`q4`
- ID178: local `0.538638`, robust `0.571839`, worst `S4`/`id10`/`q4`

ID231 vs ID226 shift: max `0.036313`, mean `0.003449`.

Public-gap projection, not proof:
- ID226: local `0.491771`; if ID178 gap repeats -> `0.554544`; needed gap for Public 0.55 `+0.058229`
- ID231: local `0.491287`; if ID178 gap repeats -> `0.554060`; needed gap for Public 0.55 `+0.058713`
- ID228: local `0.482245`; if ID178 gap repeats -> `0.545018`; needed gap for Public 0.55 `+0.067755`
- ID230: local `0.472703`; if ID178 gap repeats -> `0.535477`; needed gap for Public 0.55 `+0.077297`
- ID227: local `0.464409`; if ID178 gap repeats -> `0.527183`; needed gap for Public 0.55 `+0.085591`

Interpretation: ID231 is safer than ID228/230/227 because only one deterministic calibration target moves, but it gives almost no local margin toward Public 0.55. It is a validation-repair probe, not the score frontier.
