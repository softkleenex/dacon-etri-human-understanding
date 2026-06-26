# Post-ID226 convex bridge stability diagnostic

Confirmed Public truth remains ID178 `0.6014113009`. IDs 224-226 are unsubmitted and non-nested.

Robustness ranking among focused candidates:
- ID226: local `0.491771`, robust `0.524870`, worst `S4`/`id10`/`q2`
- ID189: local `0.492595`, robust `0.526584`, worst `S4`/`id10`/`q2`
- ID172: local `0.492762`, robust `0.527028`, worst `S4`/`id10`/`q2`
- ID225: local `0.496691`, robust `0.528599`, worst `S4`/`id10`/`q2`
- ID224: local `0.509110`, robust `0.539895`, worst `S4`/`id10`/`q4`
- ID223: local `0.518724`, robust `0.549816`, worst `S4`/`id10`/`q4`
- ID220: local `0.519561`, robust `0.550442`, worst `S4`/`id10`/`q4`
- ID217: local `0.519556`, robust `0.550573`, worst `S4`/`id10`/`q4`
- ID222: local `0.521966`, robust `0.553200`, worst `S4`/`id10`/`q4`
- ID221: local `0.532990`, robust `0.564811`, worst `S4`/`id10`/`q4`

Prediction shift vs clean ID221:
- ID222: max abs delta `0.067947`, mean abs delta `0.016390`
- ID223: max abs delta `0.091024`, mean abs delta `0.021886`
- ID224: max abs delta `0.149997`, mean abs delta `0.040494`
- ID225: max abs delta `0.299820`, mean abs delta `0.076413`
- ID226: max abs delta `0.540699`, mean abs delta `0.102600`

Public-gap projection, not proof:
- ID224: local `0.509110`; if ID178 gap repeats -> `0.571884`; needed gap for Public 0.55 is `+0.040890`
- ID225: local `0.496691`; if ID178 gap repeats -> `0.559465`; needed gap for Public 0.55 is `+0.053309`
- ID226: local `0.491771`; if ID178 gap repeats -> `0.554544`; needed gap for Public 0.55 is `+0.058229`

Interpretation: ID226 is now the lowest local frontier, but it is mostly ID172 and therefore the most public-gap fragile. ID224 is the first materially lower, cap-limited bridge; ID221 remains the cleaner branch.
