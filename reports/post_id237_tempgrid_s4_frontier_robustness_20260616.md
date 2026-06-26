# Post-ID237 temp-grid S4 frontier robustness

Formula: `overall + .20*(worst_subject-overall) + .15*(worst_date_quartile-overall) + .10*(worst_target-overall) + .05*max(0, late_q4-early_q1)`

## Ranking by robustness

- ID227: overall `0.464409`, robustness `0.495627`, ID178-gap estimate `0.527182`
- ID237: overall `0.466691`, robustness `0.497142`, ID178-gap estimate `0.529464`
- ID230: overall `0.472703`, robustness `0.504400`, ID178-gap estimate `0.535476`
- ID236: overall `0.474985`, robustness `0.505914`, ID178-gap estimate `0.537758`
- ID235: overall `0.482255`, robustness `0.511518`, ID178-gap estimate `0.545029`
- ID233: overall `0.484413`, robustness `0.513300`, ID178-gap estimate `0.547187`
- ID228: overall `0.482245`, robustness `0.513849`, ID178-gap estimate `0.545018`
- ID234: overall `0.484527`, robustness `0.515224`, ID178-gap estimate `0.547300`
- ID232: overall `0.486685`, robustness `0.516793`, ID178-gap estimate `0.549458`
- ID231: overall `0.491287`, robustness `0.524343`, ID178-gap estimate `0.554060`
- ID226: overall `0.491771`, robustness `0.524870`, ID178-gap estimate `0.554544`
- ID224: overall `0.509110`, robustness `0.539895`, ID178-gap estimate `0.571883`
- ID221: overall `0.532990`, robustness `0.564811`, ID178-gap estimate `0.595763`

## New temp-grid frontier cards

- ID234: overall `0.484527`, robustness `0.515224`, worst target `0.529776`, modified `['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S4']`.
  - S4 test mean `0.554755`, std `0.237986`, p01/p99 `0.000001`/`0.999999`, extreme rate `0.064000`.
- ID235: overall `0.482255`, robustness `0.511518`, worst target `0.513875`, modified `['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S4']`.
  - S4 test mean `0.554705`, std `0.234805`, p01/p99 `0.115385`/`0.961538`, extreme rate `0.060000`.
- ID236: overall `0.474985`, robustness `0.505914`, worst target `0.513626`, modified `['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S4']`.
  - S4 test mean `0.564571`, std `0.240788`, p01/p99 `0.071429`/`0.928571`, extreme rate `0.000000`.
- ID237: overall `0.466691`, robustness `0.497142`, worst target `0.513626`, modified `['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S4']`.
  - S4 test mean `0.552000`, std `0.255830`, p01/p99 `0.125000`/`0.875000`, extreme rate `0.000000`.

## Interpretation

ID234/235 are the more defensible temp-grid cards. ID236/237 preserve lower local-score frontier options with narrower non-S4 calibration than ID230/ID227, but S4 q80/q160 remains very high public-gap risk.