# Post-ID233 partial S4 calibration robustness

Formula: `overall + .20*(worst_subject-overall) + .15*(worst_date_quartile-overall) + .10*(worst_target-overall) + .05*max(0, late_q4-early_q1)`

## Ranking by robustness

- ID230: overall `0.472703`, robustness `0.504400`
- ID233: overall `0.484413`, robustness `0.513300`
- ID228: overall `0.482245`, robustness `0.513849`
- ID232: overall `0.486685`, robustness `0.516793`
- ID231: overall `0.491287`, robustness `0.524343`
- ID226: overall `0.491771`, robustness `0.524870`
- ID224: overall `0.509110`, robustness `0.539895`
- ID221: overall `0.532990`, robustness `0.564811`

## New candidates

- ID232: overall `0.486685`, robustness `0.516793`, worst target `0.529776`, worst subject `0.590877`.
  - S4 test mean `0.554755`, std `0.237986`, p01/p99 `0.000001`/`0.999999`, extreme rate `0.064000`.
- ID233: overall `0.484413`, robustness `0.513300`, worst target `0.516014`, worst subject `0.586310`.
  - S4 test mean `0.554705`, std `0.234805`, p01/p99 `0.115385`/`0.961538`, extreme rate `0.060000`.

## Interpretation

ID232 is the least reckless new score-margin card: it changes only Q2 and S4, keeps S4 isotonic max movement much lower than q40/q80/q160 variants, and has nominal room for an ID178-sized public gap. ID233 is kept as a riskier S4 q40 probe, not as the first-choice candidate.