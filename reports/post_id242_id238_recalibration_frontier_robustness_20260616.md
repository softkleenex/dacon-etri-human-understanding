# Post-ID242 ID238 Recalibration Frontier Robustness

Formula: `overall + .20*(worst_subject-overall) + .15*(worst_date_quartile-overall) + .10*(worst_target-overall) + .05*max(0, late_q4-early_q1)`

## Ranking by robustness

- ID242: overall `0.357651`, robustness `0.387740`, ID178-gap estimate `0.420424`
- ID241: overall `0.383459`, robustness `0.406621`, ID178-gap estimate `0.446233`
- ID240: overall `0.417349`, robustness `0.444839`, ID178-gap estimate `0.480123`
- ID239: overall `0.437967`, robustness `0.468053`, ID178-gap estimate `0.500741`
- ID238: overall `0.462365`, robustness `0.493658`, ID178-gap estimate `0.525138`

## New Frontier Cards

- ID239: local `0.437967`, robustness `0.468053`, worst target `0.483945`, worst subject `0.539511`.
  - Q1 test mean `0.533046`, std `0.301578`, p01/p99 `0.000001`/`0.999999`, extreme `0.200000`.
  - Q2 test mean `0.558209`, std `0.252842`, p01/p99 `0.000001`/`0.999999`, extreme `0.088000`.
  - Q3 test mean `0.655874`, std `0.272792`, p01/p99 `0.000001`/`0.999999`, extreme `0.052000`.
  - S1 test mean `0.680903`, std `0.261477`, p01/p99 `0.066667`/`0.999999`, extreme `0.188000`.
  - S2 test mean `0.597687`, std `0.294345`, p01/p99 `0.000001`/`0.999999`, extreme `0.224000`.
  - S4 test mean `0.570326`, std `0.349057`, p01/p99 `0.000001`/`0.999999`, extreme `0.456000`.
- ID240: local `0.417349`, robustness `0.444839`, worst target `0.465115`, worst subject `0.513580`.
  - Q1 test mean `0.540286`, std `0.268238`, p01/p99 `0.071429`/`0.928571`, extreme `0.000000`.
  - Q2 test mean `0.544286`, std `0.250867`, p01/p99 `0.071429`/`0.928571`, extreme `0.000000`.
  - Q3 test mean `0.628048`, std `0.261870`, p01/p99 `0.071429`/`0.928571`, extreme `0.000000`.
  - S1 test mean `0.663762`, std `0.237569`, p01/p99 `0.071429`/`0.928571`, extreme `0.000000`.
  - S2 test mean `0.576476`, std `0.275480`, p01/p99 `0.071429`/`0.928571`, extreme `0.000000`.
  - S4 test mean `0.570326`, std `0.349057`, p01/p99 `0.000001`/`0.999999`, extreme `0.456000`.
- ID241: local `0.383459`, robustness `0.406621`, worst target `0.465115`, worst subject `0.443427`.
  - Q1 test mean `0.524000`, std `0.264037`, p01/p99 `0.125000`/`0.875000`, extreme `0.000000`.
  - Q2 test mean `0.537667`, std `0.267806`, p01/p99 `0.125000`/`0.875000`, extreme `0.000000`.
  - Q3 test mean `0.613167`, std `0.244904`, p01/p99 `0.125000`/`0.875000`, extreme `0.000000`.
  - S1 test mean `0.623500`, std `0.252744`, p01/p99 `0.125000`/`0.875000`, extreme `0.000000`.
  - S2 test mean `0.560000`, std `0.284161`, p01/p99 `0.125000`/`0.875000`, extreme `0.000000`.
  - S4 test mean `0.570326`, std `0.349057`, p01/p99 `0.000001`/`0.999999`, extreme `0.456000`.
- ID242: local `0.357651`, robustness `0.387740`, worst target `0.465115`, worst subject `0.433243`.
  - Q1 test mean `0.520333`, std `0.263318`, p01/p99 `0.166667`/`0.833333`, extreme `0.000000`.
  - Q2 test mean `0.533667`, std `0.256374`, p01/p99 `0.166667`/`0.833333`, extreme `0.000000`.
  - Q3 test mean `0.594333`, std `0.250258`, p01/p99 `0.166667`/`0.833333`, extreme `0.000000`.
  - S1 test mean `0.604667`, std `0.253838`, p01/p99 `0.166667`/`0.833333`, extreme `0.000000`.
  - S2 test mean `0.555667`, std `0.275955`, p01/p99 `0.166667`/`0.833333`, extreme `0.000000`.
  - S4 test mean `0.570326`, std `0.349057`, p01/p99 `0.000001`/`0.999999`, extreme `0.456000`.

## Interpretation

ID242 is the new raw local frontier, but q240 after an OOF oracle base is almost certainly overfit. Treat ID239/240/241/242 as calibration stress-test cards, not public proof.