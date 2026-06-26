# Post-ID243 ID238 Recalibration Frontier Robustness

Formula: `overall + .20*(worst_subject-overall) + .15*(worst_date_quartile-overall) + .10*(worst_target-overall) + .05*max(0, late_q4-early_q1)`

## Ranking by robustness

- ID242: overall `0.357651`, robustness `0.387740`, ID178-gap estimate `0.420424`
- ID243: overall `0.360420`, robustness `0.390285`, ID178-gap estimate `0.423193`
- ID241: overall `0.383459`, robustness `0.406621`, ID178-gap estimate `0.446233`
- ID240: overall `0.417349`, robustness `0.444839`, ID178-gap estimate `0.480123`
- ID239: overall `0.437967`, robustness `0.468053`, ID178-gap estimate `0.500741`
- ID238: overall `0.462365`, robustness `0.493658`, ID178-gap estimate `0.525138`

## ID242 vs ID243

- ID242: local `0.357651`, robustness `0.387740`, S4 loss `0.331150`, S4 extreme `0.456000`, S4 p01/p99 `0.000001`/`0.999999`.
- ID243: local `0.360420`, robustness `0.390285`, S4 loss `0.350533`, S4 extreme `0.288000`, S4 p01/p99 `0.031250`/`0.986486`.

## Interpretation

ID242 is the raw local minimum. ID243 is a nearby q240-all alternative with less S4 isotonic extremeness, but both are extreme full-OOF calibration stress tests rather than public proof.