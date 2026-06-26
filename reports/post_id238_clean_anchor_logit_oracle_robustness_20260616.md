# Post-ID238 Clean-Anchor Logit Oracle Robustness

Formula: `overall + .20*(worst_subject-overall) + .15*(worst_date_quartile-overall) + .10*(worst_target-overall) + .05*max(0, late_q4-early_q1)`

## Ranking by robustness

- ID238: overall `0.462365`, robustness `0.493658`, ID178-gap estimate `0.525138`
- ID227: overall `0.464409`, robustness `0.495627`, ID178-gap estimate `0.527182`
- ID237: overall `0.466691`, robustness `0.497142`, ID178-gap estimate `0.529464`
- ID230: overall `0.472703`, robustness `0.504400`, ID178-gap estimate `0.535476`
- ID236: overall `0.474985`, robustness `0.505914`, ID178-gap estimate `0.537758`
- ID235: overall `0.482255`, robustness `0.511518`, ID178-gap estimate `0.545029`
- ID228: overall `0.482245`, robustness `0.513849`, ID178-gap estimate `0.545018`
- ID226: overall `0.491771`, robustness `0.524870`, ID178-gap estimate `0.554544`
- ID221: overall `0.532990`, robustness `0.564811`, ID178-gap estimate `0.595763`

## ID238 Details

- Local `0.462365`, robustness `0.493658`, worst target `0.509920`, worst subject `0.567964`.
- Q1: loss `0.485085`, test mean `0.531530`, std `0.282585`, p01/p99 `0.000052`/`0.999960`.
- Q2: loss `0.483905`, test mean `0.566232`, std `0.232171`, p01/p99 `0.000006`/`0.952702`.
- Q3: loss `0.509920`, test mean `0.646290`, std `0.244066`, p01/p99 `0.095698`/`0.908758`.
- S1: loss `0.423051`, test mean `0.682056`, std `0.245032`, p01/p99 `0.110016`/`0.999990`.
- S2: loss `0.468669`, test mean `0.607823`, std `0.269545`, p01/p99 `0.104143`/`0.999996`.
- S3: loss `0.465115`, test mean `0.646169`, std `0.246113`, p01/p99 `0.000001`/`0.999999`.
- S4: loss `0.400810`, test mean `0.553638`, std `0.257636`, p01/p99 `0.048404`/`0.951596`.

## Interpretation

ID238 is the new local frontier but is a high-risk full-OOF oracle recombination. It is useful as an aggressive score-margin card, not as public proof.