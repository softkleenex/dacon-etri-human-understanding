# Candidate tier manifest after ID243

Best confirmed Public remains ID178 `0.6014113009`. Public 0.55 is not proven.

## Recommendation

ID235 remains the balanced next Public-0.55 probe. ID238 is the aggressive recombination probe. ID242/ID243 are extreme local-frontier calibration stress tests and should not be treated as public proof despite very low local CV.

## Candidate rows

- ID235: local `0.482255`, robustness `0.511518`, ID178-gap estimate `0.545029` — balanced mid-risk score-margin card; still the saner public probe
- ID238: local `0.462365`, robustness `0.493658`, ID178-gap estimate `0.525138` — aggressive local frontier before recalibration; full-OOF oracle recombination
- ID239: local `0.437967`, robustness `0.468053`, ID178-gap estimate `0.500741` — ID238 isotonic recalibration; first sub-0.44 frontier, high overfit risk
- ID240: local `0.417349`, robustness `0.444839`, ID178-gap estimate `0.480123` — ID238 q80 non-S3 + S4 isotonic; sub-0.42 quantile frontier
- ID241: local `0.383459`, robustness `0.406621`, ID178-gap estimate `0.446233` — ID238 q160 non-S3 + S4 isotonic; sub-0.39 ultra-risk frontier
- ID242: local `0.357651`, robustness `0.387740`, ID178-gap estimate `0.420424` — current raw local minimum; q240 non-S3 + S4 isotonic, severe overfit/S4 extremeness risk
- ID243: local `0.360420`, robustness `0.390285`, ID178-gap estimate `0.423193` — q240-all alternative; slightly worse local than ID242 but less S4 isotonic extremeness

## Frontier interpretation

- ID242 is the current raw local minimum at `0.357651`, but it is an extreme q240-after-oracle calibration card.
- ID243 is slightly less low at `0.360420`, but S4 is less extreme than ID242.
- For actual public probing, ID235/ID238 remain more meaningful than ID242/ID243.