# Post-ID342 remaining route synthesis

Created: `2026-06-25 18:00:26 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- Current public anchor: `ID342` / `0.5960832519`
- Gap to Public 0.55: `+0.0460832519`
- Closed routes: `32/32`
- All registered routes closed: `True`
- Missingness heavy stable passes: `0`
- Missingness best public proxy: `0.5961011961387227`
- Stable-envelope best public proxy: `0.5663627987353779`
- Target-reallocation best public proxy: `0.5661666874281543`
- Penalty-proxy credible 0.55 passes: `0`
- Validation-weighted gate pass: `False`
- Validation-weighted credible 0.55 passes: `0`
- Validation-weighted credible 0.58 passes: `0`
- Solar/holiday gate pass: `False`
- Solar/holiday selected targets: `[]`
- KMA weather gate pass: `False`
- KMA weather best target/proxy: `S4` / `0.5960062364316846`
- Public-data API smoke all three checked: `True`
- KMA living-weather 2024 source available: `False`
- AirKorea API access working: `True`
- AirKorea approved 2024 source available: `False`
- AirKorea downloaded 2024 archive available: `True`
- AirKorea downloaded 2024 coverage OK: `True`
- AirKorea downloaded 2024 gate pass: `False`
- AirKorea downloaded 2024 best target/proxy: `S4` / `0.5960560645096623`
- AirKorea downloaded 2024 gate pending: `False`
- External ETRI prior gate pass: `False`
- External ETRI Q2 fatigue sleep coupling: `True`
- External ETRI Q3 stress sleep coupling: `False`
- External ETRI direct 2026 feature route: `False`
- Cross-source split gate pass: `False`
- Cross-source review targets: `[]`
- Cross-source stable-alignment targets: `['S3']`
- S3 phone/GPS gate pass: `False`
- S3 phone/GPS best public proxy: `0.5958495488259661`
- Final-day independent sweep gate pass: `False`
- Final-day best public proxy: `0.5957956128922309`
- Sequence-event selected targets: `['S4']`
- Sequence-event public proxy: `0.5960468853717344`
- MIS-LSTM block selected targets: `['S4']`
- MIS-LSTM block public proxy: `0.5958734877252216`
- Kaggle upstream wait state: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- DACON submit now: `False`
- Next action: All registered post-ID342 same-family, existing-source, sleep-proxy, raw sequence, and heavy subject-state routes are closed, the validation-weighted pseudo-public audit opens no current candidate route, the first external solar/holiday prior gate is unstable, the KMA ASOS weather context gate finds only a tiny S4 signal near Public 0.596, the approved KMA living-weather and AirKorea information APIs do not expose a 2024 historical feature source, and the separate downloaded AirKorea 2024 gate also opens no Public-0.58 review row. Historical ETRI sleep/survey priors support Q2 fatigue-to-sleep coupling, but not a direct Q3/S3/S4 feature route. The cross-source split audit finds only a too-small S3 micro-alignment, a focused S3 phone/GPS interaction gate remains near Public 0.596, and the final-day independent-source sweep opens no Public-0.58 review row. Do not submit or launch another same-family Kaggle run; the next score work must use a materially different Q3/S3/S4 source mechanism.

## Routes

| route | status | decision value | report | reason |
|---|---|---:|---|---|
| `existing_candidate_shelf` | `closed` | `0` | `reports/post_id342_stable_lens_existing_candidate_triage_20260624.json` | Existing OOF candidates pass no stable-lens review after excluding ID287/ID329 traps. |
| `public_gap_shape_lens` | `closed` | `0` | `reports/post_id342_public_gap_risk_lens_20260624.json` | Stable prediction-shape lens finds no Public-0.55 shelf row and warns against larger frontier movement. |
| `public_gap_penalty_proxy_redesign` | `closed` | `0` | `reports/post_id342_public_gap_penalty_proxy_redesign_20260625.json` | Shape-excess penalty proxy finds no credible Public-0.55 OOF shelf row. |
| `validation_weighted_pseudo_public_split` | `closed` | `False` | `reports/post_id342_validation_weighted_pseudo_public_split_audit_20260625.json` | Target-weighted pseudo-public validation audit opens no credible Public-0.55 or review-threshold row. |
| `solar_seasonal_holiday_prior` | `closed` | `False` | `reports/post_id342_solar_seasonal_holiday_prior_gate_20260625.json` | External date-derived solar/seasonal/Korean-holiday priors show only unstable S4 movement. |
| `kma_asos_weather_context` | `closed` | `False` | `reports/post_id342_kma_weather_context_gate_20260625.json` | Historical Daejeon ASOS weather context shows only a tiny unstable S4 signal near Public 0.596. |
| `kma_life_airkorea_api_feasibility` | `closed` | `False` | `reports/public_data_weather_air_api_smoke_20260625.json` | KMA living-weather and the approved AirKorea information API work for current/recent data, but do not expose a date-bounded 2024 historical feature source. |
| `airkorea_2024_historical_context` | `closed` | `False` | `reports/post_id342_airkorea_2024_context_gate_20260625.json` | Downloaded 2024 Daejeon AirKorea context opens only a tiny unstable S4 hint near Public 0.596. |
| `external_etri_sleep_survey_prior` | `closed` | `False` | `reports/post_id342_external_etri_sleep_survey_prior_20260625.json` | Historical ETRI sleep/survey prior supports Q2 fatigue coupling but opens no direct Q3/S3/S4 route. |
| `cross_source_split_stability` | `closed` | `False` | `reports/post_id342_cross_source_split_stability_audit_20260625.json` | Weak-source split audit finds only a tiny S3 stable micro-alignment and unstable S4 split behavior. |
| `s3_phone_gps_interaction` | `closed` | `False` | `reports/post_id342_s3_phone_gps_interaction_gate_20260625.json` | Focused S3 phone/GPS concat improves all splits but remains near Public 0.596 and far from review thresholds. |
| `final_day_independent_signal_sweep` | `closed` | `False` | `reports/post_id342_final_day_independent_signal_sweep_20260625.json` | Final-day S3 temporal, S4 sleep-solar, and compact joint-state sweep opens no Public-0.58 review row. |
| `stable_envelope_multisource_recombination` | `closed` | `0` | `reports/post_id342_stable_envelope_multisource_optimizer_20260625.json` | Nonlegacy multi-source ID342-convex recombination stays above Public 0.55 under the stable-gap projection. |
| `strict_target_reallocation` | `closed` | `0` | `reports/post_id342_target_reallocation_optimizer_20260625.json` | Target-wise strict existing-source reallocation improves the envelope ceiling only to about Public 0.566. |
| `raw_temporal_q3s3_sequence_image` | `closed` | `False` | `reports/post_id342_raw_temporal_q3s3_sequence_image_smoke_20260624.json` | Raw temporal sequence-image smoke selected no Q3/S3 target under ID342 gates. |
| `sequence_summary_distance_kernel` | `closed` | `False` | `reports/post_id342_sequence_summary_distance_kernel_smoke_20260624.json` | Half-hour sequence summaries and distance kernels failed the final gate. |
| `sleep_boundary_tiny_cnn` | `closed` | `False` | `reports/post_id342_tiny_temporal_cnn_q3s3_smoke_20260624.json` | Sleep-boundary/SOL and tiny temporal-CNN local proxies selected no Q3/S3 target. |
| `feature_similarity_alpha_tuning` | `closed` | `0` | `reports/post_id342_feature_similarity_artifact_gate_20260624.json` | Post-public ID342 alpha rows fail strict artifact review and remain far from Public 0.55. |
| `paper_aligned_feature_similarity_rebuild` | `closed` | `False` | `reports/post_id342_paper_aligned_feature_similarity_rebuild_gate_20260624.json` | Domain-restricted same-subject feature-similarity rebuild loses to ID342. |
| `registered_material_single_or_fusion` | `closed` | `0` | `reports/post_id342_registered_material_fusion_gate_20260624.json` | Registered material matrices and their fusion pass no local gate. |
| `raw_subject_state_daily_windows` | `closed` | `0` | `reports/post_id342_raw_subject_state_feature_gate_20260624.json` | Full raw daily/window subject-state scan has high coverage but passes no ID342/Public-0.55 gate. |
| `raw_sequence_representation_remote` | `closed` | `False` | `reports/post_id342_s3q3_raw_sequence_representation_gate_20260624.json` | Kaggle raw sequence representation run passed no Q3/S3 row and produced no promotable evidence. |
| `self_supervised_subject_state_embedding` | `closed` | `True` | `reports/post_id342_self_supervised_subject_state_remote_intake_20260624.json` | Label-free subject-state execution is safe research evidence only; downstream gates did not open a route. |
| `missingness_cross_modal_subject_state_embedding` | `closed` | `0` | `reports/post_id342_missingness_subject_state_remote_intake_20260624.json` | Full missingness-aware cross-modal embedding scan had strong coverage but zero stable passes. |
| `env_social_evening_lightweight_sources` | `closed` | `False` | `reports/post_id342_evening_state_transition_gate_20260624.json` | Env/social/routine opens only a narrow Q2 signal; evening transitions open no Q3/S3 target. |
| `donor_swap_or_slice_selector` | `closed` | `False` | `reports/post_id342_q3s3_slice_selector_anatomy_20260624.json` | Stable donors and fold-internal slice selector fail; failed-frontier donors are public traps. |
| `processed_sleep_proxy_and_baseline` | `closed` | `False` | `reports/post_id342_processed_sleep_temporal_deviation_gate_20260624.json` | Processed sleep proxies, lag features, and personal baseline deviations do not open Q3/S3. |
| `s4s3_processed_sleep_fragmentation` | `closed` | `False` | `reports/post_id342_s4s3_sleep_fragmentation_gate_20260625.json` | Focused WASO/SOL processed sleep-fragmentation features pass no S4/S3 row. |
| `direct_raw_sleep_event_reconstruction` | `closed` | `False` | `reports/post_id342_raw_sleep_event_reconstruction_feasibility_20260625.json` | Public raw schema has no direct sleep-stage, SOL, or WASO event columns. |
| `sequence_hr_light_event_reconstruction` | `closed` | `False` | `reports/post_id342_sequence_hr_light_event_gate_20260625.json` | Raw HR/light/screen event proxies open only a tiny S4 signal; S3 remains closed and Public proxy is about 0.596. |
| `mis_lstm_block_sequence_feasibility` | `closed` | `False` | `reports/post_id342_mis_lstm_block_sequence_feasibility_gate_20260625.json` | MIS-LSTM-style block-sequence local feasibility gate must pass before any Kaggle runner can open. |
| `subject_prior_or_temperature_calibration` | `closed` | `False` | `reports/post_id342_probability_temperature_gate_20260624.json` | Subject-prior forcing and temperature/shift calibration worsen ID342 under primary lenses. |
