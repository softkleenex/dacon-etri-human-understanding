#!/usr/bin/env python3
"""Post-ID342 remaining route synthesis.

This report-only diagnostic supersedes the earlier source-priority queue after
the raw-temporal, sleep-proxy, subject-prior, and temperature gates have run. It
summarizes which post-ID342 no-submit lanes are closed and what remains
score-relevant for the Public 0.55 objective.

It creates no prediction CSV, feature parquet, base prediction artifact,
notebook-history artifact, submission-log row, Kaggle push/run, or DACON
submission.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_remaining_route_synthesis_20260625.json"
KST = timezone(timedelta(hours=9))

ID342_PUBLIC = 0.5960832519
PUBLIC_055 = 0.55


REPORT_PATHS = {
    "research_gap_initial": REPORTS_DIR / "post_id342_research_gap_and_source_priority_20260624.json",
    "raw_temporal": REPORTS_DIR / "post_id342_raw_temporal_q3s3_sequence_image_smoke_20260624.json",
    "sequence_summary": REPORTS_DIR / "post_id342_sequence_summary_distance_kernel_smoke_20260624.json",
    "sleep_boundary": REPORTS_DIR / "post_id342_sleep_boundary_q3s3_rescue_smoke_20260624.json",
    "tiny_temporal_cnn": REPORTS_DIR / "post_id342_tiny_temporal_cnn_q3s3_smoke_20260624.json",
    "public_response": REPORTS_DIR / "post_id342_public_response_geometry_20260624.json",
    "feature_similarity_decomposition": REPORTS_DIR
    / "post_id342_feature_similarity_source_decomposition_20260624.json",
    "feature_similarity_artifact": REPORTS_DIR / "post_id342_feature_similarity_artifact_gate_20260624.json",
    "paper_aligned_featuresim": REPORTS_DIR / "post_id342_paper_aligned_feature_similarity_rebuild_gate_20260624.json",
    "past_only_temporal": REPORTS_DIR / "post_id342_past_only_temporal_recency_gate_20260624.json",
    "cross_target": REPORTS_DIR / "post_id342_cross_target_consistency_gate_20260624.json",
    "material_source_rescan": REPORTS_DIR / "post_id342_registered_material_source_rescan_20260624.json",
    "material_fusion": REPORTS_DIR / "post_id342_registered_material_fusion_gate_20260624.json",
    "raw_subject_state_feature_gate": REPORTS_DIR / "post_id342_raw_subject_state_feature_gate_20260624.json",
    "raw_sequence_representation_gate": REPORTS_DIR / "post_id342_s3q3_raw_sequence_representation_gate_20260624.json",
    "self_supervised_subject_state_intake": REPORTS_DIR
    / "post_id342_self_supervised_subject_state_remote_intake_20260624.json",
    "missingness_subject_state_intake": REPORTS_DIR
    / "post_id342_missingness_subject_state_remote_intake_20260624.json",
    "public_gap_risk_lens": REPORTS_DIR / "post_id342_public_gap_risk_lens_20260624.json",
    "public_gap_penalty_proxy": REPORTS_DIR / "post_id342_public_gap_penalty_proxy_redesign_20260625.json",
    "validation_weighted_pseudo_public": REPORTS_DIR
    / "post_id342_validation_weighted_pseudo_public_split_audit_20260625.json",
    "solar_seasonal_holiday_prior": REPORTS_DIR / "post_id342_solar_seasonal_holiday_prior_gate_20260625.json",
    "kma_weather_context": REPORTS_DIR / "post_id342_kma_weather_context_gate_20260625.json",
    "public_data_api_smoke": REPORTS_DIR / "public_data_weather_air_api_smoke_20260625.json",
    "airkorea_2024_hourly_audit": REPORTS_DIR / "airkorea_2024_hourly_zip_audit_20260625.json",
    "airkorea_2024_context": REPORTS_DIR / "post_id342_airkorea_2024_context_gate_20260625.json",
    "external_etri_sleep_survey_prior": REPORTS_DIR
    / "post_id342_external_etri_sleep_survey_prior_20260625.json",
    "cross_source_split_stability": REPORTS_DIR / "post_id342_cross_source_split_stability_audit_20260625.json",
    "s3_phone_gps_interaction": REPORTS_DIR / "post_id342_s3_phone_gps_interaction_gate_20260625.json",
    "final_day_independent_signal": REPORTS_DIR / "post_id342_final_day_independent_signal_sweep_20260625.json",
    "stable_envelope_multisource": REPORTS_DIR / "post_id342_stable_envelope_multisource_optimizer_20260625.json",
    "target_reallocation": REPORTS_DIR / "post_id342_target_reallocation_optimizer_20260625.json",
    "s4s3_sleep_fragmentation": REPORTS_DIR / "post_id342_s4s3_sleep_fragmentation_gate_20260625.json",
    "raw_sleep_event_feasibility": REPORTS_DIR / "post_id342_raw_sleep_event_reconstruction_feasibility_20260625.json",
    "sequence_hr_light_event": REPORTS_DIR / "post_id342_sequence_hr_light_event_gate_20260625.json",
    "mis_lstm_block_sequence": REPORTS_DIR / "post_id342_mis_lstm_block_sequence_feasibility_gate_20260625.json",
    "public_alignment": REPORTS_DIR / "post_id342_public_alignment_validation_lens_20260624.json",
    "stable_lens_triage": REPORTS_DIR / "post_id342_stable_lens_existing_candidate_triage_20260624.json",
    "env_social_routine": REPORTS_DIR / "post_id342_env_social_routine_v2_id342_gate_20260624.json",
    "evening_transition": REPORTS_DIR / "post_id342_evening_state_transition_gate_20260624.json",
    "target_swap": REPORTS_DIR / "post_id342_public_scored_target_swap_gate_20260624.json",
    "slice_selector": REPORTS_DIR / "post_id342_q3s3_slice_selector_anatomy_20260624.json",
    "lightweight_pca": REPORTS_DIR / "post_id342_lightweight_subject_state_pca_gate_20260624.json",
    "processed_sleep_proxy": REPORTS_DIR / "post_id342_processed_sleep_proxy_gate_20260624.json",
    "processed_sleep_temporal_deviation": REPORTS_DIR
    / "post_id342_processed_sleep_temporal_deviation_gate_20260624.json",
    "subject_prior_calibration": REPORTS_DIR / "post_id342_subject_prior_calibration_gate_20260624.json",
    "probability_temperature": REPORTS_DIR / "post_id342_probability_temperature_gate_20260624.json",
    "kaggle_remote_summary": REPORTS_DIR / "post_id341_kaggle_remote_result_summary_20260618.json",
}


ROUTE_DEFINITIONS = [
    {
        "route": "existing_candidate_shelf",
        "report_key": "stable_lens_triage",
        "decision_keys": ["stable_lens_review_pass_count"],
        "closed_when_zero": True,
        "status": "closed",
        "reason": "Existing OOF candidates pass no stable-lens review after excluding ID287/ID329 traps.",
    },
    {
        "route": "public_gap_shape_lens",
        "report_key": "public_gap_risk_lens",
        "decision_keys": ["public055_shape_lens_pass_count"],
        "closed_when_zero": True,
        "status": "closed",
        "reason": "Stable prediction-shape lens finds no Public-0.55 shelf row and warns against larger frontier movement.",
    },
    {
        "route": "public_gap_penalty_proxy_redesign",
        "report_key": "public_gap_penalty_proxy",
        "decision_keys": ["credible_public055_penalty_proxy_pass_count"],
        "closed_when_zero": True,
        "status": "closed",
        "reason": "Shape-excess penalty proxy finds no credible Public-0.55 OOF shelf row.",
    },
    {
        "route": "validation_weighted_pseudo_public_split",
        "report_key": "validation_weighted_pseudo_public",
        "decision_keys": ["validation_weighted_pseudo_public_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Target-weighted pseudo-public validation audit opens no credible Public-0.55 or review-threshold row.",
    },
    {
        "route": "solar_seasonal_holiday_prior",
        "report_key": "solar_seasonal_holiday_prior",
        "decision_keys": ["solar_seasonal_holiday_prior_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "External date-derived solar/seasonal/Korean-holiday priors show only unstable S4 movement.",
    },
    {
        "route": "kma_asos_weather_context",
        "report_key": "kma_weather_context",
        "decision_keys": ["weather_context_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Historical Daejeon ASOS weather context shows only a tiny unstable S4 signal near Public 0.596.",
    },
    {
        "route": "kma_life_airkorea_api_feasibility",
        "report_key": "public_data_api_smoke",
        "decision_keys": ["non_asos_approved_api_2024_historical_feature_source_available"],
        "closed_when_false": True,
        "status": "closed",
        "reason": (
            "KMA living-weather and the approved AirKorea information API work for current/recent data, "
            "but do not expose a date-bounded 2024 historical feature source."
        ),
    },
    {
        "route": "airkorea_2024_historical_context",
        "report_key": "airkorea_2024_context",
        "decision_keys": ["airkorea_2024_context_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Downloaded 2024 Daejeon AirKorea context opens only a tiny unstable S4 hint near Public 0.596.",
    },
    {
        "route": "external_etri_sleep_survey_prior",
        "report_key": "external_etri_sleep_survey_prior",
        "decision_keys": ["external_prior_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Historical ETRI sleep/survey prior supports Q2 fatigue coupling but opens no direct Q3/S3/S4 route.",
    },
    {
        "route": "cross_source_split_stability",
        "report_key": "cross_source_split_stability",
        "decision_keys": ["cross_source_split_stability_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Weak-source split audit finds only a tiny S3 stable micro-alignment and unstable S4 split behavior.",
    },
    {
        "route": "s3_phone_gps_interaction",
        "report_key": "s3_phone_gps_interaction",
        "decision_keys": ["s3_phone_gps_interaction_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Focused S3 phone/GPS concat improves all splits but remains near Public 0.596 and far from review thresholds.",
    },
    {
        "route": "final_day_independent_signal_sweep",
        "report_key": "final_day_independent_signal",
        "decision_keys": ["final_day_independent_signal_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Final-day S3 temporal, S4 sleep-solar, and compact joint-state sweep opens no Public-0.58 review row.",
    },
    {
        "route": "stable_envelope_multisource_recombination",
        "report_key": "stable_envelope_multisource",
        "decision_keys": ["public055_envelope_rows"],
        "closed_when_zero": True,
        "status": "closed",
        "reason": "Nonlegacy multi-source ID342-convex recombination stays above Public 0.55 under the stable-gap projection.",
    },
    {
        "route": "strict_target_reallocation",
        "report_key": "target_reallocation",
        "decision_keys": ["public055_envelope_rows"],
        "closed_when_zero": True,
        "status": "closed",
        "reason": "Target-wise strict existing-source reallocation improves the envelope ceiling only to about Public 0.566.",
    },
    {
        "route": "raw_temporal_q3s3_sequence_image",
        "report_key": "raw_temporal",
        "decision_keys": ["final_raw_temporal_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Raw temporal sequence-image smoke selected no Q3/S3 target under ID342 gates.",
    },
    {
        "route": "sequence_summary_distance_kernel",
        "report_key": "sequence_summary",
        "decision_keys": ["final_sequence_distance_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Half-hour sequence summaries and distance kernels failed the final gate.",
    },
    {
        "route": "sleep_boundary_tiny_cnn",
        "report_key": "tiny_temporal_cnn",
        "decision_keys": ["final_tiny_temporal_cnn_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Sleep-boundary/SOL and tiny temporal-CNN local proxies selected no Q3/S3 target.",
    },
    {
        "route": "feature_similarity_alpha_tuning",
        "report_key": "feature_similarity_artifact",
        "decision_keys": ["artifact_gate_pass_count"],
        "closed_when_zero": True,
        "status": "closed",
        "reason": "Post-public ID342 alpha rows fail strict artifact review and remain far from Public 0.55.",
    },
    {
        "route": "paper_aligned_feature_similarity_rebuild",
        "report_key": "paper_aligned_featuresim",
        "decision_keys": ["paper_aligned_feature_similarity_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Domain-restricted same-subject feature-similarity rebuild loses to ID342.",
    },
    {
        "route": "registered_material_single_or_fusion",
        "report_key": "material_fusion",
        "decision_keys": ["local_gate_pass_count"],
        "closed_when_zero": True,
        "status": "closed",
        "reason": "Registered material matrices and their fusion pass no local gate.",
    },
    {
        "route": "raw_subject_state_daily_windows",
        "report_key": "raw_subject_state_feature_gate",
        "decision_keys": ["local_gate_pass_count"],
        "closed_when_zero": True,
        "status": "closed",
        "reason": "Full raw daily/window subject-state scan has high coverage but passes no ID342/Public-0.55 gate.",
    },
    {
        "route": "raw_sequence_representation_remote",
        "report_key": "raw_sequence_representation_gate",
        "decision_keys": ["final_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Kaggle raw sequence representation run passed no Q3/S3 row and produced no promotable evidence.",
    },
    {
        "route": "self_supervised_subject_state_embedding",
        "report_key": "self_supervised_subject_state_intake",
        "decision_keys": ["remote_intake_ready"],
        "closed_when_true": True,
        "status": "closed",
        "reason": "Label-free subject-state execution is safe research evidence only; downstream gates did not open a route.",
    },
    {
        "route": "missingness_cross_modal_subject_state_embedding",
        "report_key": "missingness_subject_state_intake",
        "decision_keys": ["stable_gate_pass_count"],
        "closed_when_zero": True,
        "status": "closed",
        "reason": "Full missingness-aware cross-modal embedding scan had strong coverage but zero stable passes.",
    },
    {
        "route": "env_social_evening_lightweight_sources",
        "report_key": "evening_transition",
        "decision_keys": ["final_evening_state_transition_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Env/social/routine opens only a narrow Q2 signal; evening transitions open no Q3/S3 target.",
    },
    {
        "route": "donor_swap_or_slice_selector",
        "report_key": "slice_selector",
        "decision_keys": ["selector_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Stable donors and fold-internal slice selector fail; failed-frontier donors are public traps.",
    },
    {
        "route": "processed_sleep_proxy_and_baseline",
        "report_key": "processed_sleep_temporal_deviation",
        "decision_keys": ["processed_sleep_temporal_deviation_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Processed sleep proxies, lag features, and personal baseline deviations do not open Q3/S3.",
    },
    {
        "route": "s4s3_processed_sleep_fragmentation",
        "report_key": "s4s3_sleep_fragmentation",
        "decision_keys": ["s4s3_sleep_fragmentation_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Focused WASO/SOL processed sleep-fragmentation features pass no S4/S3 row.",
    },
    {
        "route": "direct_raw_sleep_event_reconstruction",
        "report_key": "raw_sleep_event_feasibility",
        "decision_keys": ["direct_sleep_event_route_available"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Public raw schema has no direct sleep-stage, SOL, or WASO event columns.",
    },
    {
        "route": "sequence_hr_light_event_reconstruction",
        "report_key": "sequence_hr_light_event",
        "decision_keys": ["final_sequence_hr_light_event_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Raw HR/light/screen event proxies open only a tiny S4 signal; S3 remains closed and Public proxy is about 0.596.",
    },
    {
        "route": "mis_lstm_block_sequence_feasibility",
        "report_key": "mis_lstm_block_sequence",
        "decision_keys": ["final_mis_lstm_block_sequence_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "MIS-LSTM-style block-sequence local feasibility gate must pass before any Kaggle runner can open.",
    },
    {
        "route": "subject_prior_or_temperature_calibration",
        "report_key": "probability_temperature",
        "decision_keys": ["temperature_shift_gate_pass"],
        "closed_when_false": True,
        "status": "closed",
        "reason": "Subject-prior forcing and temperature/shift calibration worsen ID342 under primary lenses.",
    },
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def report_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}, "error": "missing"}
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": rel(path),
            "exists": True,
            "parse_ok": False,
            "decision": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": rel(path),
        "exists": True,
        "parse_ok": True,
        "diagnostic": payload.get("diagnostic"),
        "decision": payload.get("decision", {}),
        "error": None,
    }


def get_first_present(decision: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in decision:
            return decision[key]
    return None


def route_rows(report_statuses: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for definition in ROUTE_DEFINITIONS:
        report_key = definition["report_key"]
        status_row = report_statuses.get(report_key, {}) if report_key else {}
        decision = status_row.get("decision", {})
        decision_value = get_first_present(decision, definition["decision_keys"])
        closed_by_value = False
        if definition.get("closed_when_zero"):
            closed_by_value = decision_value == 0
        if definition.get("closed_when_false"):
            closed_by_value = decision_value is False
        if definition.get("closed_when_true"):
            closed_by_value = decision_value is True
        status_value = definition["status"]
        if report_key and not status_row.get("exists"):
            status_value = "missing_report"
        elif report_key and not status_row.get("parse_ok"):
            status_value = "parse_error"
        elif definition["status"] == "closed" and not closed_by_value:
            status_value = "needs_review"
        rows.append(
            {
                "route": definition["route"],
                "status": status_value,
                "report_key": report_key,
                "report_path": status_row.get("path"),
                "decision_keys": definition["decision_keys"],
                "decision_value": decision_value,
                "reason": definition["reason"],
            }
        )
    return rows


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    statuses = {key: report_status(path) for key, path in REPORT_PATHS.items()}
    rows = route_rows(statuses)
    closed_count = sum(row["status"] == "closed" for row in rows)
    missing_or_review = [row for row in rows if row["status"] in {"missing_report", "parse_error", "needs_review"}]
    deferred = [row for row in rows if row["status"].startswith("deferred")]
    all_routes_closed = closed_count == len(rows) and not missing_or_review and not deferred
    remote_decision = statuses["kaggle_remote_summary"].get("decision", {})
    missingness_decision = statuses["missingness_subject_state_intake"].get("decision", {})
    probability_decision = statuses["probability_temperature"].get("decision", {})
    stable_envelope_decision = statuses["stable_envelope_multisource"].get("decision", {})
    target_reallocation_decision = statuses["target_reallocation"].get("decision", {})
    sequence_event_decision = statuses["sequence_hr_light_event"].get("decision", {})
    mis_lstm_decision = statuses["mis_lstm_block_sequence"].get("decision", {})
    penalty_proxy_decision = statuses["public_gap_penalty_proxy"].get("decision", {})
    validation_weighted_decision = statuses["validation_weighted_pseudo_public"].get("decision", {})
    solar_holiday_decision = statuses["solar_seasonal_holiday_prior"].get("decision", {})
    kma_weather_decision = statuses["kma_weather_context"].get("decision", {})
    public_data_api_decision = statuses["public_data_api_smoke"].get("decision", {})
    airkorea_2024_decision = statuses["airkorea_2024_hourly_audit"].get("decision", {})
    airkorea_context_decision = statuses["airkorea_2024_context"].get("decision", {})
    external_etri_prior_decision = statuses["external_etri_sleep_survey_prior"].get("decision", {})
    cross_source_decision = statuses["cross_source_split_stability"].get("decision", {})
    s3_phone_gps_decision = statuses["s3_phone_gps_interaction"].get("decision", {})
    final_day_decision = statuses["final_day_independent_signal"].get("decision", {})
    public_gap = ID342_PUBLIC - PUBLIC_055
    decision = {
        "current_public_anchor_id": 342,
        "current_public_score": ID342_PUBLIC,
        "target_public_score": PUBLIC_055,
        "gap_to_public_055": public_gap,
        "closed_route_count": int(closed_count),
        "route_count": int(len(rows)),
        "all_registered_routes_closed": all_routes_closed,
        "missing_or_review_route_count": int(len(missing_or_review)),
        "deferred_route_count": int(len(deferred)),
        "missingness_best_projected_public_proxy": missingness_decision.get("best_projected_public_proxy"),
        "missingness_stable_gate_pass_count": missingness_decision.get("stable_gate_pass_count"),
        "stable_envelope_best_projected_public": stable_envelope_decision.get("best_projected_public_at_stable_gap"),
        "target_reallocation_best_projected_public": target_reallocation_decision.get(
            "best_projected_public_at_stable_gap"
        ),
        "penalty_proxy_credible_public055_pass_count": penalty_proxy_decision.get(
            "credible_public055_penalty_proxy_pass_count"
        ),
        "penalty_proxy_best_candidate": penalty_proxy_decision.get("best_penalty_proxy_candidate"),
        "validation_weighted_gate_pass": validation_weighted_decision.get(
            "validation_weighted_pseudo_public_gate_pass"
        ),
        "validation_weighted_credible_public055_pass_count": validation_weighted_decision.get(
            "credible_weighted_public055_pass_count"
        ),
        "validation_weighted_credible_review058_pass_count": validation_weighted_decision.get(
            "credible_weighted_review058_pass_count"
        ),
        "validation_weighted_best_credible_shape_candidate": validation_weighted_decision.get(
            "best_credible_shape_candidate"
        ),
        "solar_seasonal_holiday_gate_pass": solar_holiday_decision.get("solar_seasonal_holiday_prior_gate_pass"),
        "solar_seasonal_holiday_selected_targets": solar_holiday_decision.get("selected_targets"),
        "solar_seasonal_holiday_best_row": solar_holiday_decision.get("best_row"),
        "kma_weather_context_gate_pass": kma_weather_decision.get("weather_context_gate_pass"),
        "kma_weather_best_candidate": kma_weather_decision.get("best_candidate"),
        "kma_weather_best_target": kma_weather_decision.get("best_target"),
        "kma_weather_best_projected_public_proxy": kma_weather_decision.get("best_projected_public_proxy"),
        "kma_weather_review_gate_pass_count": kma_weather_decision.get("review_gate_pass_count"),
        "public_data_api_all_three_checked": public_data_api_decision.get("all_three_service_families_smoke_checked"),
        "kma_life_2024_historical_feature_source_available": public_data_api_decision.get(
            "kma_life_2024_historical_feature_source_available"
        ),
        "airkorea_api_access_working": public_data_api_decision.get("airkorea_api_access_working"),
        "airkorea_approved_api_2024_historical_feature_source_available": public_data_api_decision.get(
            "airkorea_approved_info_api_2024_historical_feature_source_available"
        ),
        "non_asos_approved_api_2024_historical_feature_source_available": public_data_api_decision.get(
            "non_asos_approved_api_2024_historical_feature_source_available"
        ),
        "airkorea_2024_historical_archive_available": airkorea_2024_decision.get(
            "airkorea_2024_zip_available"
        ),
        "airkorea_2024_competition_window_coverage_ok": airkorea_2024_decision.get(
            "competition_window_coverage_ok"
        ),
        "airkorea_2024_context_gate_pass": airkorea_context_decision.get("airkorea_2024_context_gate_pass"),
        "airkorea_2024_best_candidate": airkorea_context_decision.get("best_candidate"),
        "airkorea_2024_best_target": airkorea_context_decision.get("best_target"),
        "airkorea_2024_best_projected_public_proxy": airkorea_context_decision.get(
            "best_projected_public_proxy"
        ),
        "airkorea_2024_review_gate_pass_count": airkorea_context_decision.get("review_gate_pass_count"),
        "airkorea_2024_gate_pending": bool(
            airkorea_2024_decision.get("airkorea_2024_zip_available")
            and airkorea_2024_decision.get("competition_window_coverage_ok")
            and airkorea_context_decision.get("airkorea_2024_context_gate_pass") is None
        ),
        "external_etri_prior_gate_pass": external_etri_prior_decision.get("external_prior_gate_pass"),
        "external_etri_q2_fatigue_sleep_coupling_supported": external_etri_prior_decision.get(
            "q2_fatigue_sleep_coupling_supported"
        ),
        "external_etri_q3_stress_sleep_coupling_supported": external_etri_prior_decision.get(
            "q3_stress_sleep_coupling_supported"
        ),
        "external_etri_direct_2026_feature_route_available": external_etri_prior_decision.get(
            "direct_2026_feature_route_available"
        ),
        "cross_source_split_stability_gate_pass": cross_source_decision.get(
            "cross_source_split_stability_gate_pass"
        ),
        "cross_source_review_signal_targets": cross_source_decision.get("review_signal_targets"),
        "cross_source_stable_alignment_targets": cross_source_decision.get("stable_alignment_targets"),
        "cross_source_target_statuses": cross_source_decision.get("target_statuses"),
        "cross_source_best_public_proxy_by_focus_target": cross_source_decision.get(
            "best_public_proxy_by_focus_target"
        ),
        "s3_phone_gps_interaction_gate_pass": s3_phone_gps_decision.get("s3_phone_gps_interaction_gate_pass"),
        "s3_phone_gps_best_candidate": s3_phone_gps_decision.get("best_candidate"),
        "s3_phone_gps_best_projected_public_proxy": s3_phone_gps_decision.get("best_projected_public_proxy"),
        "s3_phone_gps_local_gate_pass_count": s3_phone_gps_decision.get("local_gate_pass_count"),
        "final_day_independent_signal_gate_pass": final_day_decision.get(
            "final_day_independent_signal_gate_pass"
        ),
        "final_day_best_candidate": final_day_decision.get("best_candidate"),
        "final_day_best_projected_public_proxy": final_day_decision.get("best_projected_public_proxy"),
        "final_day_local_gate_pass_count": final_day_decision.get("local_gate_pass_count"),
        "final_day_review_gate_pass_count": final_day_decision.get("review_gate_pass_count"),
        "sequence_event_selected_targets": sequence_event_decision.get("selected_targets"),
        "sequence_event_projected_public_proxy": sequence_event_decision.get("projected_public_proxy_from_id342"),
        "mis_lstm_selected_targets": mis_lstm_decision.get("selected_targets"),
        "mis_lstm_projected_public_proxy": mis_lstm_decision.get("projected_public_proxy_from_id342"),
        "kaggle_upstream_complete_no_wait_state": True,
        "remote_promotable_evidence": bool(
            remote_decision.get("public_058_evidence_present")
            or remote_decision.get("beats_id178_public_anchor_evidence_present")
        ),
        "probability_submission_contract_confirmed": probability_decision.get(
            "submission_probability_contract_confirmed"
        ),
        "create_submission_artifact_now": False,
        "modify_submission_log_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "dacon_submit_meaningful_now": False,
        "next_score_relevant_action": (
            "All registered post-ID342 same-family, existing-source, sleep-proxy, raw sequence, and heavy "
            "subject-state routes are closed, the validation-weighted pseudo-public audit opens no current "
            "candidate route, the first external solar/holiday prior gate is unstable, the KMA ASOS weather "
            "context gate finds only a tiny S4 signal near Public 0.596, the approved KMA living-weather and "
            "AirKorea information APIs do not expose a 2024 historical feature source, and the separate "
            "downloaded AirKorea 2024 gate also opens no Public-0.58 review row. Historical ETRI sleep/survey "
            "priors support Q2 fatigue-to-sleep coupling, but not a direct Q3/S3/S4 feature route. The "
            "cross-source split audit finds only a too-small S3 micro-alignment, a focused S3 phone/GPS interaction gate "
            "remains near Public 0.596, and the final-day independent-source sweep opens no Public-0.58 review "
            "row. Do not submit or launch another same-family Kaggle run; the next score work must use a "
            "materially different Q3/S3/S4 source mechanism."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_remaining_route_synthesis",
        "purpose": "Synthesize current post-ID342 no-submit evidence and identify remaining score-relevant routes.",
        "report_statuses": statuses,
        "route_rows": rows,
        "decision": decision,
        "no_predictions_created": True,
        "no_new_feature_matrix_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action": True,
        "no_dacon_submission": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 remaining route synthesis",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Current public anchor: `ID{decision['current_public_anchor_id']}` / `{decision['current_public_score']:.10f}`",
        f"- Gap to Public 0.55: `{decision['gap_to_public_055']:+.10f}`",
        f"- Closed routes: `{decision['closed_route_count']}/{decision['route_count']}`",
        f"- All registered routes closed: `{decision['all_registered_routes_closed']}`",
        f"- Missingness heavy stable passes: `{decision['missingness_stable_gate_pass_count']}`",
        f"- Missingness best public proxy: `{decision['missingness_best_projected_public_proxy']}`",
        f"- Stable-envelope best public proxy: `{decision['stable_envelope_best_projected_public']}`",
        f"- Target-reallocation best public proxy: `{decision['target_reallocation_best_projected_public']}`",
        f"- Penalty-proxy credible 0.55 passes: `{decision['penalty_proxy_credible_public055_pass_count']}`",
        f"- Validation-weighted gate pass: `{decision['validation_weighted_gate_pass']}`",
        f"- Validation-weighted credible 0.55 passes: `{decision['validation_weighted_credible_public055_pass_count']}`",
        f"- Validation-weighted credible 0.58 passes: `{decision['validation_weighted_credible_review058_pass_count']}`",
        f"- Solar/holiday gate pass: `{decision['solar_seasonal_holiday_gate_pass']}`",
        f"- Solar/holiday selected targets: `{decision['solar_seasonal_holiday_selected_targets']}`",
        f"- KMA weather gate pass: `{decision['kma_weather_context_gate_pass']}`",
        f"- KMA weather best target/proxy: `{decision['kma_weather_best_target']}` / `{decision['kma_weather_best_projected_public_proxy']}`",
        f"- Public-data API smoke all three checked: `{decision['public_data_api_all_three_checked']}`",
        f"- KMA living-weather 2024 source available: `{decision['kma_life_2024_historical_feature_source_available']}`",
        f"- AirKorea API access working: `{decision['airkorea_api_access_working']}`",
        f"- AirKorea approved 2024 source available: `{decision['airkorea_approved_api_2024_historical_feature_source_available']}`",
        f"- AirKorea downloaded 2024 archive available: `{decision['airkorea_2024_historical_archive_available']}`",
        f"- AirKorea downloaded 2024 coverage OK: `{decision['airkorea_2024_competition_window_coverage_ok']}`",
        f"- AirKorea downloaded 2024 gate pass: `{decision['airkorea_2024_context_gate_pass']}`",
        f"- AirKorea downloaded 2024 best target/proxy: `{decision['airkorea_2024_best_target']}` / `{decision['airkorea_2024_best_projected_public_proxy']}`",
        f"- AirKorea downloaded 2024 gate pending: `{decision['airkorea_2024_gate_pending']}`",
        f"- External ETRI prior gate pass: `{decision['external_etri_prior_gate_pass']}`",
        f"- External ETRI Q2 fatigue sleep coupling: `{decision['external_etri_q2_fatigue_sleep_coupling_supported']}`",
        f"- External ETRI Q3 stress sleep coupling: `{decision['external_etri_q3_stress_sleep_coupling_supported']}`",
        f"- External ETRI direct 2026 feature route: `{decision['external_etri_direct_2026_feature_route_available']}`",
        f"- Cross-source split gate pass: `{decision['cross_source_split_stability_gate_pass']}`",
        f"- Cross-source review targets: `{decision['cross_source_review_signal_targets']}`",
        f"- Cross-source stable-alignment targets: `{decision['cross_source_stable_alignment_targets']}`",
        f"- S3 phone/GPS gate pass: `{decision['s3_phone_gps_interaction_gate_pass']}`",
        f"- S3 phone/GPS best public proxy: `{decision['s3_phone_gps_best_projected_public_proxy']}`",
        f"- Final-day independent sweep gate pass: `{decision['final_day_independent_signal_gate_pass']}`",
        f"- Final-day best public proxy: `{decision['final_day_best_projected_public_proxy']}`",
        f"- Sequence-event selected targets: `{decision['sequence_event_selected_targets']}`",
        f"- Sequence-event public proxy: `{decision['sequence_event_projected_public_proxy']}`",
        f"- MIS-LSTM block selected targets: `{decision['mis_lstm_selected_targets']}`",
        f"- MIS-LSTM block public proxy: `{decision['mis_lstm_projected_public_proxy']}`",
        f"- Kaggle upstream wait state: `{not decision['kaggle_upstream_complete_no_wait_state']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_score_relevant_action']}",
        "",
        "## Routes",
        "",
        "| route | status | decision value | report | reason |",
        "|---|---|---:|---|---|",
    ]
    for row in report["route_rows"]:
        report_path = row["report_path"] or ""
        lines.append(
            f"| `{row['route']}` | `{row['status']}` | `{row['decision_value']}` | `{report_path}` | {row['reason']} |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.output)
    decision = report["decision"]
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(f"Closed routes: {decision['closed_route_count']}/{decision['route_count']}")


if __name__ == "__main__":
    main()
