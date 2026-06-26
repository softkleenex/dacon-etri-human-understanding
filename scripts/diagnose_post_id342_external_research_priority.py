#!/usr/bin/env python3
"""Post-ID342 external research priority refresh.

This report-only diagnostic incorporates external research on ETRI lifelog
data, multimodal time series, and wearable/self-supervised learning into the
remaining post-ID342 route map after the registered local, remote, and
validation-proxy lanes have closed.

It does not read raw parquet payloads, fit embeddings, write feature matrices,
train models, create predictions, create submission artifacts, edit the
submission log, push/run/pull Kaggle, or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_external_research_priority_20260625.json"
KST = timezone(timedelta(hours=9))

REPORT_PATHS = {
    "morning_closeout": REPORTS_DIR / "post_id342_morning_closeout_20260624.json",
    "remaining_route_synthesis": REPORTS_DIR / "post_id342_remaining_route_synthesis_20260625.json",
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
    "sequence_hr_light_event": REPORTS_DIR / "post_id342_sequence_hr_light_event_gate_20260625.json",
    "mis_lstm_block_sequence": REPORTS_DIR / "post_id342_mis_lstm_block_sequence_feasibility_gate_20260625.json",
    "missingness_subject_state_intake": REPORTS_DIR
    / "post_id342_missingness_subject_state_remote_intake_20260624.json",
    "self_supervised_research_packet": REPORTS_DIR / "post_id342_self_supervised_route_research_packet_20260624.json",
    "domain_token_contract": REPORTS_DIR / "post_id342_subject_state_domain_token_contract_20260624.json",
    "raw_schema_contract": REPORTS_DIR / "post_id342_subject_state_raw_schema_contract_20260624.json",
    "heavy_success_thresholds": REPORTS_DIR / "post_id342_subject_state_heavy_success_thresholds_20260624.json",
}

FORBIDDEN_ACTIONS_NOW = {
    "read_raw_parquet_payloads": False,
    "write_token_cache": False,
    "write_feature_matrix": False,
    "fit_embedding": False,
    "train_model": False,
    "create_predictions": False,
    "create_submission_artifact": False,
    "modify_submission_log": False,
    "push_kaggle": False,
    "run_kaggle": False,
    "pull_kaggle_output": False,
    "submit_dacon": False,
}

EXTERNAL_SOURCES = [
    {
        "key": "etri_lifelog_2024",
        "title": "Understanding Human Daily Experience Through Continuous Sensing: ETRI Lifelog Dataset 2024",
        "url": "https://arxiv.org/html/2508.03698v1",
        "source_type": "dataset_paper",
        "takeaway": (
            "ETRI 2024 is explicitly built around passive 24-hour sensing from smartphones, smartwatches, "
            "sleep sensors, and self-reports for sleep quality and stress prediction."
        ),
        "implication": "The remaining route should model day-state and sleep-transition context jointly.",
    },
    {
        "key": "dacon_data_page",
        "title": "The 4th ETRI Human Understanding AI Paper Challenge data page",
        "url": "https://dacon.io/en/competitions/official/236468/data",
        "source_type": "competition_page",
        "takeaway": (
            "The official data page lists 12 public lifelog streams and frames the task as sleep health, "
            "fatigue, and stress prediction from daily-life records."
        ),
        "implication": "A future runner should preserve all 12 modalities and their missingness patterns.",
    },
    {
        "key": "pixleepflow",
        "title": "PixleepFlow: A Pixel-Based Lifelog Framework for Predicting Sleep Quality and Stress Level",
        "url": "https://arxiv.org/html/2502.17469v2",
        "source_type": "etri_competition_method",
        "takeaway": (
            "PixleepFlow treats synchronized lifelog streams as composite images and targets sleep/stress "
            "jointly, reinforcing the need for cross-channel state representations."
        ),
        "implication": "Image layout is useful as a representation prior, but local ID342 smokes say it is not enough alone.",
    },
    {
        "key": "mis_lstm",
        "title": "MIS-LSTM: Multichannel Image-Sequence LSTM for Sleep Quality and Stress Prediction",
        "url": "https://arxiv.org/abs/2509.11232",
        "source_type": "etri_competition_method",
        "takeaway": (
            "MIS-LSTM partitions continuous sensor streams into multi-hour blocks, renders them as multi-channel "
            "images, encodes sparse discrete events separately with 1D-CNN, then aggregates block embeddings with LSTM."
        ),
        "implication": (
            "A future image/sequence attempt should not replay stacked single images; it needs block-level temporal "
            "aggregation plus discrete-event handling and uncertainty checks."
        ),
    },
    {
        "key": "multi_modal_ts_survey",
        "title": "Multi-modal Time Series Analysis: A Tutorial and Survey",
        "url": "https://arxiv.org/html/2503.13709v1",
        "source_type": "survey",
        "takeaway": (
            "The survey emphasizes modality heterogeneity, alignment, noise, and missing modalities as central "
            "multi-modal time-series challenges."
        ),
        "implication": "The next design should use cross-modal alignment/fusion with explicit robustness checks.",
    },
    {
        "key": "crossl",
        "title": "CroSSL: Cross-modal Self-Supervised Learning for Time-series through Latent Masking",
        "url": "https://arxiv.org/abs/2307.16847",
        "source_type": "method_paper",
        "takeaway": (
            "CroSSL masks modality-specific latent embeddings and aggregates them into a global representation, "
            "while handling missing modalities without negative-pair sampling."
        ),
        "implication": "This is a strong template for phone/watch/sleep-context modality dropout and fusion.",
    },
    {
        "key": "lsm2_aim",
        "title": "LSM-2: Learning from Incomplete Wearable Sensor Data",
        "url": "https://arxiv.org/html/2506.05321v1",
        "source_type": "method_paper",
        "takeaway": (
            "LSM-2/AIM learns directly from inherited real-world missingness plus artificial masks instead of "
            "requiring explicit imputation."
        ),
        "implication": "Missingness should be a first-class token, not a preprocessing nuisance to fill away.",
    },
    {
        "key": "time_mae",
        "title": "TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders",
        "url": "https://arxiv.org/html/2303.00320v4",
        "source_type": "method_paper",
        "takeaway": (
            "TimeMAE raises masked modeling from individual points to sub-series semantic units, reducing cost "
            "and improving label-scarce transfer."
        ),
        "implication": "Use 30-minute to multi-hour windows as semantic units instead of minute-level reconstruction only.",
    },
    {
        "key": "tnc",
        "title": "Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding",
        "url": "https://arxiv.org/abs/2106.00750",
        "source_type": "method_paper",
        "takeaway": (
            "TNC learns representations for non-stationary time series by contrasting local temporal neighborhoods "
            "against non-neighboring windows."
        ),
        "implication": "Subject-day neighborhoods are plausible, but validation must prevent date/subject leakage.",
    },
    {
        "key": "ts2vec",
        "title": "TS2Vec: Towards Universal Representation of Time Series",
        "url": "https://arxiv.org/abs/2106.10466",
        "source_type": "method_paper",
        "takeaway": (
            "TS2Vec learns hierarchical timestamp and subsequence representations through contrastive learning."
        ),
        "implication": "Use as a secondary objective only after missingness-aware cross-modal masking is fixed.",
    },
]

METHOD_CANDIDATES = [
    {
        "route": "airkorea_2024_historical_gate",
        "priority": 4,
        "research_support": ["etri_lifelog_2024", "dacon_data_page"],
        "fit": "completed_closed",
        "why": (
            "The downloaded 2024 AirKorea archive provides a true historical external source with full "
            "Daejeon hourly coverage, but its ID342 gate opens no local or Public-0.58 review row."
        ),
        "execution_shape": (
            "Keep as negative source evidence; do not create artifacts, Kaggle runs, or DACON submissions."
        ),
        "risk": "The small S4 movement is date-structured and failed frontier-correlation checks.",
    },
    {
        "route": "materially_new_external_source_hypothesis",
        "priority": 1,
        "research_support": ["etri_lifelog_2024", "dacon_data_page", "lsm2_aim", "crossl", "time_mae"],
        "fit": "highest_after_validation_audit_closed",
        "why": (
            "The registered OOF shelf, raw sleep/event routes, block-sequence proxy, and validation-weighted audit "
            "all fail to open even a credible Public-0.58 review row."
        ),
        "execution_shape": (
            "Write a fresh source-hypothesis packet that names a new modality interaction or external-derived "
            "feature family, defines a local gate, and blocks artifacts unless the gate beats ID342 and survives "
            "stable-shape/failed-frontier checks."
        ),
        "risk": "The bar is high: small local overlays and same-family heavy reruns have repeatedly compressed near Public 0.596.",
    },
    {
        "route": "external_etri_sleep_survey_prior",
        "priority": 4,
        "research_support": ["etri_lifelog_2024", "dacon_data_page"],
        "fit": "completed_closed",
        "why": (
            "Historical ETRI sleep/survey files provide direct PM fatigue/stress and sleep metrics, but the "
            "audit supports Q2 fatigue coupling only and opens no direct Q3/S3/S4 2026 feature route."
        ),
        "execution_shape": "Keep as domain-prior evidence; do not create artifacts, Kaggle runs, or DACON submissions.",
        "risk": "Directly concatenating historical users or survey/sleep labels would not match the 2026 feature contract.",
    },
    {
        "route": "solar_seasonal_holiday_prior",
        "priority": 3,
        "research_support": ["etri_lifelog_2024", "dacon_data_page"],
        "fit": "completed_closed",
        "why": (
            "External date-derived Seoul/Korea photoperiod, season, and holiday proximity features produced only "
            "unstable S4 movement and no eligible Q3/S3/S4 row."
        ),
        "execution_shape": "Keep as negative source evidence; do not escalate to artifacts or Kaggle.",
        "risk": "The S4 mean delta is tempting but loses subject/generalization stability.",
    },
    {
        "route": "kma_asos_weather_context",
        "priority": 4,
        "research_support": ["etri_lifelog_2024", "dacon_data_page"],
        "fit": "completed_closed",
        "why": (
            "Historical Daejeon ASOS hourly weather provides true external observation data, but the first "
            "ID342 gate finds only a small unstable S4 signal and no Public-0.58 review row."
        ),
        "execution_shape": "Keep as negative weather-source evidence; do not escalate to artifacts or Kaggle.",
        "risk": "Weather is strongly date-structured and can mimic failed frontier/date movement if overweighted.",
    },
    {
        "route": "approved_public_data_api_feasibility",
        "priority": 4,
        "research_support": ["etri_lifelog_2024", "dacon_data_page"],
        "fit": "completed_closed",
        "why": (
            "KMA living-weather and AirKorea information endpoints are reachable, but the smoke does not expose "
            "a date-bounded 2024 historical source beyond KMA ASOS."
        ),
        "execution_shape": "Keep KMA living-weather and approved AirKorea information endpoints report-only.",
        "risk": "Recent/realtime air-quality data cannot be backfilled into the 2024 train/test window without a separate source.",
    },
    {
        "route": "validation_weighted_pseudo_public_split",
        "priority": 4,
        "research_support": ["etri_lifelog_2024", "dacon_data_page"],
        "fit": "completed_closed",
        "why": (
            "The audit evaluated target-weighted full/subject-hole/LOSO/test-neighbor lenses and found no credible "
            "Public-0.55 or Public-0.58 stable-shape row."
        ),
        "execution_shape": "Keep as a guardrail for future candidates, not as a candidate-generation route.",
        "risk": "Failed-frontier rows still look locally attractive, so this lens must stay paired with shape penalties.",
    },
    {
        "route": "mis_lstm_multichannel_block_sequence_v2",
        "priority": 5,
        "research_support": ["mis_lstm", "pixleepflow", "etri_lifelog_2024", "dacon_data_page"],
        "fit": "high_but_local_and_validation_gates_failed",
        "why": (
            "The first local block-sequence proxy found only S4. A second attempt must change the representation, "
            "not simply rerun the same block aggregates."
        ),
        "execution_shape": (
            "Only revisit with a truly sequence-aware model or target-specific loss after validation weighting is repaired."
        ),
        "risk": "Do not launch Kaggle from the current local proxy, which failed Q3/S3 and projected only 0.5959.",
    },
    {
        "route": "aim_cross_modal_masked_subject_state_v2",
        "priority": 6,
        "research_support": ["etri_lifelog_2024", "dacon_data_page", "lsm2_aim", "crossl", "multi_modal_ts_survey"],
        "fit": "high_but_already_failed_once",
        "why": (
            "AIM/CroSSL still matches incomplete lifelog data, but the post-ID342 subject-state and missingness runs "
            "did not open a supervised gate."
        ),
        "execution_shape": (
            "Only revisit if the design changes materially: block-temporal tokens, target-specific downstream gates, "
            "and explicit Public-0.55 promotion thresholds before remote execution."
        ),
        "risk": "Same-family missingness/subject-state reruns are closed unless the local feasibility story changes.",
    },
    {
        "route": "time_mae_semantic_window_subject_state",
        "priority": 7,
        "research_support": ["time_mae", "etri_lifelog_2024", "pixleepflow"],
        "fit": "high",
        "why": "The dataset is label-scarce and long-horizon; semantic sub-series windows are more relevant than point loss.",
        "execution_shape": (
            "Build 30-minute, 2-hour, evening, and sleep-transition tokens; reconstruct masked semantic units with "
            "visible/masked decoupling."
        ),
        "risk": "Can collapse into another generic sequence summary unless cross-modal missingness is retained.",
    },
    {
        "route": "crossl_latent_modality_dropout",
        "priority": 8,
        "research_support": ["crossl", "multi_modal_ts_survey", "dacon_data_page"],
        "fit": "high",
        "why": "The raw streams have natural missing modalities and different sampling frequencies.",
        "execution_shape": (
            "Use small modality-specific encoders, latent masks, and a global aggregator; downstream uses frozen "
            "embeddings plus low-capacity fold-internal heads."
        ),
        "risk": "May be too small to close a 0.046 public gap unless paired with AIM-style inherited masking.",
    },
    {
        "route": "tnc_subject_temporal_neighborhood",
        "priority": 9,
        "research_support": ["tnc"],
        "fit": "medium",
        "why": "Q3/S3 are state-transition targets, so local temporal smoothness is plausible.",
        "execution_shape": "Contrast same-subject nearby windows against distant windows without labels or public-score donors.",
        "risk": "High leakage risk if neighborhoods are not fold- and date-safe; not a standalone first route.",
    },
    {
        "route": "ts2vec_hierarchical_context",
        "priority": 10,
        "research_support": ["ts2vec"],
        "fit": "medium",
        "why": "Hierarchical timestamp/subsequence embeddings are broadly useful for irregular lifelog windows.",
        "execution_shape": "Train contextual encoders on synchronized sequences and aggregate target-specific windows.",
        "risk": "Less directly matched to missing modality and cross-device behavior than AIM/CroSSL.",
    },
    {
        "route": "pixleepflow_image_layout_only",
        "priority": 11,
        "research_support": ["pixleepflow"],
        "fit": "low_as_standalone",
        "why": "Image conversion is domain-relevant, but local sequence-image and tiny CNN gates did not open Q3/S3.",
        "execution_shape": "Use only as a visualization/channel-layout prior inside a stronger masked model.",
        "risk": "Do not launch another image-only or random-conv route from current evidence.",
    },
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}, "error": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
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
    }


def decision(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("decision", {})
    return value if isinstance(value, dict) else {}


def derive_prerequisites(reports: dict[str, dict[str, Any]]) -> dict[str, bool]:
    synthesis = decision(reports["remaining_route_synthesis"])
    penalty = decision(reports["public_gap_penalty_proxy"])
    missingness = decision(reports["missingness_subject_state_intake"])
    sequence_event = decision(reports["sequence_hr_light_event"])
    mis_lstm = decision(reports["mis_lstm_block_sequence"])
    validation_weighted = decision(reports["validation_weighted_pseudo_public"])
    solar_holiday = decision(reports["solar_seasonal_holiday_prior"])
    kma_weather = decision(reports["kma_weather_context"])
    public_data_api = decision(reports["public_data_api_smoke"])
    airkorea_2024 = decision(reports["airkorea_2024_hourly_audit"])
    airkorea_context = decision(reports["airkorea_2024_context"])
    external_etri_prior = decision(reports["external_etri_sleep_survey_prior"])
    cross_source = decision(reports["cross_source_split_stability"])
    s3_phone_gps = decision(reports["s3_phone_gps_interaction"])
    final_day = decision(reports["final_day_independent_signal"])
    domain = decision(reports["domain_token_contract"])
    schema = decision(reports["raw_schema_contract"])
    thresholds = decision(reports["heavy_success_thresholds"])
    return {
        "input_reports_parse": all(report.get("parse_ok") is True for report in reports.values()),
        "goal_not_achieved": synthesis.get("current_public_score", 1.0) > 0.55,
        "id342_anchor_current": synthesis.get("current_public_anchor_id") == 342,
        "registered_routes_closed": synthesis.get("all_registered_routes_closed") is True
        and int(synthesis.get("closed_route_count", 0)) >= 23,
        "public_gap_penalty_closed": penalty.get("credible_public055_penalty_proxy_pass_count") == 0,
        "validation_weighted_route_already_failed": validation_weighted.get(
            "validation_weighted_pseudo_public_gate_pass"
        )
        is False
        and validation_weighted.get("credible_weighted_public055_pass_count") == 0
        and validation_weighted.get("credible_weighted_review058_pass_count") == 0,
        "solar_seasonal_holiday_route_already_failed": solar_holiday.get(
            "solar_seasonal_holiday_prior_gate_pass"
        )
        is False,
        "kma_weather_context_route_already_failed": kma_weather.get("weather_context_gate_pass") is False
        and kma_weather.get("review_gate_pass_count") == 0,
        "approved_public_data_api_smoke_checked": public_data_api.get("all_three_service_families_smoke_checked")
        is True,
        "approved_non_asos_public_data_api_historical_route_closed": public_data_api.get(
            "non_asos_approved_api_2024_historical_feature_source_available"
        )
        is False,
        "airkorea_2024_historical_archive_ready": airkorea_2024.get("airkorea_2024_zip_available") is True
        and airkorea_2024.get("competition_window_coverage_ok") is True,
        "airkorea_2024_historical_gate_already_failed": airkorea_context.get(
            "airkorea_2024_context_gate_pass"
        )
        is False
        and airkorea_context.get("review_gate_pass_count") == 0,
        "external_etri_sleep_survey_prior_already_failed": external_etri_prior.get("external_prior_gate_pass")
        is False
        and external_etri_prior.get("direct_2026_feature_route_available") is False,
        "cross_source_split_stability_route_already_failed": cross_source.get(
            "cross_source_split_stability_gate_pass"
        )
        is False
        and cross_source.get("review_signal_targets") == [],
        "s3_phone_gps_interaction_route_already_failed": s3_phone_gps.get(
            "s3_phone_gps_interaction_gate_pass"
        )
        is False
        and s3_phone_gps.get("local_gate_pass_count") == 0,
        "final_day_independent_signal_route_already_failed": final_day.get(
            "final_day_independent_signal_gate_pass"
        )
        is False
        and final_day.get("review_gate_pass_count") == 0,
        "missingness_route_already_failed": missingness.get("stable_gate_pass_count") == 0,
        "sequence_event_route_already_failed": sequence_event.get("final_sequence_hr_light_event_gate_pass") is False,
        "mis_lstm_block_route_already_failed": mis_lstm.get("final_mis_lstm_block_sequence_gate_pass") is False,
        "domain_contract_ready": domain.get("domain_token_contract_ready") is True,
        "raw_schema_contract_ready": schema.get("raw_schema_contract_ready") is True,
        "raw_payload_not_read": schema.get("raw_payload_rows_read") == 0,
        "success_thresholds_ready": thresholds.get("heavy_success_threshold_contract_ready") is True,
        "execution_closed_now": synthesis.get("run_kaggle_now") is False
        and synthesis.get("push_kaggle_now") is False
        and synthesis.get("dacon_submit_now") is False,
    }


def build_research_implications() -> list[dict[str, str]]:
    return [
        {
            "finding": "The large public gap is representation-sized, not calibration-sized.",
            "evidence": "ID342 is still +0.0460832519 from Public 0.55 after 32 registered routes closed.",
            "action": "Do not spend DACON on alpha, temperature, target-swap, or image-only variants.",
        },
        {
            "finding": "Missingness remains important, but the first missingness-heavy execution did not transfer.",
            "evidence": "The post-ID342 missingness subject-state no-submit route scanned all raw rows and passed 0 stable rows.",
            "action": "Do not rerun AIM/CroSSL-style missingness unless the architecture and local gate materially change.",
        },
        {
            "finding": "Temporal block structure has been locally tested and is not sufficient by itself.",
            "evidence": "The local MIS-LSTM-style block proxy found only S4 and failed Q3/S3.",
            "action": "Do not launch a Kaggle runner from this route; repair validation or change the representation materially.",
        },
        {
            "finding": "Validation weighting is useful as a guardrail but does not create a candidate route.",
            "evidence": "The validation-weighted pseudo-public audit found 0 credible Public-0.55 rows and 0 credible Public-0.58 review rows.",
            "action": "Use it to reject future shelf rows; do not rerun it as another candidate scan.",
        },
        {
            "finding": "The first external date-derived source is not stable enough.",
            "evidence": "Solar/seasonal/Korean-holiday priors move S4 locally in 3/5 lenses but pass 0 stable rows.",
            "action": "Do not promote the external-calendar prior; require a different source mechanism.",
        },
        {
            "finding": "Historical KMA weather is available but not strong enough.",
            "evidence": "The ASOS weather gate finds a best S4 pressure/precipitation row near Public 0.596006 with 0 review passes.",
            "action": "Keep KMA weather as negative external-source evidence unless a new source changes the mechanism.",
        },
        {
            "finding": "The approved non-ASOS public-data APIs do not cover the 2024 feature window.",
            "evidence": (
                "The public-data smoke verifies KMA living-weather current forecasts and AirKorea current/recent "
                "endpoints, but KMA living-weather rejects 2024 history and AirKorea station 3MONTH returns recent 2026 data."
            ),
            "action": "Do not build features from these approved endpoints; require a separate historical/statistics source.",
        },
        {
            "finding": "The separate downloaded AirKorea 2024 historical source is valid but too weak.",
            "evidence": (
                "The archive audit finds 13 Daejeon stations and all 4,200 expected timestamps, but the "
                "ID342 AirKorea gate has 0 local/review/DACON passes and best proxy 0.596056065."
            ),
            "action": "Do not promote air-quality features; require a materially different Q3/S3/S4 source mechanism.",
        },
        {
            "finding": "Historical ETRI sleep/survey priors do not rescue Q3.",
            "evidence": (
                "The 2020 PM-to-next-sleep audit pairs 523 rows. Lower fatigue aligns with shorter sleep latency "
                "and lower WASO, but lower stress does not align with better sleep outcomes and no direct 2026 "
                "feature route is available."
            ),
            "action": "Use this as a warning against simple Q3-from-sleep coupling; do not promote or launch from it.",
        },
        {
            "finding": "Cross-source weak-signal alignment is too small to promote.",
            "evidence": "S3 has a phone/GPS stable micro-alignment, but its best proxy remains near 0.596; S4 conflicts on LOSO.",
            "action": "Use the S3 alignment as design hint only; do not create artifacts or launch Kaggle from it.",
        },
        {
            "finding": "Focused S3 phone/GPS interactions improve local stability but not public distance.",
            "evidence": "The best S3 phone/GPS concat row wins 5/5 splits, but still projects only to 0.595849549.",
            "action": "Do not submit the micro-signal; require a materially larger S3/S4 source before promotion.",
        },
        {
            "finding": "The final-day independent sweep found a local S4 signal but no reviewable candidate.",
            "evidence": "The best S4 sleep-solar row wins 4/5 with mean delta -0.002013473, but projects only to 0.595795613.",
            "action": "Keep submission closed unless a future row clears the Public-0.58 review proxy.",
        },
        {
            "finding": "Image-style layouts are useful as priors but not sufficient.",
            "evidence": "PixleepFlow supports synchronized sensor images, but local ID342 image/CNN smokes did not pass.",
            "action": "Use image layout only with block-sequence aggregation and discrete-event handling.",
        },
    ]


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    reports = {key: read_json(path) for key, path in REPORT_PATHS.items()}
    prerequisites = derive_prerequisites(reports)
    ready = all(prerequisites.values())
    closeout = decision(reports["morning_closeout"])
    synthesis = decision(reports["remaining_route_synthesis"])
    thresholds = decision(reports["heavy_success_thresholds"])
    method_candidates = sorted(METHOD_CANDIDATES, key=lambda row: (row["priority"], row["route"]))
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_external_research_priority",
        "purpose": "Refresh external/source research priorities after all registered post-ID342 routes closed.",
        "input_reports": reports,
        "prerequisites": prerequisites,
        "external_sources": EXTERNAL_SOURCES,
        "research_implications": build_research_implications(),
        "method_candidates": method_candidates,
        "score_state": {
            "best_public_anchor_id": synthesis.get("current_public_anchor_id", closeout.get("best_public_anchor_id")),
            "best_public_score": synthesis.get("current_public_score", closeout.get("best_public_score")),
            "target_public_score": synthesis.get("target_public_score", 0.55),
            "gap_to_public_055": synthesis.get("gap_to_public_055", closeout.get("gap_to_public_055")),
            "closed_route_count": synthesis.get("closed_route_count"),
            "route_count": synthesis.get("route_count"),
        },
        "recommended_next_route": {
            "name": "materially_new_external_source_hypothesis",
            "priority_components": [
                "new source mechanism rather than another weather/air-quality date prior",
                "avoid simple Q3-from-sleep coupling because external ETRI prior did not support it",
                "target-specific Q3/S3/S4 local gate before artifacts",
                "validation-weighted pseudo-public and stable-shape review",
                "failed-frontier exclusion and leave-one-out stress checks",
            ],
            "review_threshold_public_proxy": thresholds.get("artifact_review_threshold_public_proxy"),
            "dacon_candidate_threshold_public_proxy": thresholds.get("dacon_candidate_threshold_public_proxy"),
        },
        "decision": {
            "external_research_priority_ready": ready,
            "top_priority_route": "materially_new_external_source_hypothesis",
            "research_priority_updated": ready,
            "same_family_heavy_rerun_allowed": False,
            "mis_lstm_kaggle_allowed_now": False,
            "heavy_compute_allowed_now": False,
            "create_runner_files_now": False,
            "create_submission_artifact_now": False,
            "modify_submission_log_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "dacon_submit_now": False,
            "next_action": (
                "Keep execution closed. The validation-weighted pseudo-public audit and the first external "
                "solar/holiday and KMA ASOS weather gates are now closed, the approved KMA living-weather and "
                "AirKorea information APIs have been smoke-tested and do not provide a 2024 historical feature "
                "source, and the separate downloaded AirKorea 2024 context gate also opens no reviewable row. "
                "The historical ETRI sleep/survey prior supports Q2 fatigue coupling but opens no direct Q3/S3/S4 "
                "feature route. The cross-source split audit finds no reviewable target, the focused S3 phone/GPS "
                "interaction remains far from review thresholds, and the final-day independent sweep opens no "
                "Public-0.58 review row; next, define a materially different Q3/S3/S4 source hypothesis before "
                "any Kaggle, artifact, or DACON action."
            ),
        },
        "forbidden_actions_now": FORBIDDEN_ACTIONS_NOW,
        "report_only": True,
        "no_raw_payload_read": True,
        "no_embedding_fit": True,
        "no_feature_matrix_created": True,
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_kaggle_output_pull_made": True,
        "no_dacon_submission_made": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision_payload = report["decision"]
    lines = [
        "# Post-ID342 external research priority refresh",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This report updates research priority only. It performs no raw payload read, embedding fit, feature creation, model training, Kaggle action, submission-log edit, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Research priority ready: `{decision_payload['external_research_priority_ready']}`",
        f"- Top priority route: `{decision_payload['top_priority_route']}`",
        f"- Same-family heavy rerun allowed: `{decision_payload['same_family_heavy_rerun_allowed']}`",
        f"- MIS-LSTM Kaggle allowed now: `{decision_payload['mis_lstm_kaggle_allowed_now']}`",
        f"- Heavy compute allowed now: `{decision_payload['heavy_compute_allowed_now']}`",
        f"- Create runner files now: `{decision_payload['create_runner_files_now']}`",
        f"- Run Kaggle now: `{decision_payload['run_kaggle_now']}`",
        f"- DACON submit now: `{decision_payload['dacon_submit_now']}`",
        f"- Next action: {decision_payload['next_action']}",
        "",
        "## Score State",
        "",
        "```json",
        json.dumps(report["score_state"], indent=2, sort_keys=True),
        "```",
        "",
        "## Recommended Next Route",
        "",
        "```json",
        json.dumps(report["recommended_next_route"], indent=2, sort_keys=True),
        "```",
        "",
        "## Research Implications",
        "",
        "| finding | evidence | action |",
        "|---|---|---|",
    ]
    for row in report["research_implications"]:
        lines.append(f"| {row['finding']} | {row['evidence']} | {row['action']} |")
    lines.extend(
        [
            "",
            "## Method Candidates",
            "",
            "| priority | route | fit | why | risk |",
            "|---:|---|---|---|---|",
        ]
    )
    for row in report["method_candidates"]:
        lines.append(f"| {row['priority']} | `{row['route']}` | {row['fit']} | {row['why']} | {row['risk']} |")
    lines.extend(
        [
            "",
            "## External Sources",
            "",
            "| source | type | URL | implication |",
            "|---|---|---|---|",
        ]
    )
    for row in report["external_sources"]:
        lines.append(f"| {row['title']} | {row['source_type']} | {row['url']} | {row['implication']} |")
    lines.extend(
        [
            "",
            "## Prerequisites",
            "",
            "```json",
            json.dumps(report["prerequisites"], indent=2, sort_keys=True),
            "```",
            "",
            "## Forbidden Actions Now",
            "",
            "```json",
            json.dumps(report["forbidden_actions_now"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, args.output)
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(f"Research priority ready: {report['decision']['external_research_priority_ready']}")
    print(f"Run Kaggle now: {report['decision']['run_kaggle_now']}")


if __name__ == "__main__":
    main()
