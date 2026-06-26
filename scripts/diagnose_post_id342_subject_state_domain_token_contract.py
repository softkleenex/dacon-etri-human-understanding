#!/usr/bin/env python3
"""Post-ID342 subject-state domain/token contract.

This report-only diagnostic turns the remaining heavy self-supervised route
into a concrete domain and tokenization contract. It intentionally does not
read raw parquet payloads, create feature matrices, fit embeddings, create
predictions, push/run Kaggle, edit the submission log, or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_subject_state_domain_token_contract_20260624.json"
KST = timezone(timedelta(hours=9))

HEAVY_APPROVAL_PHRASE = "Self-supervised subject-state heavy compute no-submit 실행 승인"

INPUT_REPORTS = {
    "heavy_contract": REPORTS_DIR / "post_id342_self_supervised_heavy_compute_contract_20260624.json",
    "runner_spec": REPORTS_DIR / "post_id342_self_supervised_heavy_runner_spec_20260624.json",
    "research_packet": REPORTS_DIR / "post_id342_self_supervised_route_research_packet_20260624.json",
    "remaining_route_synthesis": REPORTS_DIR / "post_id342_remaining_route_synthesis_20260624.json",
    "goal_state_audit": REPORTS_DIR / "post_id342_goal_state_audit_20260624.json",
}

PRIMARY_SOURCES = [
    {
        "name": "Human Understanding AI Paper Challenge 2024 - Dataset Design",
        "url": "https://arxiv.org/abs/2403.16509",
        "contract_use": (
            "Use participant/date-aware validation and account for noisy, missing, real-world phone and wearable "
            "streams."
        ),
    },
    {
        "name": "Understanding Human Daily Experience Through Continuous Sensing: ETRI Lifelog Dataset 2024",
        "url": "https://arxiv.org/html/2508.03698v1",
        "contract_use": (
            "Treat Q3 as within-subject pre-sleep stress and S3 as sleep-onset-latency adherence; keep all "
            "timestamps in KST and align by subject/date."
        ),
    },
    {
        "name": "MoCA: Multi-modal Cross-masked Autoencoder for Digital Health Measurements",
        "url": "https://arxiv.org/abs/2506.02260",
        "contract_use": "Use cross-modality masking so missing phone/watch channels are part of the objective.",
    },
    {
        "name": "CroSSL: Cross-modal Self-Supervised Learning for Time-series Signals",
        "url": "https://arxiv.org/abs/2307.16847",
        "contract_use": (
            "Use modality-specific encoders plus a cross-modal aggregator instead of concatenating all streams first."
        ),
    },
    {
        "name": "Personalization of Stress Mobile Sensing using Self-Supervised Learning",
        "url": "https://arxiv.org/abs/2308.02731",
        "contract_use": "Evaluate Q3 with explicit subject personalization, not only pooled validation loss.",
    },
    {
        "name": "Personalized Prediction of Recurrent Stress Events Using Self-supervised Learning",
        "url": "https://arxiv.org/abs/2307.03337",
        "contract_use": "Keep a subject-baseline dynamics objective for stress-state transfer as an ablation.",
    },
]

MODALITY_CONTRACTS = [
    {
        "modality": "mUsageStats",
        "domain": "phone_app_load",
        "raw_file": "data/raw/ch2025_data_items/ch2025_mUsageStats.parquet",
        "sample_interval": "10-min",
        "primary_targets": ["Q2", "Q3"],
        "token_role": "app-session burden, late-evening stimulation, routine break, social/media load",
        "token_examples": [
            "category-duration bins by anchored relative hour",
            "late-session entropy and burst count",
            "subject-relative novelty of evening app mix",
        ],
        "privacy_guard": "Hash or bucket app names; do not expose raw app-name identifiers in model reports.",
    },
    {
        "modality": "mScreenStatus",
        "domain": "sleep_boundary_phone_use",
        "raw_file": "data/raw/ch2025_data_items/ch2025_mScreenStatus.parquet",
        "sample_interval": "1-min",
        "primary_targets": ["S3", "Q3"],
        "token_role": "screen-off settling, last use before sleep, night interruption proxy",
        "token_examples": [
            "last screen-on in 18:00-03:00 grid",
            "screen-free run length before inferred sleep segment",
            "screen reactivation count after long still/dark run",
        ],
        "privacy_guard": "Use binary state transitions only.",
    },
    {
        "modality": "mACStatus",
        "domain": "charge_bedside_routine",
        "raw_file": "data/raw/ch2025_data_items/ch2025_mACStatus.parquet",
        "sample_interval": "10-min",
        "primary_targets": ["S3", "S1"],
        "token_role": "bedside charging habit and nightly phone parking proxy",
        "token_examples": [
            "charge-start relative hour",
            "charge duration overlapping still/dark window",
            "missing-charge routine deviation",
        ],
        "privacy_guard": "Use charging binary state only.",
    },
    {
        "modality": "mActivity",
        "domain": "phone_activity_state",
        "raw_file": "data/raw/ch2025_data_items/ch2025_mActivity.parquet",
        "sample_interval": "1-min",
        "primary_targets": ["S3", "Q3"],
        "token_role": "stillness onset, activity fragmentation, commute/activity stress proxy",
        "token_examples": [
            "longest still run in evening-to-sleep grid",
            "active-to-still transition count",
            "late vehicle/on-foot state before sleep",
        ],
        "privacy_guard": "Use coarse activity labels only.",
    },
    {
        "modality": "wHr",
        "domain": "wearable_recovery_arousal",
        "raw_file": "data/raw/ch2025_data_items/ch2025_wHr.parquet",
        "sample_interval": "1-min",
        "primary_targets": ["Q3", "S3", "S4"],
        "token_role": "physiological arousal, recovery slope, nocturnal fragmentation proxy",
        "token_examples": [
            "day-to-evening HR delta",
            "night HR dip timing",
            "short-window HR variability token",
        ],
        "privacy_guard": "Normalize inside fold by subject; do not fit subject baselines on validation labels.",
    },
    {
        "modality": "wPedo",
        "domain": "wearable_exertion_inactivity",
        "raw_file": "data/raw/ch2025_data_items/ch2025_wPedo.parquet",
        "sample_interval": "1-min",
        "primary_targets": ["Q2", "Q3", "S3"],
        "token_role": "daytime load, evening inactivity, sleep-preparation settling",
        "token_examples": [
            "daytime step load percentiles",
            "evening low-step run length",
            "post-high-load recovery interaction token",
        ],
        "privacy_guard": "Use aggregate movement states only.",
    },
    {
        "modality": "mLight",
        "domain": "phone_light_context",
        "raw_file": "data/raw/ch2025_data_items/ch2025_mLight.parquet",
        "sample_interval": "10-min",
        "primary_targets": ["S3", "Q3"],
        "token_role": "darkness onset, light exposure, pre-sleep environment proxy",
        "token_examples": [
            "log-light evening bins",
            "first sustained dark interval",
            "darkness alignment with screen-off/stillness",
        ],
        "privacy_guard": "Use scalar light values only.",
    },
    {
        "modality": "wLight",
        "domain": "watch_light_context",
        "raw_file": "data/raw/ch2025_data_items/ch2025_wLight.parquet",
        "sample_interval": "1-min",
        "primary_targets": ["S3", "Q3"],
        "token_role": "body-worn light exposure and dark-state confirmation",
        "token_examples": [
            "watch-light drop around sleep boundary",
            "phone/watch light disagreement token",
            "sustained low-light duration",
        ],
        "privacy_guard": "Use scalar light values only.",
    },
    {
        "modality": "mGps",
        "domain": "mobility_place_routine",
        "raw_file": "data/raw/ch2025_data_items/ch2025_mGps.parquet",
        "sample_interval": "1-min",
        "primary_targets": ["Q2", "Q3"],
        "token_role": "mobility load, place regularity, commute/stress context",
        "token_examples": [
            "coarse distance-from-home proxy",
            "place transition entropy",
            "late mobility after usual settling time",
        ],
        "privacy_guard": "Never expose raw coordinates; derive coarse, subject-relative tokens only.",
    },
    {
        "modality": "mWifi",
        "domain": "place_stability",
        "raw_file": "data/raw/ch2025_data_items/ch2025_mWifi.parquet",
        "sample_interval": "10-min",
        "primary_targets": ["Q3", "S3"],
        "token_role": "home/work/place stability and routine deviation",
        "token_examples": [
            "hashed access-point novelty",
            "evening home-stability proxy",
            "subject-relative place entropy",
        ],
        "privacy_guard": "Hash BSSID and report only counts/entropy.",
    },
    {
        "modality": "mBle",
        "domain": "near_device_social_context",
        "raw_file": "data/raw/ch2025_data_items/ch2025_mBle.parquet",
        "sample_interval": "10-min",
        "primary_targets": ["Q3"],
        "token_role": "near-device density, social/place context, routine deviation",
        "token_examples": [
            "hashed device novelty",
            "near-device count by time bucket",
            "evening deviation from subject baseline",
        ],
        "privacy_guard": "Hash device IDs and use only counts/classes/RSSI summaries.",
    },
    {
        "modality": "mAmbience",
        "domain": "ambient_sound_context",
        "raw_file": "data/raw/ch2025_data_items/ch2025_mAmbience.parquet",
        "sample_interval": "2-min",
        "primary_targets": ["Q3", "S3"],
        "token_role": "noise/speech/music/silence context and sleep environment proxy",
        "token_examples": [
            "ambient label probability bins",
            "speech/music/silence evening mixture",
            "night-noise interruption token",
        ],
        "privacy_guard": "Use label categories and probabilities only; no audio payloads.",
    },
]

TARGET_TOKEN_VIEWS = [
    {
        "target": "Q3",
        "meaning": "low stress before sleep relative to the participant baseline",
        "positive_class": "subjectively lower stress than the participant average",
        "negative_class": "subjectively higher stress than the participant average",
        "must_capture": [
            "subject-relative arousal/load deviation",
            "evening recovery and routine break",
            "phone/social/place novelty",
            "wearable HR and activity recovery",
        ],
        "danger": "A global stress prior can look locally good but miss participant-specific thresholds.",
    },
    {
        "target": "S3",
        "meaning": "sleep onset latency adheres to sleep-health guidance",
        "positive_class": "sleep onset latency in recommended range",
        "negative_class": "delayed or otherwise inappropriate sleep onset latency",
        "must_capture": [
            "screen-off to still/dark transition",
            "charging and phone parking routine",
            "light/activity/HR settling order",
            "night interruption before stable sleep onset",
        ],
        "danger": "Daily aggregates flatten the transition ordering that likely determines SOL.",
    },
    {
        "target": "Q2",
        "meaning": "low physical fatigue before sleep relative to the participant baseline",
        "positive_class": "subjectively less fatigued than participant average",
        "negative_class": "subjectively more fatigued than participant average",
        "must_capture": [
            "daytime physical load",
            "late recovery from high-load day",
            "sleep-transition sanity interactions",
        ],
        "danger": "Use Q2 only as sanity support; it must not distract from Q3/S3.",
    },
]

TOKENIZATION_WINDOWS = [
    {
        "name": "day_grid",
        "relative_hours": "00:00-24:00 by clock day",
        "purpose": "daily load and routine stability for Q2/Q3",
        "binning": "24 hourly bins with missingness mask",
    },
    {
        "name": "anchored_evening_grid",
        "relative_hours": "18:00-03:00 anchored to lifelog/sleep date",
        "purpose": "pre-sleep load, stress recovery, and transition onset",
        "binning": "30 or 60 minute bins with modality masks",
    },
    {
        "name": "sleep_transition_grid",
        "relative_hours": "last active/light/screen state through first sustained still/dark state",
        "purpose": "S3 sleep-onset-latency state ordering",
        "binning": "event tokens plus coarse time deltas",
    },
    {
        "name": "subject_baseline_delta",
        "relative_hours": "fold-internal subject history only",
        "purpose": "Q3/Q2 participant-relative labels",
        "binning": "deviation from fold-fit subject medians or rolling unlabeled norms",
    },
]

OBJECTIVE_ABLATIONS = [
    {
        "stage": 1,
        "name": "masked_single_modality_reconstruction",
        "why": "Sanity-check token quality and missingness masks before cross-modal learning.",
        "allowed_after_approval": True,
    },
    {
        "stage": 2,
        "name": "cross_modal_masked_reconstruction",
        "why": "Primary MoCA-style objective for phone/watch correlations and missing channels.",
        "allowed_after_approval": True,
    },
    {
        "stage": 3,
        "name": "latent_masking_with_modality_specific_encoders",
        "why": "CroSSL-style ablation for robust missing-modality aggregation.",
        "allowed_after_approval": True,
    },
    {
        "stage": 4,
        "name": "subject_baseline_dynamics_prediction",
        "why": "Personalized stress SSL ablation for Q3 participant thresholds.",
        "allowed_after_approval": True,
    },
    {
        "stage": 5,
        "name": "fold_internal_downstream_smoke",
        "why": "Only after label-free embeddings are frozen; compare Q3/S3 against ID342 stable lenses.",
        "allowed_after_approval": True,
    },
]

FORBIDDEN_ACTIONS_NOW = {
    "read_raw_parquet_payloads": False,
    "write_token_cache": False,
    "fit_embedding": False,
    "write_feature_matrix": False,
    "train_downstream_model": False,
    "create_predictions": False,
    "create_submission_artifact": False,
    "modify_submission_log": False,
    "push_kaggle": False,
    "run_kaggle": False,
    "submit_dacon": False,
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "parse_ok": False, "decision": {}, "path": rel(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"exists": True, "parse_ok": False, "decision": {}, "path": rel(path), "error": str(exc)}
    return {"exists": True, "parse_ok": True, "decision": payload.get("decision", {}), "path": rel(path)}


def prerequisite_status(reports: dict[str, dict[str, Any]]) -> dict[str, bool]:
    heavy = reports["heavy_contract"].get("decision", {})
    runner = reports["runner_spec"].get("decision", {})
    research = reports["research_packet"].get("decision", {})
    synthesis = reports["remaining_route_synthesis"].get("decision", {})
    goal = reports["goal_state_audit"].get("decision", {})
    return {
        "all_input_reports_parse": all(report.get("parse_ok") is True for report in reports.values()),
        "heavy_contract_ready": heavy.get("post_id342_heavy_compute_contract_ready") is True,
        "heavy_execution_blocked_now": heavy.get("heavy_compute_allowed_now") is False,
        "runner_spec_ready": runner.get("runner_spec_ready") is True,
        "runner_execution_blocked_now": runner.get("run_kaggle_now") is False,
        "research_packet_ready": research.get("research_packet_ready") is True,
        "only_heavy_route_remaining": synthesis.get("closed_route_count") == 11 and synthesis.get("route_count") == 12,
        "goal_not_achieved": goal.get("goal_achieved") is False and goal.get("public_055_reached") is False,
    }


def modality_summary() -> dict[str, Any]:
    q3 = sorted({row["modality"] for row in MODALITY_CONTRACTS if "Q3" in row["primary_targets"]})
    s3 = sorted({row["modality"] for row in MODALITY_CONTRACTS if "S3" in row["primary_targets"]})
    q2 = sorted({row["modality"] for row in MODALITY_CONTRACTS if "Q2" in row["primary_targets"]})
    return {
        "modality_count": len(MODALITY_CONTRACTS),
        "q3_modalities": q3,
        "s3_modalities": s3,
        "q2_modalities": q2,
        "cross_target_overlap_modalities": sorted(set(q3).intersection(s3)),
    }


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    reports = {key: read_json(path) for key, path in INPUT_REPORTS.items()}
    prereq = prerequisite_status(reports)
    ready = all(prereq.values())
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_subject_state_domain_token_contract",
        "purpose": "Freeze domain semantics and tokenization for the remaining post-ID342 heavy route.",
        "input_reports": reports,
        "primary_sources": PRIMARY_SOURCES,
        "modality_contracts": MODALITY_CONTRACTS,
        "target_token_views": TARGET_TOKEN_VIEWS,
        "tokenization_windows": TOKENIZATION_WINDOWS,
        "objective_ablations": OBJECTIVE_ABLATIONS,
        "modality_summary": modality_summary(),
        "prerequisites": prereq,
        "decision": {
            "domain_token_contract_ready": ready,
            "target_focus": ["Q2", "Q3", "S3"],
            "primary_score_targets": ["Q3", "S3"],
            "secondary_sanity_target": "Q2",
            "requires_exact_heavy_compute_approval": True,
            "heavy_approval_phrase": HEAVY_APPROVAL_PHRASE,
            "create_runner_files_now": False,
            "read_raw_parquet_payloads_now": False,
            "write_token_cache_now": False,
            "fit_embedding_now": False,
            "write_feature_matrix_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_submission_artifact_now": False,
            "modify_submission_log_now": False,
            "dacon_submit_now": False,
            "next_action": (
                "Use this contract to implement the no-submit runner only after exact heavy approval."
                if ready
                else "Refresh prerequisite post-ID342 reports before implementing the runner."
            ),
        },
        "forbidden_actions_now": FORBIDDEN_ACTIONS_NOW,
        "report_only": True,
        "no_raw_parquet_payload_read": True,
        "no_feature_matrix_created": True,
        "no_embedding_fit": True,
        "no_prediction_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 subject-state domain/token contract",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This report freezes domain semantics for the only remaining post-ID342 heavy route. It does not read raw parquet payloads, fit embeddings, create features, run Kaggle, or submit to DACON.",
        "",
        "## Decision",
        "",
        f"- Domain/token contract ready: `{decision['domain_token_contract_ready']}`",
        f"- Primary score targets: `{decision['primary_score_targets']}`",
        f"- Secondary sanity target: `{decision['secondary_sanity_target']}`",
        f"- Create runner files now: `{decision['create_runner_files_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Required phrase: `{decision['heavy_approval_phrase']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Modality Summary",
        "",
        "```json",
        json.dumps(report["modality_summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Target Token Views",
        "",
        "| target | meaning | must capture | danger |",
        "|---|---|---|---|",
    ]
    for row in report["target_token_views"]:
        lines.append(f"| `{row['target']}` | {row['meaning']} | {'; '.join(row['must_capture'])} | {row['danger']} |")

    lines.extend(
        [
            "",
            "## Modality Contracts",
            "",
            "| modality | domain | primary targets | token role | privacy guard |",
            "|---|---|---|---|---|",
        ]
    )
    for row in report["modality_contracts"]:
        lines.append(
            f"| `{row['modality']}` | `{row['domain']}` | `{row['primary_targets']}` | "
            f"{row['token_role']} | {row['privacy_guard']} |"
        )

    lines.extend(
        [
            "",
            "## Tokenization Windows",
            "",
            "| name | relative hours | purpose | binning |",
            "|---|---|---|---|",
        ]
    )
    for row in report["tokenization_windows"]:
        lines.append(f"| `{row['name']}` | {row['relative_hours']} | {row['purpose']} | {row['binning']} |")

    lines.extend(
        [
            "",
            "## Objective Ablations After Approval",
            "",
            "| stage | name | why |",
            "|---:|---|---|",
        ]
    )
    for row in report["objective_ablations"]:
        lines.append(f"| {row['stage']} | `{row['name']}` | {row['why']} |")

    lines.extend(
        [
            "",
            "## Primary Sources",
            "",
            "| source | url | contract use |",
            "|---|---|---|",
        ]
    )
    for row in report["primary_sources"]:
        lines.append(f"| {row['name']} | {row['url']} | {row['contract_use']} |")

    lines.extend(
        [
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
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.output)
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(f"Domain/token contract ready: {report['decision']['domain_token_contract_ready']}")
    print(f"Run Kaggle now: {report['decision']['run_kaggle_now']}")


if __name__ == "__main__":
    main()
