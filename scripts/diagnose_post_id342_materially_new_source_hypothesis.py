#!/usr/bin/env python3
"""Define materially new post-ID342 source hypotheses.

This report-only diagnostic combines target definitions, train/test row
structure, raw parquet schemas, processed feature-matrix schemas, and closed
post-ID342 route reports. It creates no feature matrix, predictions, submission
CSV, Kaggle action, submission-log row, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw"
RAW_ITEM_DIR = RAW_DIR / "ch2025_data_items"
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_materially_new_source_hypothesis_20260625.json"
KST = timezone(timedelta(hours=9))

TARGETS = ("Q1", "Q2", "Q3", "S1", "S2", "S3", "S4")
FOCUS_TARGETS = ("Q3", "S3", "S4")
CURRENT_PUBLIC_ANCHOR_ID = 342
CURRENT_PUBLIC_SCORE = 0.5960832519
PUBLIC_055 = 0.55
REVIEW_PUBLIC_THRESHOLD = 0.58

METRICS_TRAIN = RAW_DIR / "ch2026_metrics_train.csv"
SAMPLE_SUBMISSION = RAW_DIR / "ch2026_submission_sample.csv"

CLOSED_REPORTS = {
    "remaining_route_synthesis": REPORTS_DIR / "post_id342_remaining_route_synthesis_20260625.json",
    "external_research_priority": REPORTS_DIR / "post_id342_external_research_priority_20260625.json",
    "subject_prior_calibration": REPORTS_DIR / "post_id342_subject_prior_calibration_gate_20260624.json",
    "processed_sleep_temporal_deviation": REPORTS_DIR
    / "post_id342_processed_sleep_temporal_deviation_gate_20260624.json",
    "final_day_independent_signal": REPORTS_DIR / "post_id342_final_day_independent_signal_sweep_20260625.json",
    "s3_phone_gps_interaction": REPORTS_DIR / "post_id342_s3_phone_gps_interaction_gate_20260625.json",
    "sequence_hr_light_event": REPORTS_DIR / "post_id342_sequence_hr_light_event_gate_20260625.json",
    "airkorea_2024_context": REPORTS_DIR / "post_id342_airkorea_2024_context_gate_20260625.json",
    "kma_weather_context": REPORTS_DIR / "post_id342_kma_weather_context_gate_20260625.json",
}

FEATURE_NOVELTY_TOKENS = {
    "raw_app_name_lexicon": ("app_name", "lexicon", "stress_app"),
    "home_anchor": ("home_anchor", "home_arrival", "home_return", "night_home"),
    "subject_quota": ("quota", "label_quota", "rank_quota", "cumulative_target"),
}

RAW_MODALITY_ROLES = {
    "ch2025_mUsageStats.parquet": "app names, total app time, session mix",
    "ch2025_mScreenStatus.parquet": "screen on/off and phone-use timing",
    "ch2025_mAmbience.parquet": "audio scene tags such as speech, music, vehicle, place",
    "ch2025_mWifi.parquet": "recurring proximity tokens usable for place anchors",
    "ch2025_mBle.parquet": "nearby device recurrence and proximity density",
    "ch2025_mGps.parquet": "mobility, stationarity, approximate path geometry",
    "ch2025_wHr.parquet": "wearable heart-rate state",
    "ch2025_wPedo.parquet": "steps, speed, activity intensity",
    "ch2025_mACStatus.parquet": "charging and phone-at-rest proxy",
    "ch2025_mLight.parquet": "phone light exposure",
    "ch2025_wLight.parquet": "wearable light exposure",
    "ch2025_mActivity.parquet": "phone activity classifier",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "path": rel(path),
            "exists": True,
            "parse_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "decision": {},
        }
    return {
        "path": rel(path),
        "exists": True,
        "parse_ok": True,
        "decision": payload.get("decision", {}),
    }


def parquet_file_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False}
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    return {
        "path": rel(path),
        "exists": True,
        "rows": parquet.metadata.num_rows,
        "column_count": len(schema.names),
        "columns": schema.names,
    }


def raw_schema_inventory() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(RAW_ITEM_DIR.glob("*.parquet")):
        item = parquet_file_summary(path)
        item["role"] = RAW_MODALITY_ROLES.get(path.name, "raw modality")
        rows.append(item)
    return rows


def processed_schema_inventory() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(PROCESSED_DIR.glob("post_id341_*features_v1.parquet")):
        item = parquet_file_summary(path)
        cols = [str(col).lower() for col in item.get("columns", [])]
        novelty_hits = {
            family: sorted({token for token in tokens if any(token in col for col in cols)})
            for family, tokens in FEATURE_NOVELTY_TOKENS.items()
        }
        item["novelty_token_hits"] = novelty_hits
        item["feature_count_excluding_keys"] = max(0, int(item.get("column_count", 0)) - 4)
        rows.append(item)
    return rows


def target_distribution_summary() -> dict[str, Any]:
    train = pd.read_csv(METRICS_TRAIN)
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    target_means = {target: float(train[target].mean()) for target in TARGETS}
    target_counts = {
        target: {str(key): int(value) for key, value in train[target].value_counts(dropna=False).items()}
        for target in TARGETS
    }
    subject_target_means = (
        train.groupby("subject_id")[list(TARGETS)]
        .mean()
        .round(6)
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(sample)),
        "train_subjects": int(train["subject_id"].nunique()),
        "test_subjects": int(sample["subject_id"].nunique()),
        "train_rows_per_subject": {
            key: float(value)
            for key, value in train.groupby("subject_id").size().describe().round(6).to_dict().items()
        },
        "test_rows_per_subject": {
            key: float(value)
            for key, value in sample.groupby("subject_id").size().describe().round(6).to_dict().items()
        },
        "train_date_min": str(train["lifelog_date"].min()),
        "train_date_max": str(train["lifelog_date"].max()),
        "test_date_min": str(sample["lifelog_date"].min()),
        "test_date_max": str(sample["lifelog_date"].max()),
        "target_means": target_means,
        "target_counts": target_counts,
        "subject_target_means": subject_target_means,
    }


def all_novelty_hits(processed_inventory: list[dict[str, Any]]) -> dict[str, list[str]]:
    out = {family: set() for family in FEATURE_NOVELTY_TOKENS}
    for item in processed_inventory:
        hits = item.get("novelty_token_hits", {})
        for family, tokens in hits.items():
            out.setdefault(family, set()).update(tokens)
    return {family: sorted(tokens) for family, tokens in out.items()}


def closed_route_status(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    synthesis = reports["remaining_route_synthesis"].get("decision", {})
    priority = reports["external_research_priority"].get("decision", {})
    return {
        "all_reports_parse": all(report.get("parse_ok") is True for report in reports.values()),
        "registered_routes_closed": synthesis.get("all_registered_routes_closed"),
        "closed_route_count": synthesis.get("closed_route_count"),
        "route_count": synthesis.get("route_count"),
        "current_public_anchor_id": synthesis.get("current_public_anchor_id"),
        "current_public_score": synthesis.get("current_public_score"),
        "external_priority_ready": priority.get("external_research_priority_ready"),
        "same_family_heavy_rerun_allowed": priority.get("same_family_heavy_rerun_allowed"),
        "run_kaggle_now": synthesis.get("run_kaggle_now") or priority.get("run_kaggle_now"),
        "dacon_submit_now": synthesis.get("dacon_submit_now") or priority.get("dacon_submit_now"),
    }


def proposed_routes(novelty_hits: dict[str, list[str]]) -> list[dict[str, Any]]:
    return [
        {
            "route": "q3_app_name_stress_lexicon_v1",
            "priority": 1,
            "primary_targets": ["Q3"],
            "secondary_targets": ["Q2"],
            "source_modalities": ["mUsageStats", "mScreenStatus", "mAmbience"],
            "domain_link": (
                "Q3 is subjective pre-sleep stress relative to the subject average. Late app mix, "
                "communication bursts, finance/workload apps, and rapid app switching are plausible stress proxies."
            ),
            "why_materially_new": (
                "Existing usage features keep seven coarse categories and Markov/session summaries; processed "
                "matrices contain no raw app-name lexicon or stress-specific lexical aggregates."
            ),
            "novelty_evidence": {
                "processed_raw_app_name_lexicon_hits": novelty_hits.get("raw_app_name_lexicon", []),
                "required_for_novelty": "empty or lexicon-only aggregates absent from existing processed matrices",
            },
            "minimum_dryrun": [
                "Predeclare a small Korean/English app-name lexicon into stress/work/finance/social/media/routine buckets.",
                "Emit only aggregate counts, time shares, late-evening bursts, and entropy; do not persist raw app names.",
                "Join mScreenStatus and mAmbience only for late-screen-on and speech/noise context.",
            ],
            "gate": {
                "local_targets": ["Q3"],
                "must_beat_id342_on_primary_splits": "at least 4/5",
                "mean_delta_target_vs_id342_max": -0.002,
                "review_public_proxy_max": REVIEW_PUBLIC_THRESHOLD,
                "dacon_public_proxy_max": PUBLIC_055,
                "failed_frontier_corr_abs_max": 0.20,
            },
            "stop_conditions": [
                "If the lexicon route only improves Q2 or only produces calibration movement, close it.",
                "If raw names are needed in artifacts, stop and redesign into non-secret aggregates.",
                "If gains vanish under subject-hole or LOSO splits, do not create a submission CSV.",
            ],
        },
        {
            "route": "s3_s4_home_return_sleep_boundary_v1",
            "priority": 2,
            "primary_targets": ["S3", "S4"],
            "secondary_targets": ["Q3"],
            "source_modalities": ["mWifi", "mBle", "mGps", "mScreenStatus", "mACStatus"],
            "domain_link": (
                "S3/S4 are sleep-onset latency and wakefulness-after-sleep-onset. Late return home, unstable "
                "night-place anchors, and in-bed phone activity are plausible onset/fragmentation precursors."
            ),
            "why_materially_new": (
                "Existing GPS/proximity features count tokens and activity windows, but do not explicitly infer a "
                "subject-specific night-home anchor, last-home-arrival time, or home-stability deviation."
            ),
            "novelty_evidence": {
                "processed_home_anchor_hits": novelty_hits.get("home_anchor", []),
                "required_for_novelty": "home-anchor and home-return aggregates absent from existing processed matrices",
            },
            "minimum_dryrun": [
                "Infer a subject-level night-home anchor from recurring late-night WiFi/BLE/GPS observations.",
                "Build daily last-home-arrival, late-away-from-home, night anchor stability, and screen-after-return features.",
                "Use only train/test public raw sensor timing; no labels or target leakage in anchor construction.",
            ],
            "gate": {
                "local_targets": ["S3", "S4"],
                "must_beat_id342_on_primary_splits": "at least 4/5 for one focus target",
                "mean_delta_target_vs_id342_max": -0.002,
                "review_public_proxy_max": REVIEW_PUBLIC_THRESHOLD,
                "dacon_public_proxy_max": PUBLIC_055,
                "failed_frontier_corr_abs_max": 0.20,
            },
            "stop_conditions": [
                "If anchor coverage is below 85 percent of rows, close or restrict to report-only.",
                "If movement is only S4 weather/date-like movement, do not promote.",
                "If gains are only test-neighbor split wins and fail LOSO, do not create artifacts.",
            ],
        },
        {
            "route": "q_target_rank_quota_constraint_v1",
            "priority": 3,
            "primary_targets": ["Q1", "Q2", "Q3"],
            "secondary_targets": [],
            "source_modalities": ["metrics calendar", "OOF/test prediction shelf"],
            "domain_link": (
                "Q targets are binarized relative to a subject's own experimental-period average, so a pure static "
                "subject prior is not enough; any usable constraint must rank days within subject and calendar."
            ),
            "why_materially_new": (
                "Subject-prior calibration failed; this route is only distinct if it uses within-subject rank/order "
                "constraints plus independent raw-signal scores, not a direct push toward train subject means."
            ),
            "novelty_evidence": {
                "processed_subject_quota_hits": novelty_hits.get("subject_quota", []),
                "required_for_novelty": "rank/quota constraint absent from existing processed matrices",
            },
            "minimum_dryrun": [
                "Audit whether Q label rates by subject are stable enough to justify any quota model.",
                "Use independent raw-signal Q scores as ranks before any quota transform.",
                "Reject static mean forcing; it is already closed by the subject-prior gate.",
            ],
            "gate": {
                "local_targets": ["Q3"],
                "must_beat_id342_on_primary_splits": "at least 5/5 because this is a calibration-like route",
                "mean_delta_target_vs_id342_max": -0.003,
                "review_public_proxy_max": REVIEW_PUBLIC_THRESHOLD,
                "dacon_public_proxy_max": PUBLIC_055,
                "failed_frontier_corr_abs_max": 0.15,
            },
            "stop_conditions": [
                "If the route resembles subject-prior calibration, close it immediately.",
                "If no independent rank score exists, do not run a quota transform.",
                "If it changes only prediction mean without target-wise loss gain, do not promote.",
            ],
        },
    ]


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    now = datetime.now(KST)
    reports = {key: read_json(path) for key, path in CLOSED_REPORTS.items()}
    raw_inventory = raw_schema_inventory()
    processed_inventory = processed_schema_inventory()
    novelty_hits = all_novelty_hits(processed_inventory)
    routes = proposed_routes(novelty_hits)
    closed_status = closed_route_status(reports)
    packet_ready = (
        closed_status["all_reports_parse"]
        and bool(raw_inventory)
        and bool(processed_inventory)
        and closed_status["registered_routes_closed"] is True
    )
    return {
        "created_at_kst": now.strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_materially_new_source_hypothesis",
        "purpose": "Define non-repeated Q3/S3/S4 source mechanisms after registered post-ID342 routes closed.",
        "score_state": {
            "current_public_anchor_id": CURRENT_PUBLIC_ANCHOR_ID,
            "current_public_score": CURRENT_PUBLIC_SCORE,
            "target_public_score": PUBLIC_055,
            "gap_to_public_055": CURRENT_PUBLIC_SCORE - PUBLIC_055,
        },
        "closed_route_status": closed_status,
        "target_distribution_summary": target_distribution_summary(),
        "raw_schema_inventory": raw_inventory,
        "processed_schema_inventory": processed_inventory,
        "processed_novelty_token_hits": novelty_hits,
        "proposed_routes": routes,
        "do_not_repeat": [
            "alpha/temperature/subject-prior calibration without independent signal",
            "weather, air-quality, solar, and holiday date priors",
            "same-family missingness or subject-state heavy reruns without a changed local gate",
            "image-only or generic sequence-summary reruns",
            "tiny S3/S4 micro-overlays near Public 0.596",
        ],
        "decision": {
            "hypothesis_packet_ready": packet_ready,
            "top_route": routes[0]["route"],
            "route_count": len(routes),
            "score_candidate_now": False,
            "create_feature_matrix_now": False,
            "create_submission_artifact_now": False,
            "modify_submission_log_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "submit_dacon_now": False,
            "next_action": (
                "If paper-deadline work is stable, implement only the bounded dry-run for "
                "q3_app_name_stress_lexicon_v1; stop before artifacts unless the target gate clears."
            ),
        },
        "report_only": True,
        "no_raw_payload_full_scan": True,
        "no_feature_matrix_created": True,
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| priority | route | targets | why materially new |", "|---:|---|---|---|"]
    for row in rows:
        targets = ", ".join([*row["primary_targets"], *row["secondary_targets"]])
        lines.append(f"| {row['priority']} | `{row['route']}` | `{targets}` | {row['why_materially_new']} |")
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    score = report["score_state"]
    top_route = report["proposed_routes"][0]
    lines = [
        "# Post-ID342 materially new source hypothesis",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no feature matrix, predictions, submission artifact, Kaggle run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Hypothesis packet ready: `{decision['hypothesis_packet_ready']}`",
        f"- Top route: `{decision['top_route']}`",
        f"- Score candidate now: `{decision['score_candidate_now']}`",
        f"- Create feature matrix now: `{decision['create_feature_matrix_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['submit_dacon_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Score State",
        "",
        f"- Current public anchor: `ID{score['current_public_anchor_id']}` / `{score['current_public_score']:.10f}`",
        f"- Gap to Public 0.55: `{score['gap_to_public_055']:+.10f}`",
        "",
        "## Proposed Routes",
        "",
    ]
    lines.extend(markdown_table(report["proposed_routes"]))
    lines.extend(
        [
            "",
            "## Top Route Gate",
            "",
            "```json",
            json.dumps(
                {
                    "route": top_route["route"],
                    "minimum_dryrun": top_route["minimum_dryrun"],
                    "gate": top_route["gate"],
                    "stop_conditions": top_route["stop_conditions"],
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Processed Novelty Token Hits",
            "",
            "```json",
            json.dumps(report["processed_novelty_token_hits"], indent=2, sort_keys=True),
            "```",
            "",
            "## Target Distribution Summary",
            "",
            "```json",
            json.dumps(
                {
                    key: report["target_distribution_summary"][key]
                    for key in (
                        "train_rows",
                        "test_rows",
                        "train_subjects",
                        "test_subjects",
                        "train_date_min",
                        "train_date_max",
                        "test_date_min",
                        "test_date_max",
                        "target_means",
                    )
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Closed Route Status",
            "",
            "```json",
            json.dumps(report["closed_route_status"], indent=2, sort_keys=True),
            "```",
            "",
            "## Do Not Repeat",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["do_not_repeat"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    decision = report["decision"]
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(
        "Hypothesis packet ready: "
        f"{decision['hypothesis_packet_ready']} | top_route={decision['top_route']} | submit_now={decision['submit_dacon_now']}"
    )


if __name__ == "__main__":
    main()
