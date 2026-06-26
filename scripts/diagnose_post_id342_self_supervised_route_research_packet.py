#!/usr/bin/env python3
"""Post-ID342 self-supervised subject-state route research packet.

This report-only diagnostic translates the remaining post-ID342 route into an
execution-ready but still blocked research packet. It reads only existing report
JSON files and records domain rationale, success gates, and approval boundaries.

It does not read full raw streams, fit embeddings, write feature matrices,
train models, create prediction or submission artifacts, edit the submission
log, push/run Kaggle, or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_self_supervised_route_research_packet_20260624.json"
KST = timezone(timedelta(hours=9))

HEAVY_APPROVAL_PHRASE = "Self-supervised subject-state heavy compute no-submit 실행 승인"
KAGGLE_UPSTREAM_APPROVAL_PHRASE = "Kaggle upstream rebuild no-submit 실행 승인"

REPORT_PATHS = {
    "remaining_route_synthesis": REPORTS_DIR / "post_id342_remaining_route_synthesis_20260624.json",
    "self_supervised_design": REPORTS_DIR / "post_id341_self_supervised_subject_state_embedding_design_20260619.json",
    "heavy_compute_contract": REPORTS_DIR / "post_id342_self_supervised_heavy_compute_contract_20260624.json",
    "public_alignment_lens": REPORTS_DIR / "post_id342_public_alignment_validation_lens_20260624.json",
    "lightweight_subject_state_pca": REPORTS_DIR / "post_id342_lightweight_subject_state_pca_gate_20260624.json",
    "processed_sleep_temporal_deviation": REPORTS_DIR
    / "post_id342_processed_sleep_temporal_deviation_gate_20260624.json",
    "probability_temperature": REPORTS_DIR / "post_id342_probability_temperature_gate_20260624.json",
    "kaggle_remote_summary": REPORTS_DIR / "post_id341_kaggle_remote_result_summary_20260618.json",
}

RESEARCH_SOURCES = [
    {
        "source": "ETRI Human Understanding AI Paper Challenge 2024 dataset design",
        "url": "https://arxiv.org/html/2403.16509v1",
        "takeaway": (
            "Train/validation/test cohorts differ, validation/test participants are relabeled, "
            "and Q3/S3 are participant-relative or guideline-derived targets."
        ),
    },
    {
        "source": "DACON 4th ETRI Human Understanding data page",
        "url": "https://dacon.io/en/competitions/official/236468/data",
        "takeaway": (
            "The public data description emphasizes 12 lifelog streams from smartphone, smartwatch, "
            "sleep sensor, and self-recording app sources."
        ),
    },
    {
        "source": "PixleepFlow",
        "url": "https://arxiv.org/html/2502.17469v2",
        "takeaway": (
            "Pixel/image-style multimodal lifelog representations are motivated by sensor synchronization, "
            "sleep/stress target coupling, and interpretable channel importance."
        ),
    },
]

FORBIDDEN_ACTIONS_NOW = {
    "read_full_raw_streams_now": False,
    "fit_embedding_now": False,
    "write_feature_matrix_now": False,
    "train_downstream_model_now": False,
    "write_oof_or_test_predictions_now": False,
    "create_submission_artifact": False,
    "modify_submission_log": False,
    "push_kaggle_now": False,
    "run_kaggle_now": False,
    "dacon_submit_now": False,
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_report(path: Path) -> dict[str, Any]:
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
    payload["path"] = rel(path)
    payload["exists"] = True
    payload["parse_ok"] = True
    return payload


def decision(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("decision", {})
    return value if isinstance(value, dict) else {}


def find_route(report: dict[str, Any], route_name: str) -> dict[str, Any]:
    rows = report.get("route_rows", [])
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and row.get("route") == route_name:
            return row
    return {}


def domain_signal_map() -> list[dict[str, Any]]:
    return [
        {
            "target": "Q3",
            "meaning": "Stress before sleep; in the ETRI 2024 target definition, 1 means below the participant average.",
            "primary_modalities": ["mUsageStats", "mScreenStatus", "mActivity", "wHr", "mAmbience", "mGps"],
            "why_existing_lightweight_failed": (
                "Single-family routine, evening, processed-sleep, and PCA overlays did not capture a stable "
                "subject-specific stress state under the ID342 lenses."
            ),
            "heavy_route_hypothesis": (
                "Learn label-free day/evening state tokens that combine app usage, phone rest, movement, HR, "
                "ambient context, and location routine before fold-internal Q3 testing."
            ),
        },
        {
            "target": "S3",
            "meaning": "Sleep onset latency guideline adherence; 1 means the sleep record meets the guideline.",
            "primary_modalities": ["mScreenStatus", "mACStatus", "mLight", "wLight", "mActivity", "wPedo", "wHr"],
            "why_existing_lightweight_failed": (
                "Sleep-boundary, processed sleep lag/deviation, and tiny temporal-CNN proxies selected no stable S3 "
                "target against ID342."
            ),
            "heavy_route_hypothesis": (
                "Represent the transition into sleep as a synchronized phone-wearable context state rather than "
                "hand-built SOL or single-channel boundary summaries."
            ),
        },
        {
            "target": "Q2",
            "meaning": "Pre-sleep affect/fatigue-style subjective state depending on the competition-year target spec.",
            "primary_modalities": ["mUsageStats", "mActivity", "mGps", "mAmbience", "wHr"],
            "why_existing_lightweight_failed": (
                "Env/social/routine opened only a narrow Q2 micro-signal; that scale is far from the Public 0.55 gap."
            ),
            "heavy_route_hypothesis": (
                "Use Q2 as a secondary sanity target: it should improve only if the representation learns real "
                "day-state structure and does not overfit public-feedback geometry."
            ),
        },
    ]


def route_success_gates() -> list[dict[str, Any]]:
    return [
        {
            "gate": "approval",
            "required_before_execution": True,
            "criterion": f"Exact phrase required: {HEAVY_APPROVAL_PHRASE}",
            "blocks": ["full raw read", "embedding fit", "Kaggle run"],
        },
        {
            "gate": "label_free_pretraining",
            "required_before_artifact_review": True,
            "criterion": "No target labels, no OOF predictions, no public-score feedback, and no donor submission predictions in embedding selection.",
            "blocks": ["OOF recombination", "ID287/ID329 direction tuning", "post-public target swap"],
        },
        {
            "gate": "fold_internal_smoke",
            "required_before_artifact_review": True,
            "criterion": "Q3/S3 must beat ID342 on stable near-anchor lenses and must not win only under failed-frontier-inclusive views.",
            "blocks": ["local-frontier-only gain", "subject-slice donor selector trap"],
        },
        {
            "gate": "public_gap_budget",
            "required_before_artifact_review": True,
            "criterion": "Projected public proxy must show material movement beyond ID342, not another 0.595-0.596 micro-change.",
            "blocks": ["same-family Kaggle rerun", "tiny calibration/temperature/alpha edit"],
        },
        {
            "gate": "artifact_contract",
            "required_before_submission_file": True,
            "criterion": "A separate future artifact contract must name the prediction artifact, log row, notebook/source, and no-submit review.",
            "blocks": ["submission CSV", "submission_log.csv edit", "DACON submission"],
        },
    ]


def execution_boundary() -> list[dict[str, Any]]:
    return [
        {
            "stage": "current_packet",
            "allowed_now": True,
            "output": "report-only research packet",
            "guard": "Reads existing reports only.",
        },
        {
            "stage": "heavy_contract_refresh",
            "allowed_now": False,
            "after_exact_heavy_approval": True,
            "output": "refreshed no-submit contract",
            "guard": "Still no embedding fit until contract is refreshed and passes.",
        },
        {
            "stage": "kaggle_no_submit_embedding_run",
            "allowed_now": False,
            "after_exact_heavy_approval": True,
            "output": "ignored build/artifact embeddings plus report-only fold smoke",
            "guard": "No submission artifact or log row.",
        },
        {
            "stage": "artifact_review",
            "allowed_now": False,
            "after_exact_heavy_approval": False,
            "output": "none",
            "guard": "Requires a later artifact contract and exact user approval.",
        },
        {
            "stage": "dacon_submission",
            "allowed_now": False,
            "after_exact_heavy_approval": False,
            "output": "none",
            "guard": "Requires exact DACON file/memo approval; this packet does not provide it.",
        },
    ]


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    reports = {key: load_report(path) for key, path in REPORT_PATHS.items()}
    synthesis = reports["remaining_route_synthesis"]
    synthesis_decision = decision(synthesis)
    heavy_route = find_route(synthesis, "self_supervised_subject_state_embedding")
    design_decision = decision(reports["self_supervised_design"])
    heavy_decision = decision(reports["heavy_compute_contract"])
    pca_decision = decision(reports["lightweight_subject_state_pca"])
    remote_decision = decision(reports["kaggle_remote_summary"])

    decision_payload = {
        "research_packet_ready": True,
        "current_public_anchor_id": synthesis_decision.get("current_public_anchor_id"),
        "current_public_score": synthesis_decision.get("current_public_score"),
        "target_public_score": synthesis_decision.get("target_public_score"),
        "gap_to_public_055": synthesis_decision.get("gap_to_public_055"),
        "post_id342_closed_routes": synthesis_decision.get("closed_route_count"),
        "post_id342_route_count": synthesis_decision.get("route_count"),
        "post_id342_missing_or_review_routes": synthesis_decision.get("missing_or_review_route_count"),
        "heavy_route_is_only_non_closed_route": heavy_route.get("status")
        == "deferred_requires_exact_heavy_compute_approval",
        "post_id341_design_ready": design_decision.get("design_ready"),
        "post_id342_heavy_contract_ready": heavy_decision.get(
            "post_id342_heavy_compute_contract_ready", heavy_decision.get("heavy_compute_contract_ready")
        ),
        "dedicated_heavy_runner_required": heavy_decision.get("dedicated_heavy_runner_required"),
        "existing_upstream_runner_is_sufficient_for_heavy_execution": heavy_decision.get(
            "existing_upstream_runner_is_sufficient_for_heavy_execution"
        ),
        "lightweight_pca_pass_count": pca_decision.get("eligible_row_count"),
        "remote_promotable_evidence": bool(
            remote_decision.get("public_058_evidence_present")
            or remote_decision.get("beats_id178_public_anchor_evidence_present")
        ),
        "requires_exact_heavy_compute_approval": True,
        "heavy_approval_phrase": HEAVY_APPROVAL_PHRASE,
        "kaggle_upstream_approval_phrase_is_not_enough": True,
        "kaggle_upstream_approval_phrase": KAGGLE_UPSTREAM_APPROVAL_PHRASE,
        "heavy_compute_allowed_now": False,
        "create_submission_artifact_now": False,
        "modify_submission_log_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "next_action": (
            "Do not execute heavy compute yet. If the exact heavy approval phrase is provided later, "
            "refresh the heavy-compute contract first, then run only no-submit label-free stages."
        ),
    }

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_self_supervised_route_research_packet",
        "purpose": "Convert the only remaining post-ID342 route into a blocked, auditable research packet.",
        "research_sources": RESEARCH_SOURCES,
        "input_reports": reports,
        "domain_signal_map": domain_signal_map(),
        "route_success_gates": route_success_gates(),
        "execution_boundary": execution_boundary(),
        "decision": decision_payload,
        "forbidden_actions_now": FORBIDDEN_ACTIONS_NOW,
        "no_full_raw_stream_read": True,
        "no_embedding_fit": True,
        "no_feature_matrix_created": True,
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action": True,
        "no_dacon_submission": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision_payload = report["decision"]
    lines = [
        "# Post-ID342 self-supervised route research packet",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It reads existing reports only and creates no raw-stream cache, embedding, feature matrix, model, prediction, submission artifact, Kaggle run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Current anchor: `ID{decision_payload['current_public_anchor_id']}` / `{decision_payload['current_public_score']:.10f}`",
        f"- Gap to Public 0.55: `{decision_payload['gap_to_public_055']:+.10f}`",
        f"- Post-ID342 closed routes: `{decision_payload['post_id342_closed_routes']}/{decision_payload['post_id342_route_count']}`",
        f"- Heavy route is only non-closed route: `{decision_payload['heavy_route_is_only_non_closed_route']}`",
        f"- Heavy compute allowed now: `{decision_payload['heavy_compute_allowed_now']}`",
        f"- Dedicated heavy runner required: `{decision_payload['dedicated_heavy_runner_required']}`",
        f"- Existing upstream runner sufficient: `{decision_payload['existing_upstream_runner_is_sufficient_for_heavy_execution']}`",
        f"- Required phrase: `{decision_payload['heavy_approval_phrase']}`",
        f"- Kaggle upstream phrase is not enough: `{decision_payload['kaggle_upstream_approval_phrase_is_not_enough']}`",
        f"- Push Kaggle now: `{decision_payload['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision_payload['run_kaggle_now']}`",
        f"- DACON submit now: `{decision_payload['dacon_submit_now']}`",
        f"- Next action: {decision_payload['next_action']}",
        "",
        "## Domain Signal Map",
        "",
        "| target | meaning | primary modalities | heavy-route hypothesis |",
        "|---|---|---|---|",
    ]
    for row in report["domain_signal_map"]:
        lines.append(
            f"| `{row['target']}` | {row['meaning']} | {', '.join(row['primary_modalities'])} | {row['heavy_route_hypothesis']} |"
        )

    lines.extend(
        [
            "",
            "## Success Gates",
            "",
            "| gate | required before execution/artifact | criterion | blocks |",
            "|---|---:|---|---|",
        ]
    )
    for row in report["route_success_gates"]:
        required = (
            row.get("required_before_execution")
            or row.get("required_before_artifact_review")
            or row.get("required_before_submission_file")
        )
        lines.append(f"| `{row['gate']}` | `{bool(required)}` | {row['criterion']} | {', '.join(row['blocks'])} |")

    lines.extend(
        [
            "",
            "## Research Sources",
            "",
            "| source | url | takeaway |",
            "|---|---|---|",
        ]
    )
    for row in report["research_sources"]:
        lines.append(f"| {row['source']} | {row['url']} | {row['takeaway']} |")

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
    print(f"Heavy compute allowed now: {report['decision']['heavy_compute_allowed_now']}")


if __name__ == "__main__":
    main()
