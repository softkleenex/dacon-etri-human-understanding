#!/usr/bin/env python3
"""Map Q3/S3 negative local evidence after post-ID341.

This report-only diagnostic aggregates the local no-approval Q3/S3 lanes that
have already been tested: physio-light-onset, semantic-routine, calendar/sleep
prior, subject-drift, raw event-cadence/silence-gap, place/social novelty,
routine-stability/place-entropy, and sensor-adherence/dropout. Its job is not
to find a new candidate; it prevents the workflow from repeatedly reopening
locally weak Q3/S3 families as if they were fresh Public-0.58 evidence.

It creates no predictions, submission CSV, base prediction artifact,
notebook/history artifact, submission-log row, Kaggle push/run, or DACON
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
KST = timezone(timedelta(hours=9))
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q3s3_negative_evidence_map_20260618.json"
TARGET_FOCUS = ["Q3", "S3"]
PRIMARY_ROUTE = "explicit_approved_clean_upstream_id221_to_id226"

INPUT_REPORTS = {
    "physio_light_onset": "post_id341_q3s3_physio_light_onset_smoke_20260618.json",
    "semantic_routine_closure": "post_id341_q3s3_semantic_routine_lane_closure_20260618.json",
    "calendar_sleep_prior": "post_id341_q3s3_calendar_sleep_prior_smoke_20260618.json",
    "subject_drift_gate": "post_id341_q3s3_subject_drift_gate_20260618.json",
    "event_cadence": "post_id341_q3s3_event_cadence_smoke_20260618.json",
    "place_social_novelty": "place_social_novelty_smoke_fixedlist_post_id341_20260618.json",
    "routine_stability_place_entropy": "routine_stability_place_entropy_smoke_post_id341_20260618.json",
    "sensor_adherence_dropout": "sensor_adherence_dropout_smoke_post_id341_20260618.json",
    "local_lane_closure": "post_id341_q3s3_local_lane_closure_20260618.json",
    "source_route": "post_id341_q2q3s3_source_route_20260618.json",
}
OPTIONAL_SCOUT_REPORTS = {
    "cross_modal_disagreement": "post_id341_q3s3_cross_modal_disagreement_smoke_20260618.json",
}


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    return str(path.relative_to(ROOT))


def load_report(name: str) -> dict[str, Any]:
    path = REPORTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {rel(path)}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_report(name: str) -> dict[str, Any] | None:
    path = REPORTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def lane_row(lane: str, status: str, q3: Any, s3: Any, reason: str, report_name: str) -> dict[str, Any]:
    return {
        "lane": lane,
        "status": status,
        "q3_evidence": q3,
        "s3_evidence": s3,
        "reason": reason,
        "report": rel(REPORTS_DIR / report_name),
        "artifact_allowed": False,
    }


def build_report() -> dict[str, Any]:
    reports = {key: load_report(name) for key, name in INPUT_REPORTS.items()}
    scout_reports = {
        key: payload
        for key, name in OPTIONAL_SCOUT_REPORTS.items()
        if (payload := load_optional_report(name)) is not None
    }
    physio = reports["physio_light_onset"]["decision"]
    semantic = reports["semantic_routine_closure"]["decision"]
    calendar = reports["calendar_sleep_prior"]["decision"]
    subject = reports["subject_drift_gate"]["decision"]
    cadence = reports["event_cadence"]["decision"]
    place_social = reports["place_social_novelty"]["decision"]
    routine_stability = reports["routine_stability_place_entropy"]["decision"]
    sensor_adherence = reports["sensor_adherence_dropout"]["decision"]
    closure = reports["local_lane_closure"]["decision"]
    route = reports["source_route"]["decision"]

    lane_rows = [
        lane_row(
            "physio_light_onset",
            "closed",
            physio.get("q3_signal_found"),
            physio.get("s3_signal_found"),
            "No selected Q3/S3 target from wearable/light/movement/onset features.",
            INPUT_REPORTS["physio_light_onset"],
        ),
        lane_row(
            "semantic_routine",
            "closed",
            semantic.get("semantic_routine_lane_closure_passed"),
            semantic.get("q3_s3_still_uncovered"),
            "State/routine lane was structurally computable but selected no Q3/S3 target.",
            INPUT_REPORTS["semantic_routine_closure"],
        ),
        lane_row(
            "calendar_sleep_prior",
            "closed",
            calendar.get("q3_signal_found"),
            calendar.get("s3_signal_found"),
            "Calendar, subject-progress, demographics, and materialized sleep-window summaries selected no target.",
            INPUT_REPORTS["calendar_sleep_prior"],
        ),
        lane_row(
            "subject_drift",
            "closed",
            f"{subject.get('q3_primary_wins_vs_id341')}/5 primary wins",
            f"{subject.get('s3_primary_wins_vs_id341')}/5 primary wins",
            "Subject residual drift is too weak for Q3/S3 and fails ID178/Public-0.58 promotion requirements.",
            INPUT_REPORTS["subject_drift_gate"],
        ),
        lane_row(
            "event_cadence",
            "closed",
            cadence.get("q3_signal_found"),
            cadence.get("s3_signal_found"),
            "Raw timestamp cadence, silence gaps, and inter-event intervals selected no Q3/S3 target.",
            INPUT_REPORTS["event_cadence"],
        ),
        lane_row(
            "place_social_novelty",
            "closed",
            place_social.get("selected_targets"),
            place_social.get("reviewable_low_movement_candidate_exists"),
            "GPS/WiFi/BLE place-social novelty selected no Q3/S3 target and has no reviewable low-movement row.",
            INPUT_REPORTS["place_social_novelty"],
        ),
        lane_row(
            "routine_stability_place_entropy",
            "closed",
            routine_stability.get("selected_targets"),
            routine_stability.get("reviewable_low_movement_candidate_exists"),
            "Routine-stability/place-entropy selected no Q3/S3 target and has no proxy-below-anchor row.",
            INPUT_REPORTS["routine_stability_place_entropy"],
        ),
        lane_row(
            "sensor_adherence_dropout",
            "closed",
            sensor_adherence.get("selected_targets"),
            sensor_adherence.get("reviewable_low_movement_candidate_exists"),
            "Sensor-adherence/dropout selected no Q3/S3 target and is closed as a standalone local lane.",
            INPUT_REPORTS["sensor_adherence_dropout"],
        ),
    ]
    scout_rows = []
    if cross_modal := scout_reports.get("cross_modal_disagreement"):
        cross_decision = cross_modal["decision"]
        scout_rows.append(
            lane_row(
                "cross_modal_disagreement_scout",
                "closed_scout",
                cross_decision.get("q3_signal_found"),
                cross_decision.get("s3_signal_found"),
                (
                    "Phone/watch cross-modal disagreement selected no Q3/S3 target; kept as optional scout "
                    "evidence without changing the authoritative 8-lane prelaunch contract."
                ),
                OPTIONAL_SCOUT_REPORTS["cross_modal_disagreement"],
            )
        )
    all_lanes_closed = all(row["status"] == "closed" and row["artifact_allowed"] is False for row in lane_rows)
    scout_reports_closed = all(
        row["status"] == "closed_scout" and row["artifact_allowed"] is False for row in scout_rows
    )
    no_public_058_evidence = all(
        reports[key].get("decision", {}).get("public_058_evidence") is not True
        for key in [
            "physio_light_onset",
            "semantic_routine_closure",
            "calendar_sleep_prior",
            "subject_drift_gate",
            "event_cadence",
            "place_social_novelty",
            "routine_stability_place_entropy",
            "sensor_adherence_dropout",
            "local_lane_closure",
        ]
    ) and all(payload.get("decision", {}).get("public_058_evidence") is not True for payload in scout_reports.values())
    local_closure_contract = bool(
        closure.get("local_fallback_closed_for_submission_artifact") is True
        and closure.get("q3_s3_uncovered_after_local_fallback") == TARGET_FOCUS
        and closure.get("local_artifact_allowed") is False
        and closure.get("primary_next_score_relevant_route") == PRIMARY_ROUTE
    )
    route_contract = bool(
        route.get("primary_score_relevant_route") == PRIMARY_ROUTE
        and route.get("remote_no_submit_prelaunch_ready") is True
        and route.get("local_q2q3s3_artifact_allowed") is False
    )
    map_passed = bool(all_lanes_closed and no_public_058_evidence and local_closure_contract and route_contract)

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q3s3_negative_evidence_map",
        "purpose": (
            "Aggregate local Q3/S3 no-approval evidence so weak or closed source families are not repeatedly "
            "treated as artifact candidates."
        ),
        "target_focus": TARGET_FOCUS,
        "inputs": {key: rel(REPORTS_DIR / name) for key, name in INPUT_REPORTS.items()},
        "optional_scout_inputs": {
            key: rel(REPORTS_DIR / name)
            for key, name in OPTIONAL_SCOUT_REPORTS.items()
            if (REPORTS_DIR / name).exists()
        },
        "lane_rows": lane_rows,
        "scout_rows": scout_rows,
        "checks": {
            "all_local_q3s3_lanes_closed": all_lanes_closed,
            "optional_scout_reports_closed": scout_reports_closed,
            "no_public_058_evidence": no_public_058_evidence,
            "local_closure_contract": local_closure_contract,
            "route_contract": route_contract,
        },
        "decision": {
            "q3s3_negative_evidence_map_passed": map_passed,
            "target_focus": TARGET_FOCUS,
            "closed_local_lane_count": int(sum(row["status"] == "closed" for row in lane_rows)),
            "closed_optional_scout_count": int(sum(row["status"] == "closed_scout" for row in scout_rows)),
            "all_local_q3s3_lanes_closed": all_lanes_closed,
            "q3_s3_still_uncovered": True,
            "public_058_evidence": False,
            "local_artifact_allowed": False,
            "primary_next_score_relevant_route": PRIMARY_ROUTE,
            "requires_explicit_kaggle_approval": True,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Do not reopen closed local Q3/S3 lanes for artifact creation. The score-relevant path remains the "
                "explicit-approved clean ID221→ID226 no-submit upstream rebuild; without approval, only a materially "
                "new Q3/S3 source family should be tested."
            ),
        },
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Post-ID341 Q3/S3 negative evidence map",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Closed local lanes",
        "",
        "| lane | status | Q3 evidence | S3 evidence | reason |",
        "|---|---|---|---|---|",
    ]
    for row in report["lane_rows"]:
        lines.append(
            f"| `{row['lane']}` | `{row['status']}` | `{row['q3_evidence']}` | "
            f"`{row['s3_evidence']}` | {row['reason']} |"
        )
    if report["scout_rows"]:
        lines.extend(
            [
                "",
                "## Optional closed scouts",
                "",
                "| scout | status | Q3 evidence | S3 evidence | reason |",
                "|---|---|---|---|---|",
            ]
        )
        for row in report["scout_rows"]:
            lines.append(
                f"| `{row['lane']}` | `{row['status']}` | `{row['q3_evidence']}` | "
                f"`{row['s3_evidence']}` | {row['reason']} |"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Negative evidence map passed: `{decision['q3s3_negative_evidence_map_passed']}`",
            f"- Closed local lane count: `{decision['closed_local_lane_count']}`",
            f"- Closed optional scout count: `{decision['closed_optional_scout_count']}`",
            f"- All local Q3/S3 lanes closed: `{decision['all_local_q3s3_lanes_closed']}`",
            f"- Q3/S3 still uncovered: `{decision['q3_s3_still_uncovered']}`",
            f"- Public `0.58` evidence: `{decision['public_058_evidence']}`",
            f"- Local artifact allowed: `{decision['local_artifact_allowed']}`",
            f"- Primary next score-relevant route: `{decision['primary_next_score_relevant_route']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
