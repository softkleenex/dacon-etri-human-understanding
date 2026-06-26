#!/usr/bin/env python3
"""Close the local Q3/S3 fallback lane after no-approval local smokes.

This report synthesizes the Q2 lead, Q3/S3 physio-light-onset failure,
calendar/sleep-prior failure, and the existing Q2/Q3/S3 route contract. It is
report-only and authorizes no artifact, Kaggle execution, or DACON submission.
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
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q3s3_local_lane_closure_20260618.json"

INPUT_REPORTS = {
    "source_route": "post_id341_q2q3s3_source_route_20260618.json",
    "env_social_model_smoke": "post_id341_q2q3s3_fold_internal_model_smoke_20260618.json",
    "q2_geometry_gate": "post_id341_q2q3s3_q2_lead_geometry_gate_20260618.json",
    "physio_light_onset": "post_id341_q3s3_physio_light_onset_smoke_20260618.json",
    "calendar_sleep_prior": "post_id341_q3s3_calendar_sleep_prior_smoke_20260618.json",
    "subject_drift_gate": "post_id341_q3s3_subject_drift_gate_20260618.json",
    "external_prior": "external_etri_prior_smoke_post_id341_20260618.json",
}
MISSING_TARGETS = ["Q2", "Q3", "S3"]
UNCOVERED_AFTER_LOCAL = ["Q3", "S3"]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_report(name: str) -> dict[str, Any]:
    path = REPORTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict[str, Any]:
    reports = {key: load_report(name) for key, name in INPUT_REPORTS.items()}
    route_decision = reports["source_route"]["decision"]
    q2_model_decision = reports["env_social_model_smoke"]["decision"]
    q2_geometry_decision = reports["q2_geometry_gate"]["decision"]
    physio_decision = reports["physio_light_onset"]["decision"]
    calendar_decision = reports["calendar_sleep_prior"]["decision"]
    subject_drift_decision = reports["subject_drift_gate"]["decision"]
    external_decision = reports["external_prior"]["decision"]

    q2_component_context = {
        "selected_in_model_smoke": q2_model_decision.get("selected_targets") == ["Q2"],
        "reviewable_low_movement": q2_geometry_decision.get("reviewable_low_movement_candidate_exists") is True,
        "best_proxy": q2_geometry_decision.get("best_proxy"),
        "public_058_plausible": q2_geometry_decision.get("public_058_plausible") is True,
        "artifact_allowed": False,
    }
    q3s3_physio_closed = (
        physio_decision.get("selected_targets") == []
        and physio_decision.get("q3_signal_found") is False
        and physio_decision.get("s3_signal_found") is False
    )
    q3s3_calendar_closed = (
        calendar_decision.get("selected_targets") == []
        and calendar_decision.get("q3_signal_found") is False
        and calendar_decision.get("s3_signal_found") is False
    )
    q3s3_subject_drift_closed = (
        subject_drift_decision.get("q3s3_subject_drift_gate_passed") is True
        and subject_drift_decision.get("q3s3_subject_drift_artifact_allowed") is False
        and subject_drift_decision.get("q3s3_subject_drift_closed_for_artifact") is True
    )
    q3s3_closed = q3s3_physio_closed and q3s3_calendar_closed and q3s3_subject_drift_closed
    local_artifact_allowed = False
    upstream_ready = route_decision.get("remote_no_submit_prelaunch_ready") is True
    external_regularizer_only = not bool(external_decision.get("reviewable_candidate_exists"))

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q3s3_local_lane_closure",
        "purpose": "Decide whether the no-approval local Q2/Q3/S3 fallback still has a score-relevant path after Q2-only and Q3/S3 physio-light evidence.",
        "inputs": {key: rel(REPORTS_DIR / name) for key, name in INPUT_REPORTS.items()},
        "evidence": {
            "missing_targets_before_lane": MISSING_TARGETS,
            "q2_component_context": q2_component_context,
            "q3s3_physio_light_onset_closed": q3s3_physio_closed,
            "q3s3_calendar_sleep_prior_closed": q3s3_calendar_closed,
            "q3s3_subject_drift_closed": q3s3_subject_drift_closed,
            "external_prior_regularizer_only": external_regularizer_only,
            "primary_route": route_decision.get("primary_score_relevant_route"),
            "fallback_route": route_decision.get("fallback_no_approval_route"),
            "remote_no_submit_prelaunch_ready": upstream_ready,
        },
        "decision": {
            "local_q2_component_keep_as_context": bool(q2_component_context["reviewable_low_movement"]),
            "q3_s3_uncovered_after_local_fallback": UNCOVERED_AFTER_LOCAL,
            "local_fallback_closed_for_submission_artifact": bool(
                q3s3_closed and not q2_component_context["public_058_plausible"]
            ),
            "local_artifact_allowed": local_artifact_allowed,
            "public_058_evidence": False,
            "primary_next_score_relevant_route": route_decision.get("primary_score_relevant_route"),
            "requires_explicit_kaggle_approval": True,
            "prelaunch_ready_for_explicit_approval": upstream_ready,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Do not create a local submission artifact from Q2-only context. The next score-relevant path is the "
                "explicit-approved clean ID221→ID226 no-submit upstream rebuild; without that approval, only document "
                "new genuinely independent Q3/S3 hypotheses, not broader local tuning."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_made": True,
        "no_kaggle_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    evidence = report["evidence"]
    decision = report["decision"]
    q2 = evidence["q2_component_context"]
    lines = [
        "# Q3/S3 local lane closure — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Evidence",
        "",
        f"- Missing targets before local fallback: `{', '.join(evidence['missing_targets_before_lane'])}`",
        f"- Q2 component selected and low-movement: `{q2['selected_in_model_smoke']}` / `{q2['reviewable_low_movement']}`",
        f"- Q2 best proxy: `{q2['best_proxy']}`; Public `0.58` plausible: `{q2['public_058_plausible']}`",
        f"- Q3/S3 physio-light-onset closed: `{evidence['q3s3_physio_light_onset_closed']}`",
        f"- Q3/S3 calendar/sleep-prior closed: `{evidence['q3s3_calendar_sleep_prior_closed']}`",
        f"- Q3/S3 subject-drift closed: `{evidence['q3s3_subject_drift_closed']}`",
        f"- External prior status: regularizer-only `{evidence['external_prior_regularizer_only']}`",
        f"- Remote no-submit prelaunch ready: `{evidence['remote_no_submit_prelaunch_ready']}`",
        "",
        "## Decision",
        "",
        f"- Keep Q2 as context: `{decision['local_q2_component_keep_as_context']}`",
        f"- Q3/S3 still uncovered: `{decision['q3_s3_uncovered_after_local_fallback']}`",
        f"- Local fallback closed for submission artifact: `{decision['local_fallback_closed_for_submission_artifact']}`",
        f"- Public `0.58` evidence: `{decision['public_058_evidence']}`",
        f"- Primary next score-relevant route: `{decision['primary_next_score_relevant_route']}`",
        f"- Requires explicit Kaggle approval: `{decision['requires_explicit_kaggle_approval']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
    ]
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
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
