#!/usr/bin/env python3
"""Summarize the post-ID342 research gap and next source priority.

This report is intentionally non-generative: it trains no model, creates no
prediction files, and makes no Kaggle or DACON external action. Its job is to
turn the corrected ID342 public feedback into an operator-facing research queue.
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

DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_research_gap_and_source_priority_20260624.json"

ID178_PUBLIC = 0.6014113009
ID337_PUBLIC = 0.6006206286219425
ID342_PUBLIC = 0.5960832519
TARGET_PUBLIC = 0.55


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "error": None}
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": rel(path),
            "exists": True,
            "parse_ok": False,
            "bytes": path.stat().st_size,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": rel(path),
        "exists": True,
        "parse_ok": True,
        "bytes": path.stat().st_size,
        "diagnostic": payload.get("diagnostic"),
        "decision": payload.get("decision", {}),
        "error": None,
    }


def get_nested(obj: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_optional(path: Path) -> dict[str, Any]:
    return load_json(path) if path.exists() else {}


def build_source_priority_rows() -> list[dict[str, Any]]:
    return [
        {
            "rank": 1,
            "route": "raw_temporal_q3s3_sequence_image",
            "status": "open_next",
            "target_focus": ["Q3", "S3"],
            "why": (
                "Closed aggregate local lanes did not open Q3, while ID342 showed that same-subject behavioral "
                "similarity can transfer publicly. A low-resolution temporal representation can test the same idea "
                "without replaying the historical ID221 full-OOF selector."
            ),
            "required_evidence": (
                "Report-only fold-internal Q3/S3 diagnostic must beat ID342 on primary lenses with low ID287/ID329 "
                "exposure before any artifact review."
            ),
            "kaggle_fit": "good_cpu_first_no_submit",
            "dacon_submit_allowed": False,
        },
        {
            "rank": 2,
            "route": "paper_baseline_time_zone_rebuild_id342_aware",
            "status": "open_supporting",
            "target_focus": ["Q3", "S3", "Q1", "S4"],
            "why": (
                "The official ETRI paper used fixed time-zone descriptors and app categories. Rebuilding that lens "
                "against ID342 residuals is cheap and can provide interpretable fallback evidence."
            ),
            "required_evidence": "Must be nested and target-specific; broad aggregate replay is already weak.",
            "kaggle_fit": "local_or_kaggle_cpu",
            "dacon_submit_allowed": False,
        },
        {
            "rank": 3,
            "route": "self_supervised_subject_state_embedding",
            "status": "heavy_deferred",
            "target_focus": ["Q2", "Q3", "S3"],
            "why": (
                "All 12 raw modalities are available and the target bottleneck is subject-relative state. This is "
                "materially different, but it is a heavier training contract than the current overnight first pass."
            ),
            "required_evidence": "Separate heavy-compute contract; no artifact without a no-submit review report.",
            "kaggle_fit": "kaggle_cpu_or_gpu_later",
            "dacon_submit_allowed": False,
        },
        {
            "rank": 4,
            "route": "material_feature_micro_overlays",
            "status": "deprioritized",
            "target_focus": ["S3"],
            "why": (
                "Phone-wearable and GPS matrices show only tiny S3 signals and Q3 remains closed. This cannot plausibly "
                "explain the gap from Public 0.596 to 0.55 by itself."
            ),
            "required_evidence": "Only useful as a small companion after an independent raw/sequence signal appears.",
            "kaggle_fit": "not_primary",
            "dacon_submit_allowed": False,
        },
    ]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "id342_experiment": REPORTS_DIR / "experiment_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.json",
        "dacon_three_slot_result": REPORTS_DIR / "dacon_three_slot_submission_result_20260623.json",
        "q3s3_target_anatomy": REPORTS_DIR / "post_id341_q3s3_target_anatomy_20260619.json",
        "q3s3_negative_evidence": REPORTS_DIR / "post_id341_q3s3_negative_evidence_map_20260618.json",
        "public060_wall": REPORTS_DIR / "post_id341_public060_wall_root_cause_20260621.json",
        "clean_backbone_lane_closure": REPORTS_DIR / "post_id341_clean_backbone_lane_closure_20260618.json",
        "phase2_block_a": REPORTS_DIR / "phase2_block_a_q1_s4_sleep_residual_bridge_post_id341_20260618.json",
        "phase2_block_b": REPORTS_DIR / "phase2_block_b_s2_sleep_recovery_retry_post_id341_20260618.json",
        "phase3_usage_temporal": REPORTS_DIR / "phase3_usage_temporal_microstructure_smoke_post_id341_20260618.json",
        "remote_summary": REPORTS_DIR / "post_id341_kaggle_remote_result_summary_20260618.json",
    }
    inputs = {name: status(path) for name, path in paths.items()}

    id342 = load_optional(paths["id342_experiment"])
    slate = load_optional(paths["dacon_three_slot_result"])
    anatomy = load_optional(paths["q3s3_target_anatomy"])
    negative = load_optional(paths["q3s3_negative_evidence"])
    wall = load_optional(paths["public060_wall"])
    clean_close = load_optional(paths["clean_backbone_lane_closure"])
    block_a = load_optional(paths["phase2_block_a"])
    block_b = load_optional(paths["phase2_block_b"])
    usage = load_optional(paths["phase3_usage_temporal"])
    remote = load_optional(paths["remote_summary"])

    anchor = get_nested(slate, ["confirmed_team_best_after_batch"], {})
    target_scores = id342.get("target_scores", {})
    q3s3 = get_nested(anatomy, ["anatomy_read"], {})
    negative_decision = negative.get("decision", {})
    wall_decision = wall.get("decision", {})
    clean_decision = clean_close.get("decision", {})
    remote_decision = remote.get("decision", {})

    source_priority_rows = build_source_priority_rows()
    id342_gap_to_target = ID342_PUBLIC - TARGET_PUBLIC
    id342_delta_vs_id178 = ID342_PUBLIC - ID178_PUBLIC
    id342_delta_vs_id337 = ID342_PUBLIC - ID337_PUBLIC

    evidence = {
        "anchor": {
            "id": int(anchor.get("id", 342)),
            "public_score": float(anchor.get("public_score", ID342_PUBLIC)),
            "local_cv": float(id342.get("local_cv", 0.5340000057700848)),
            "dacon_submission_id": anchor.get("dacon_submission_id", 1467937),
            "leaderboard_time": anchor.get("c_time", "2026-06-23 23:30:37"),
            "delta_vs_id178_public": id342_delta_vs_id178,
            "delta_vs_id337_public": id342_delta_vs_id337,
            "gap_to_public_055": id342_gap_to_target,
        },
        "id342_target_scores": target_scores,
        "target_bottleneck": {
            "q3_score": target_scores.get("Q3"),
            "s3_score": target_scores.get("S3"),
            "s4_score": target_scores.get("S4"),
            "q3_positive_rate": q3s3.get("q3_positive_rate"),
            "s3_positive_rate": q3s3.get("s3_positive_rate"),
            "q3_lag1_agreement_rate": q3s3.get("q3_lag1_agreement_rate"),
            "s3_lag1_agreement_rate": q3s3.get("s3_lag1_agreement_rate"),
            "q3_top_same_day_correlation": q3s3.get("q3_top_same_day_correlation"),
            "s3_top_same_day_correlation": q3s3.get("s3_top_same_day_correlation"),
        },
        "closed_lanes": {
            "all_local_q3s3_lanes_closed": negative_decision.get("all_local_q3s3_lanes_closed"),
            "closed_local_lane_count": negative_decision.get("closed_local_lane_count"),
            "closed_optional_scout_count": negative_decision.get("closed_optional_scout_count"),
            "local_artifact_allowed": negative_decision.get("local_artifact_allowed"),
        },
        "remote_v3": {
            "clean_backbone_closed": clean_decision.get("close_clean_backbone_lane_now"),
            "next_lane": clean_decision.get("next_lane"),
            "block_a_selected_targets": get_nested(block_a, ["decision", "selected_targets"], []),
            "block_a_final_gate_pass": get_nested(block_a, ["decision", "final_block_a_gate_pass"]),
            "block_b_final_gate_pass": get_nested(block_b, ["decision", "final_block_b_gate_pass"]),
            "usage_temporal_gate_pass": get_nested(usage, ["decision", "final_usage_temporal_gate_pass"]),
            "remote_public_058_evidence": remote_decision.get("public_058_evidence_present"),
            "remote_beats_id178_evidence": remote_decision.get("beats_id178_public_anchor_evidence_present"),
        },
        "wall_decision": {
            "local_cv_only_candidate_path_closed": wall_decision.get("local_cv_only_candidate_path_closed"),
            "same_family_micro_probe_path_closed": wall_decision.get("same_family_micro_probe_path_closed"),
            "q3s3_independent_signal_required": wall_decision.get("q3s3_independent_signal_required"),
        },
    }

    q3_closed = evidence["closed_lanes"]["all_local_q3s3_lanes_closed"] is True
    clean_closed = evidence["remote_v3"]["clean_backbone_closed"] is True
    raw_temporal_is_next = bool(q3_closed and clean_closed)

    decision = {
        "post_id342_research_gap_ready": True,
        "current_public_anchor_id": 342,
        "target_public_score": TARGET_PUBLIC,
        "current_gap_to_target": id342_gap_to_target,
        "dacon_submit_now": False,
        "create_submission_artifact_now": False,
        "modify_submission_log_now": False,
        "local_material_micro_overlay_primary": False,
        "clean_id221_id226_replay_primary": False,
        "raw_temporal_q3s3_sequence_image_primary": raw_temporal_is_next,
        "kaggle_no_submit_recommended": raw_temporal_is_next,
        "kaggle_push_now_from_this_report": False,
        "requires_separate_runner_or_smoke_gate": True,
        "next_action": (
            "Implement and run a report-only ID342-aware raw temporal Q3/S3 sequence-image diagnostic. "
            "If local gates pass, push it through Kaggle no-submit. Do not create or submit a DACON artifact "
            "from the existing material feature overlays or clean ID221/ID226 replay."
        ),
    }

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_research_gap_and_source_priority",
        "purpose": "Convert corrected ID342 public feedback into an overnight research queue for Public 0.55 pursuit.",
        "inputs": inputs,
        "evidence": evidence,
        "source_priority_rows": source_priority_rows,
        "decision": decision,
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    ev = report["evidence"]
    anchor = ev["anchor"]
    bottleneck = ev["target_bottleneck"]
    closed = ev["closed_lanes"]
    remote = ev["remote_v3"]
    decision = report["decision"]
    rows = report["source_priority_rows"]
    lines = [
        "# Post-ID342 research gap and source priority",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Anchor read",
        "",
        f"- Current public anchor: ID{anchor['id']} Public `{anchor['public_score']:.10f}`, local `{anchor['local_cv']:.6f}`",
        f"- Delta vs ID178 Public: `{anchor['delta_vs_id178_public']:+.10f}`",
        f"- Delta vs ID337 Public: `{anchor['delta_vs_id337_public']:+.10f}`",
        f"- Gap to Public 0.55 target: `{anchor['gap_to_public_055']:+.10f}`",
        "",
        "## Target bottleneck",
        "",
        f"- ID342 Q3/S3/S4 local target losses: `{bottleneck['q3_score']}`, `{bottleneck['s3_score']}`, `{bottleneck['s4_score']}`",
        f"- Q3 lag-1 agreement: `{bottleneck['q3_lag1_agreement_rate']}`; top same-day correlation: `{bottleneck['q3_top_same_day_correlation']}`",
        f"- S3 lag-1 agreement: `{bottleneck['s3_lag1_agreement_rate']}`; top same-day correlation: `{bottleneck['s3_top_same_day_correlation']}`",
        "",
        "## Closed evidence",
        "",
        f"- Local Q3/S3 lanes closed: `{closed['all_local_q3s3_lanes_closed']}`; closed lanes: `{closed['closed_local_lane_count']}`",
        f"- Clean backbone closed: `{remote['clean_backbone_closed']}`; next lane: `{remote['next_lane']}`",
        f"- Phase-2 block A selected targets: `{remote['block_a_selected_targets']}`; final gate pass: `{remote['block_a_final_gate_pass']}`",
        f"- Phase-2 block B pass: `{remote['block_b_final_gate_pass']}`; usage temporal pass: `{remote['usage_temporal_gate_pass']}`",
        "",
        "## Source priority",
        "",
        "| rank | route | status | focus | Kaggle fit | submit allowed |",
        "|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {rank} | `{route}` | `{status}` | `{focus}` | `{fit}` | `{submit}` |".format(
                rank=row["rank"],
                route=row["route"],
                status=row["status"],
                focus=",".join(row["target_focus"]),
                fit=row["kaggle_fit"],
                submit=row["dacon_submit_allowed"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Raw temporal Q3/S3 sequence-image primary: `{decision['raw_temporal_q3s3_sequence_image_primary']}`",
            f"- Kaggle no-submit recommended: `{decision['kaggle_no_submit_recommended']}`",
            f"- Create submission artifact now: `{decision['create_submission_artifact_now']}`",
            f"- DACON submit now: `{decision['dacon_submit_now']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    print(f"Wrote {rel(args.output)} and {rel(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
