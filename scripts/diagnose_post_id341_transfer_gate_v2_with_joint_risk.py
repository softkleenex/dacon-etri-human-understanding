#!/usr/bin/env python3
"""Post-ID341 transfer gate v2 with joint-risk evidence.

Report-only diagnostic. This combines the public-transfer proxy gate with the
joint-distribution warning lens so that a candidate must be acceptable on both
axes before it can be considered a Public 0.58 review candidate.

It creates no submission CSV, no base predictions, no notebook/history artifact,
no submission-log row, no Kaggle run, and no DACON submission.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts import diagnose_joint_risk_candidate_rerank_post_id341 as joint_mod
from scripts import diagnose_post_id341_public_transfer_audit as transfer_mod

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
PUBLIC_058_PROXY_GATE = transfer_mod.PUBLIC_058_GATE
RECENT_CONTROL_IDS = [178, 287, 329, 334, 335, 336, 337, 338, 339, 340, 341]
JOINT_ACCEPTABLE_TIERS = {"low_joint_warning", "watch_one_warning"}


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def sort_value(value: Any, default: float = 999.0) -> float:
    out = safe_float(value)
    return default if out is None else out


def load_transfer_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recompute all transfer rows, not only the summarized top-80 JSON rows."""
    log = transfer_mod.load_log()
    public_rows = transfer_mod.clean_real_public_rows(log)
    recent_rows = public_rows[public_rows["id_num"].astype(int).isin(transfer_mod.PUBLIC_ANCHOR_IDS)]
    if recent_rows.empty:
        raise ValueError("Missing recent public anchors 178/329 in submission log.")

    public_base = float(recent_rows["public_lb_score_num"].min())
    recent_gap_floor = float(recent_rows["gap"].min())
    local_public_corr = float(public_rows[["local_cv_score_num", "public_lb_score_num"]].corr().iloc[0, 1])

    files = transfer_mod.find_base_test_files()
    required = [178, 287, 329, 341]
    missing = [candidate_id for candidate_id in required if candidate_id not in files]
    if missing:
        raise FileNotFoundError(f"Missing required base_test ids: {missing}")

    arr178 = transfer_mod.load_pred(files[178])
    arr287 = transfer_mod.load_pred(files[287])
    arr329 = transfer_mod.load_pred(files[329])
    arr341 = transfer_mod.load_pred(files[341])
    id341_proxy = transfer_mod.read_id341_proxy()

    rows = [
        transfer_mod.row_for_candidate(
            candidate_id,
            path,
            log,
            arr178,
            arr287,
            arr329,
            arr341,
            public_base,
            recent_gap_floor,
            id341_proxy,
        )
        for candidate_id, path in sorted(files.items())
    ]
    meta = {
        "rows": len(rows),
        "public_rows_used": int(len(public_rows)),
        "recent_anchor_gap_floor": recent_gap_floor,
        "local_public_corr": local_public_corr,
        "public_base": public_base,
        "id341_proxy": id341_proxy,
    }
    return rows, meta


def load_joint_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recompute all joint-risk rows from existing submissions."""
    public_risk = joint_mod.load_json(joint_mod.PUBLIC_JOINT_RISK_PATH)
    log, train, test = joint_mod.load_candidate_rows()
    rows, skipped = joint_mod.build_metrics_for_log(log, train, test)
    anchor_rows = [row for row in rows if row["id"] in [*joint_mod.SAFE_ANCHOR_IDS, joint_mod.COLLAPSE_ANCHOR_ID]]
    collapse_row = next((row for row in rows if row["id"] == joint_mod.COLLAPSE_ANCHOR_ID), None)
    thresholds = joint_mod.compute_thresholds(anchor_rows, collapse_row)
    decorated = joint_mod.decorate_rows(rows, thresholds)
    meta = {
        "rows": len(decorated),
        "unscored_rows": sum(1 for row in decorated if row.get("public_lb") is None),
        "skipped_rows": len(skipped),
        "strong_signal_metric_names": public_risk.get("decision", {}).get("strong_signal_metric_names", []),
        "thresholds": thresholds,
        "skipped": skipped[:80],
    }
    return decorated, meta


def transfer_gate_pass(row: dict[str, Any]) -> bool:
    return bool(
        row.get("public_lb") is None
        and row.get("artifact_exists")
        and row.get("risk_tier") == "would_pass_numeric_gate_review_needed"
        and row.get("passes_public_058_gate")
        and sort_value(row.get("proxy_public_risk")) <= PUBLIC_058_PROXY_GATE
    )


def joint_gate_pass(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    return bool(
        row.get("public_lb") is None
        and row.get("joint_risk_tier") in JOINT_ACCEPTABLE_TIERS
        and int(row.get("joint_warning_count") or 0) <= 1
    )


def block_reasons(
    transfer_row: dict[str, Any],
    joint_row: dict[str, Any] | None,
    transfer_ok: bool,
    joint_ok: bool,
) -> list[str]:
    reasons: list[str] = []
    if transfer_row.get("public_lb") is not None:
        reasons.append("already_scored_reference")
    if not transfer_ok:
        proxy = safe_float(transfer_row.get("proxy_public_risk"))
        if proxy is None or proxy > PUBLIC_058_PROXY_GATE:
            reasons.append("transfer_proxy_above_public_058_gate")
        if transfer_row.get("risk_tier") != "would_pass_numeric_gate_review_needed":
            reasons.append(f"transfer_tier:{transfer_row.get('risk_tier')}")
        if not transfer_row.get("artifact_exists"):
            reasons.append("missing_artifact")
    if joint_row is None:
        reasons.append("missing_joint_metrics")
    elif not joint_ok:
        reasons.append(f"joint_tier:{joint_row.get('joint_risk_tier')}")
        if int(joint_row.get("joint_warning_count") or 0) > 1:
            reasons.append("joint_warning_count_gt_1")
    return list(dict.fromkeys(reasons))


def combined_tier(transfer_ok: bool, joint_ok: bool, reasons: list[str]) -> str:
    if transfer_ok and joint_ok:
        return "review_public_058_candidate"
    if joint_ok and "transfer_proxy_above_public_058_gate" in reasons:
        return "joint_safe_but_transfer_blocked"
    if transfer_ok and not joint_ok:
        return "transfer_pass_but_joint_blocked"
    if any(reason.startswith("joint_tier:reject") for reason in reasons):
        return "reject_joint_risk"
    if "transfer_proxy_above_public_058_gate" in reasons:
        return "reject_transfer_proxy"
    return "reject_other_gate"


def short_joint(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            "joint_risk_tier": "missing_joint_metrics",
            "joint_warning_count": None,
            "joint_warning_flags": [],
            "target_mean_l1_to_train": None,
            "pair11_l1_to_train": None,
            "extreme_rate_05_95": None,
            "prediction_entropy": None,
            "prediction_sharpness": None,
            "temporal_volatility_ratio_to_train": None,
            "mean_abs_delta_vs_id178_joint": None,
            "max_abs_delta_vs_id178_joint": None,
        }
    return {
        "joint_risk_tier": row.get("joint_risk_tier"),
        "joint_warning_count": row.get("joint_warning_count"),
        "joint_warning_flags": row.get("joint_warning_flags") or [],
        "target_mean_l1_to_train": row.get("target_mean_l1_to_train"),
        "pair11_l1_to_train": row.get("pair11_l1_to_train"),
        "extreme_rate_05_95": row.get("extreme_rate_05_95"),
        "prediction_entropy": row.get("prediction_entropy"),
        "prediction_sharpness": row.get("prediction_sharpness"),
        "temporal_volatility_ratio_to_train": row.get("temporal_volatility_ratio_to_train"),
        "mean_abs_delta_vs_id178_joint": row.get("mean_abs_delta_vs_id178"),
        "max_abs_delta_vs_id178_joint": row.get("max_abs_delta_vs_id178"),
    }


def combine_rows(
    transfer_rows: list[dict[str, Any]], joint_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    joint_by_id = {int(row["id"]): row for row in joint_rows}
    transfer_ids = {int(row["id"]) for row in transfer_rows}
    combined = []
    for transfer_row in transfer_rows:
        candidate_id = int(transfer_row["id"])
        joint_row = joint_by_id.get(candidate_id)
        transfer_ok = transfer_gate_pass(transfer_row)
        joint_ok = joint_gate_pass(joint_row)
        reasons = block_reasons(transfer_row, joint_row, transfer_ok, joint_ok)
        combined.append(
            {
                "id": candidate_id,
                "submission_file": transfer_row.get("submission_file"),
                "base_test_file": transfer_row.get("base_test_file"),
                "model_description": transfer_row.get("model_description"),
                "local_cv": transfer_row.get("local_cv"),
                "public_lb": transfer_row.get("public_lb"),
                "status_clean": transfer_row.get("status_clean"),
                "artifact_exists": transfer_row.get("artifact_exists"),
                "family": transfer_row.get("family"),
                "proxy_public_risk": transfer_row.get("proxy_public_risk"),
                "passes_public_058_gate": transfer_row.get("passes_public_058_gate"),
                "transfer_risk_tier": transfer_row.get("risk_tier"),
                "mean_delta_vs_178": transfer_row.get("mean_delta_vs_178"),
                "max_delta_vs_178": transfer_row.get("max_delta_vs_178"),
                "mean_delta_vs_341": transfer_row.get("mean_delta_vs_341"),
                "max_delta_vs_341": transfer_row.get("max_delta_vs_341"),
                "projection_failed": transfer_row.get("projection_failed"),
                "projection_329": transfer_row.get("projection_329"),
                "projection_341": transfer_row.get("projection_341"),
                **short_joint(joint_row),
                "transfer_gate_pass": transfer_ok,
                "joint_gate_pass": joint_ok,
                "combined_gate_pass": bool(transfer_ok and joint_ok),
                "combined_tier": combined_tier(transfer_ok, joint_ok, reasons),
                "block_reasons": reasons,
            }
        )

    joint_only_reviewable = [
        row
        for row in joint_rows
        if int(row["id"]) not in transfer_ids
        and row.get("public_lb") is None
        and row.get("joint_risk_tier") in JOINT_ACCEPTABLE_TIERS
    ]
    return combined, joint_only_reviewable


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "id",
        "submission_file",
        "local_cv",
        "public_lb",
        "family",
        "proxy_public_risk",
        "transfer_risk_tier",
        "joint_risk_tier",
        "joint_warning_count",
        "joint_warning_flags",
        "mean_delta_vs_178",
        "max_delta_vs_178",
        "max_delta_vs_341",
        "projection_failed",
        "extreme_rate_05_95",
        "pair11_l1_to_train",
        "prediction_entropy",
        "transfer_gate_pass",
        "joint_gate_pass",
        "combined_gate_pass",
        "combined_tier",
        "block_reasons",
    ]
    return {key: row.get(key) for key in keep}


def compact_joint_only(row: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "id",
        "submission_file",
        "local_cv",
        "public_lb",
        "status_clean",
        "joint_risk_tier",
        "joint_warning_count",
        "joint_warning_flags",
        "mean_abs_delta_vs_id178",
        "max_abs_delta_vs_id178",
        "extreme_rate_05_95",
        "pair11_l1_to_train",
        "prediction_entropy",
    ]
    return {key: row.get(key) for key in keep}


def build_report() -> dict[str, Any]:
    transfer_rows, transfer_meta = load_transfer_rows()
    joint_rows, joint_meta = load_joint_rows()
    combined, joint_only_reviewable = combine_rows(transfer_rows, joint_rows)

    unscored = [row for row in combined if row.get("public_lb") is None]
    pass_rows = [row for row in unscored if row["combined_gate_pass"]]
    joint_safe_transfer_blocked = [row for row in unscored if row["joint_gate_pass"] and not row["transfer_gate_pass"]]
    transfer_pass_joint_blocked = [row for row in unscored if row["transfer_gate_pass"] and not row["joint_gate_pass"]]
    local_frontier = sorted(unscored, key=lambda row: sort_value(row.get("local_cv")))[:30]
    top_transfer = sorted(
        unscored, key=lambda row: (sort_value(row.get("proxy_public_risk")), sort_value(row.get("max_delta_vs_341")))
    )[:30]
    recent_controls = [row for row in combined if row["id"] in RECENT_CONTROL_IDS]
    combined_counts = dict(Counter(row["combined_tier"] for row in combined))
    unscored_counts = dict(Counter(row["combined_tier"] for row in unscored))

    any_pass = bool(pass_rows)
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_transfer_gate_v2_with_joint_risk",
        "purpose": (
            "Combine the public-transfer proxy gate and the post-ID341 joint-distribution warning lens. "
            "A candidate must pass both gates before it can be called a Public 0.58 review candidate."
        ),
        "input_reports": {
            "public_transfer_audit": "reports/post_id341_public_transfer_audit_20260618.json",
            "joint_risk_rerank": "reports/joint_risk_candidate_rerank_post_id341_20260618.json",
        },
        "gate_definition": {
            "public_058_proxy_gate": PUBLIC_058_PROXY_GATE,
            "transfer_gate_pass": (
                "unscored candidate with artifact_exists=true, transfer_tier "
                "`would_pass_numeric_gate_review_needed`, and proxy_public_risk <= public_058_proxy_gate"
            ),
            "joint_gate_pass": "unscored candidate with joint tier in low/watch and warning_count <= 1",
            "combined_gate_pass": "transfer_gate_pass and joint_gate_pass",
        },
        "coverage": {
            "transfer_rows": transfer_meta["rows"],
            "joint_rows": joint_meta["rows"],
            "combined_rows": len(combined),
            "unscored_combined_rows": len(unscored),
            "joint_only_reviewable_without_base_test": len(joint_only_reviewable),
            "joint_skipped_rows": joint_meta["skipped_rows"],
        },
        "transfer_meta": transfer_meta,
        "joint_meta": {key: value for key, value in joint_meta.items() if key not in {"thresholds", "skipped"}},
        "combined_tier_counts": combined_counts,
        "unscored_combined_tier_counts": unscored_counts,
        "recent_control_rows": [compact_row(row) for row in sorted(recent_controls, key=lambda item: item["id"])],
        "combined_pass_rows": [
            compact_row(row) for row in sorted(pass_rows, key=lambda item: sort_value(item.get("proxy_public_risk")))
        ],
        "top_transfer_proxy_rows": [compact_row(row) for row in top_transfer],
        "joint_safe_but_transfer_blocked": [
            compact_row(row)
            for row in sorted(
                joint_safe_transfer_blocked,
                key=lambda item: (sort_value(item.get("proxy_public_risk")), sort_value(item.get("local_cv"))),
            )[:30]
        ],
        "transfer_pass_but_joint_blocked": [
            compact_row(row)
            for row in sorted(
                transfer_pass_joint_blocked,
                key=lambda item: (sort_value(item.get("proxy_public_risk")), sort_value(item.get("local_cv"))),
            )[:30]
        ],
        "local_cv_frontier_combined": [compact_row(row) for row in local_frontier],
        "joint_only_reviewable_without_base_test": [
            compact_joint_only(row)
            for row in sorted(
                joint_only_reviewable,
                key=lambda item: (
                    int(item.get("joint_warning_count") or 0),
                    sort_value(item.get("mean_abs_delta_vs_id178")),
                    sort_value(item.get("local_cv")),
                ),
            )[:30]
        ],
        "decision": {
            "any_candidate_passes_combined_public_058_gate": any_pass,
            "public_058_plausible_candidate_exists": any_pass,
            "best_combined_pass_id": pass_rows[0]["id"] if pass_rows else None,
            "best_combined_pass_file": pass_rows[0]["submission_file"] if pass_rows else None,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "next_action": (
                "No local candidate should be submitted unless it passes this combined gate and then receives a "
                "separate artifact-contract/manual approval review. If no row passes, stop same-family local "
                "recombination and move to the explicit-approved Kaggle upstream rebuild."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def fmt(value: Any, digits: int = 6) -> str:
    value = safe_float(value)
    return "" if value is None else f"{value:.{digits}f}"


def bool_text(value: Any) -> str:
    return "Y" if bool(value) else "N"


def table_rows(rows: list[dict[str, Any]], limit: int = 30) -> list[str]:
    lines = []
    for row in rows[:limit]:
        reasons = ",".join(row.get("block_reasons") or [])
        flags = ",".join(row.get("joint_warning_flags") or [])
        lines.append(
            f"| {row['id']} | {fmt(row.get('local_cv'))} | {fmt(row.get('public_lb'))} | "
            f"{fmt(row.get('proxy_public_risk'))} | `{row.get('transfer_risk_tier')}` | "
            f"`{row.get('joint_risk_tier')}` | {row.get('joint_warning_count')} | "
            f"{bool_text(row.get('transfer_gate_pass'))}/{bool_text(row.get('joint_gate_pass'))}/"
            f"{bool_text(row.get('combined_gate_pass'))} | `{row.get('combined_tier')}` | "
            f"`{flags}` | `{reasons}` | `{row.get('submission_file')}` |"
        )
    return lines


def joint_only_table_rows(rows: list[dict[str, Any]], limit: int = 30) -> list[str]:
    lines = []
    for row in rows[:limit]:
        flags = ",".join(row.get("joint_warning_flags") or [])
        lines.append(
            f"| {row['id']} | {fmt(row.get('local_cv'))} | `{row.get('joint_risk_tier')}` | "
            f"{row.get('joint_warning_count')} | {fmt(row.get('mean_abs_delta_vs_id178'))} | "
            f"{fmt(row.get('extreme_rate_05_95'))} | `{flags}` | `{row.get('submission_file')}` |"
        )
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Post-ID341 transfer gate v2 with joint-risk evidence",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, no notebook/history artifact, no submission-log row, no Kaggle run, and no DACON submission.",
        "",
        "## Gate definition",
        "",
        f"- Conservative Public 0.58 proxy cutoff: `{report['gate_definition']['public_058_proxy_gate']:.6f}`",
        f"- Transfer gate: {report['gate_definition']['transfer_gate_pass']}",
        f"- Joint gate: {report['gate_definition']['joint_gate_pass']}",
        f"- Combined gate: `{report['gate_definition']['combined_gate_pass']}`",
        "",
        "## Coverage",
        "",
        f"- Transfer rows recomputed: `{report['coverage']['transfer_rows']}`",
        f"- Joint-risk rows recomputed: `{report['coverage']['joint_rows']}`",
        f"- Combined rows: `{report['coverage']['combined_rows']}`",
        f"- Unscored combined rows: `{report['coverage']['unscored_combined_rows']}`",
        f"- Joint-reviewable rows without base-test transfer artifact: `{report['coverage']['joint_only_reviewable_without_base_test']}`",
        "",
        "## Combined tier counts",
        "",
        "```json",
        json.dumps(report["unscored_combined_tier_counts"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Recent/control rows",
        "",
        "| id | local | public | proxy | transfer tier | joint tier | warn | T/J/C | combined tier | flags | block reasons | file |",
        "|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|",
        *table_rows(report["recent_control_rows"], limit=80),
        "",
        "## Combined pass rows",
        "",
        "| id | local | public | proxy | transfer tier | joint tier | warn | T/J/C | combined tier | flags | block reasons | file |",
        "|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|",
        *table_rows(report["combined_pass_rows"], limit=50),
        "",
        "## Best transfer-proxy rows under the combined gate",
        "",
        "| id | local | public | proxy | transfer tier | joint tier | warn | T/J/C | combined tier | flags | block reasons | file |",
        "|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|",
        *table_rows(report["top_transfer_proxy_rows"], limit=30),
        "",
        "## Joint-safe but transfer-blocked rows",
        "",
        "| id | local | public | proxy | transfer tier | joint tier | warn | T/J/C | combined tier | flags | block reasons | file |",
        "|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|",
        *table_rows(report["joint_safe_but_transfer_blocked"], limit=30),
        "",
        "## Transfer-pass but joint-blocked rows",
        "",
        "| id | local | public | proxy | transfer tier | joint tier | warn | T/J/C | combined tier | flags | block reasons | file |",
        "|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|",
        *table_rows(report["transfer_pass_but_joint_blocked"], limit=30),
        "",
        "## Local-CV frontier under combined gate",
        "",
        "| id | local | public | proxy | transfer tier | joint tier | warn | T/J/C | combined tier | flags | block reasons | file |",
        "|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|",
        *table_rows(report["local_cv_frontier_combined"], limit=30),
        "",
        "## Joint-reviewable rows missing base-test transfer artifact",
        "",
        "| id | local | joint tier | warn | Δ178 mean | ext05/95 | flags | file |",
        "|---:|---:|---|---:|---:|---:|---|---|",
        *joint_only_table_rows(report["joint_only_reviewable_without_base_test"], limit=30),
        "",
        "## Decision",
        "",
        f"- Any candidate passes combined Public 0.58 gate: `{report['decision']['any_candidate_passes_combined_public_058_gate']}`",
        f"- Public 0.58 plausible candidate exists: `{report['decision']['public_058_plausible_candidate_exists']}`",
        f"- Best combined-pass ID: `{report['decision']['best_combined_pass_id']}`",
        f"- Create submission artifact: `{report['decision']['create_submission_artifact']}`",
        f"- Submit now: `{report['decision']['submit_now']}`",
        f"- Push Kaggle now: `{report['decision']['push_kaggle_now']}`",
        f"- Next action: {report['decision']['next_action']}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_transfer_gate_v2_joint_risk_20260618.json",
    )
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
