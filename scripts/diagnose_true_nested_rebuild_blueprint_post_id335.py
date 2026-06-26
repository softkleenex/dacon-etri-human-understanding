#!/usr/bin/env python3
"""Blueprint a true nested rebuild after the post-ID335 fold-internal diagnostic.

This script creates a report only. It inspects the fixed artifacts selected by
the fold-internal failed-direction diagnostic and explains what would have to be
rebuilt inside outer folds before this branch can become submission evidence.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
SUBMISSION_LOG = ROOT / "submissions" / "submission_log.csv"
KST = timezone(timedelta(hours=9))

FOLD_INTERNAL_REPORT = REPORTS_DIR / "fold_internal_failed_direction_rebuild_post_id335_20260617.json"
DEFAULT_SOURCE_IDS = [242, 243, 249, 252, 268, 276, 278]
PUBLIC_ANCHORS = {
    178: {"local": 0.538638, "public": 0.6014113009, "role": "best confirmed recent public anchor"},
    287: {"local": 0.378934, "public": 0.6698787318, "role": "catastrophic local-frontier failure"},
    329: {"local": 0.516718, "public": 0.6017908577, "role": "guarded combo plateau / no public gain"},
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_json_path(experiment_id: int) -> Path | None:
    matches = sorted(REPORTS_DIR.glob(f"experiment_{experiment_id}_*.json"))
    return matches[0] if matches else None


def safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_submission_log() -> pd.DataFrame:
    if not SUBMISSION_LOG.exists():
        return pd.DataFrame()
    return pd.read_csv(SUBMISSION_LOG)


def submission_log_row(submission_log: pd.DataFrame, experiment_id: int) -> dict[str, Any]:
    if submission_log.empty or "id" not in submission_log.columns:
        return {}
    rows = submission_log[submission_log["id"].astype(str) == str(experiment_id)]
    if rows.empty:
        return {}
    row = rows.iloc[0].to_dict()
    return {
        "submission_file": row.get("submission_file"),
        "model_description": row.get("model_description"),
        "local_cv_score": safe_float(row.get("local_cv_score")),
        "public_lb_score": safe_float(row.get("public_lb_score")),
        "status_clean": row.get("status_clean"),
        "risk_reason": row.get("risk_reason"),
        "safe_for_default_ensemble": bool(row.get("safe_for_default_ensemble"))
        if "safe_for_default_ensemble" in row
        else None,
    }


def ids_from_text(value: Any) -> set[int]:
    if value is None:
        return set()
    text = str(value)
    ids = {int(match) for match in re.findall(r"(?:^|[^A-Za-z0-9])id[_-]?(\d+)(?:[^0-9]|$)", text, flags=re.I)}
    ids.update(int(match) for match in re.findall(r"base_oof_id_(\d+)_", text))
    ids.update(int(match) for match in re.findall(r"base_test_id_(\d+)_", text))
    return ids


def structured_upstream_ids(metadata: dict[str, Any], experiment_id: int) -> list[int]:
    selector = metadata.get("selector_info", {})
    found: set[int] = set()

    for key in ["base_id", "artifact_ids"]:
        value = selector.get(key)
        if isinstance(value, int):
            found.add(value)
        elif isinstance(value, list):
            found.update(int(item) for item in value if isinstance(item, int))

    for key in ["base_source", "scan_path", "diagnostic_report", "selection_contract", "validation"]:
        found.update(ids_from_text(selector.get(key)))
        found.update(ids_from_text(metadata.get(key)))

    selected_configs = selector.get("selected_configs")
    if isinstance(selected_configs, dict):
        for config in selected_configs.values():
            if not isinstance(config, dict):
                continue
            for key in ["id_a", "id_b", "source_id", "candidate_id"]:
                value = config.get(key)
                if isinstance(value, int):
                    found.add(value)

    found.discard(experiment_id)
    return sorted(found)


def family_from_metadata(metadata: dict[str, Any]) -> str:
    local_cv_name = str(metadata.get("local_cv_name", "")).lower()
    method = str(metadata.get("method", "")).lower()
    if "second_stage_full_oof" in local_cv_name or "second-stage" in method:
        return "second_stage_full_oof_calibration"
    if "pair_prob_logit_blend" in local_cv_name or "pair" in method:
        return "targetwise_pair_prob_logit_blend"
    if "delta_capped_bridge" in local_cv_name or "bridge" in method:
        return "targetwise_delta_capped_bridge"
    return "unknown_or_mixed"


def risk_class(metadata: dict[str, Any]) -> str:
    family = family_from_metadata(metadata)
    local_cv = float(metadata.get("local_cv", 1.0))
    strict = bool(metadata.get("strict_subject_hole_comparable", False))
    nested = bool(metadata.get("fold_nested_calibration", False))
    if nested and strict:
        return "reviewable"
    if family == "second_stage_full_oof_calibration" or local_cv < 0.38:
        return "extreme_non_nested"
    if family == "targetwise_pair_prob_logit_blend":
        return "high_non_nested"
    return "non_nested_research"


def fold_internal_summary(report: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for split_name, split_rows in report.get("splits", {}).items():
        for row in split_rows:
            selected_ids = sorted(
                {
                    int(source_id)
                    for target_counts in row.get("choice_counts", {}).values()
                    for source_id in target_counts
                    if int(source_id) not in {178}
                }
            )
            rows.append(
                {
                    "split": split_name,
                    "setting": row.get("setting", {}).get("name"),
                    "overall": row.get("overall"),
                    "robust_score": row.get("robust_score"),
                    "selected_source_ids": selected_ids,
                }
            )
    best_by_split: dict[str, dict[str, Any]] = {}
    for row in rows:
        split = str(row["split"])
        if split not in best_by_split or float(row["overall"]) < float(best_by_split[split]["overall"]):
            best_by_split[split] = row
    return {"rows": rows, "best_by_split": best_by_split}


def source_record(experiment_id: int, submission_log: pd.DataFrame) -> dict[str, Any]:
    path = experiment_json_path(experiment_id)
    metadata = read_json(path) if path else {}
    log_row = submission_log_row(submission_log, experiment_id)
    upstream = structured_upstream_ids(metadata, experiment_id)
    return {
        "id": experiment_id,
        "experiment_json": str(path.relative_to(ROOT)) if path else None,
        "submission_file": metadata.get("submission_file") or log_row.get("submission_file"),
        "base_oof_file": metadata.get("base_oof_file"),
        "base_test_file": metadata.get("base_test_file"),
        "notebook_file": metadata.get("notebook_file"),
        "local_cv": metadata.get("local_cv") if metadata else log_row.get("local_cv_score"),
        "method": metadata.get("method") or log_row.get("model_description"),
        "family": family_from_metadata(metadata),
        "risk_class": risk_class(metadata) if metadata else "missing_metadata",
        "strict_subject_hole_comparable": bool(metadata.get("strict_subject_hole_comparable", False)),
        "fold_nested_calibration": bool(metadata.get("fold_nested_calibration", False)),
        "status_clean": log_row.get("status_clean"),
        "risk_reason": log_row.get("risk_reason"),
        "upstream_ids": upstream,
        "validation": metadata.get("validation"),
        "selection_contract": metadata.get("selector_info", {}).get("selection_contract") if metadata else None,
        "caution": metadata.get("selector_info", {}).get("caution") if metadata else None,
    }


def build_dependency_edges(records: list[dict[str, Any]]) -> list[dict[str, int]]:
    ids = {int(record["id"]) for record in records}
    edges = []
    for record in records:
        for upstream_id in record["upstream_ids"]:
            edges.append(
                {
                    "from_id": int(upstream_id),
                    "to_id": int(record["id"]),
                    "upstream_is_in_selected_source_set": int(upstream_id) in ids,
                }
            )
    return sorted(edges, key=lambda row: (row["to_id"], row["from_id"]))


def build_rebuild_plan(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    second_stage_ids = [record["id"] for record in records if record["family"] == "second_stage_full_oof_calibration"]
    pair_ids = [record["id"] for record in records if record["family"] == "targetwise_pair_prob_logit_blend"]
    bridge_ids = [record["id"] for record in records if record["family"] == "targetwise_delta_capped_bridge"]
    return [
        {
            "lane": "A_delta_capped_bridge_selection_rebuild",
            "purpose": "Check whether the target-wise capped bridge survives when bridge choices are learned inside outer folds.",
            "source_ids": bridge_ids,
            "outer_splits": ["subject_hole", "loso"],
            "allowed_operations": [
                "choose per-target bridge families and caps inside the outer-training rows only",
                "score only on untouched outer-validation rows",
                "reject any cap that relies on full-OOF target loss or hidden-test movement alone",
            ],
            "blocked_operations": [
                "reusing ID249's full-OOF target-wise selection as proof",
                "treating a capped test delta as a substitute for nested validation",
            ],
        },
        {
            "lane": "B_pair_blend_without_new_calibration",
            "purpose": "Check whether the pair/logit blend signal survives when target choices are learned inside outer folds.",
            "source_ids": pair_ids,
            "outer_splits": ["subject_hole", "loso"],
            "allowed_operations": [
                "choose target-wise single/pair/logit blend configs inside the outer-training rows only",
                "score only on untouched outer-validation rows",
                "carry full-train test predictions only after the nested OOF score is reviewed",
            ],
            "blocked_operations": [
                "using full OOF target loss to choose a target config",
                "promoting a submission CSV directly from existing ID268/276/278 artifacts",
            ],
        },
        {
            "lane": "C_second_stage_calibration_rebuild",
            "purpose": "Test whether the extreme calibration signal is real or only a full-OOF artifact.",
            "source_ids": second_stage_ids,
            "outer_splits": ["subject_hole", "loso"],
            "allowed_operations": [
                "fit isotonic/quantile calibrators only on inner-training predictions",
                "choose calibration family and quantile count inside inner folds",
                "penalize extreme probabilities and per-target instability before scoring",
            ],
            "blocked_operations": [
                "fitting isotonic or quantile maps on the same OOF rows used for evaluation",
                "treating local CV below 0.40 as leaderboard evidence without anchor explanation",
            ],
        },
        {
            "lane": "D_anchor_explanation_gate",
            "purpose": "Reject any rebuild that cannot explain the known public outcomes.",
            "source_ids": [178, 287, 329],
            "required_checks": [
                "ID287-like local-frontier movement must be penalized",
                "ID329-like local gain must not be mistaken for public improvement",
                "candidate must improve robust nested score without large movement away from ID178",
            ],
        },
    ]


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# True nested rebuild blueprint — post ID335",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no CSV, notebook, submission-log row, base prediction, or DACON submission.",
        "",
        "## Why this exists",
        "",
        "The fold-internal failed-direction diagnostic preserved a meaningful local signal, but the selected source artifacts were still generated with full-OOF/non-nested procedures. This report identifies what must be rebuilt before that signal can become submission evidence.",
        "",
        "## Fold-internal signal to explain",
        "",
        "| split | best setting | overall | robust | selected source ids |",
        "|---|---|---:|---:|---|",
    ]
    for split_name, row in report["fold_internal_summary"]["best_by_split"].items():
        lines.append(
            f"| `{split_name}` | `{row['setting']}` | {row['overall']:.6f} | "
            f"{row['robust_score']:.6f} | `{row['selected_source_ids']}` |"
        )
    lines.extend(
        [
            "",
            "## Selected source artifact audit",
            "",
            "| id | local CV | family | risk | immediate upstream ids | method |",
            "|---:|---:|---|---|---|---|",
        ]
    )
    for record in report["source_records"]:
        local = record["local_cv"]
        local_text = f"{float(local):.6f}" if local is not None else ""
        method = str(record.get("method") or "").replace("|", "/")
        lines.append(
            f"| {record['id']} | {local_text} | `{record['family']}` | `{record['risk_class']}` | "
            f"`{record['upstream_ids']}` | {method} |"
        )
    lines.extend(
        [
            "",
            "## Dependency edges",
            "",
            "| upstream id | selected source id | upstream also selected? |",
            "|---:|---:|---|",
        ]
    )
    for edge in report["dependency_edges"]:
        lines.append(
            f"| {edge['from_id']} | {edge['to_id']} | {'yes' if edge['upstream_is_in_selected_source_set'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Required rebuild contract",
            "",
            "1. Freeze outer subject-hole and LOSO splits before looking at scores.",
            "2. For every outer fold, generate or select each candidate family using only the outer-training rows.",
            "3. Fit target calibrators, bridge weights, projection penalties, and source choices inside that outer-training partition only.",
            "4. Score only the untouched outer-validation rows, then aggregate target, subject, and date-slice robustness.",
            "5. Do not create a DACON-facing artifact unless the nested result also explains ID178, ID287, and ID329 public behavior.",
            "",
            "## Rebuild lanes",
            "",
        ]
    )
    for lane in report["rebuild_plan"]:
        lines.extend(
            [
                f"### {lane['lane']}",
                "",
                lane["purpose"],
                "",
                f"- Source IDs: `{lane['source_ids']}`",
            ]
        )
        if "allowed_operations" in lane:
            lines.append(f"- Allowed: `{lane['allowed_operations']}`")
        if "blocked_operations" in lane:
            lines.append(f"- Blocked: `{lane['blocked_operations']}`")
        if "required_checks" in lane:
            lines.append(f"- Required checks: `{lane['required_checks']}`")
        lines.append("")
    lines.extend(["## Decision", "", report["decision"]["reason"], ""])
    return "\n".join(lines)


def run(output_json: Path, source_ids: list[int]) -> dict[str, Any]:
    submission_log = load_submission_log()
    fold_report = read_json(FOLD_INTERNAL_REPORT)
    source_records = [source_record(source_id, submission_log) for source_id in source_ids]
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "true_nested_rebuild_blueprint_post_id335",
        "source_ids": source_ids,
        "public_anchors": PUBLIC_ANCHORS,
        "fold_internal_summary": fold_internal_summary(fold_report),
        "source_records": source_records,
        "dependency_edges": build_dependency_edges(source_records),
        "rebuild_plan": build_rebuild_plan(source_records),
        "decision": {
            "create_submission_artifact": False,
            "dacon_submission_made": False,
            "public_058_plausible": False,
            "next_action": "Implement a true model-level nested rebuild lane before any leaderboard-facing promotion.",
            "reason": (
                "The selected source IDs preserve signal under fold-internal replay, but every selected source is "
                "non-nested and most are high/extreme-risk full-OOF calibration or target-wise blend artifacts. "
                "They justify a nested rebuild blueprint, not a submission."
            ),
        },
    }
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md = output_json.with_suffix(".md")
    output_md.write_text(build_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_DIR / "true_nested_rebuild_blueprint_post_id335_20260617.json",
    )
    parser.add_argument("--source-ids", type=int, nargs="+", default=DEFAULT_SOURCE_IDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.output_json, args.source_ids)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(report["decision"]["reason"])


if __name__ == "__main__":
    main()
