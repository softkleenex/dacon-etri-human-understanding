#!/usr/bin/env python3
"""Report-only audit for failed-direction/orthogonal candidates after ID335."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_id179_vs_id335_sequence_probe_review import (  # noqa: E402
    TARGETS,
    align,
    load_artifact,
    load_train_test,
    local_scores,
    robust_score,
    slice_summary,
)

REPORTS_DIR = ROOT / "reports"
LOG_PATH = ROOT / "submissions" / "submission_log.csv"
KST = timezone(timedelta(hours=9))
REVIEW_IDS = [178, 179, 295, 296, 297, 304, 305, 306, 307, 310, 311, 329, 334, 335]


def load_log() -> pd.DataFrame:
    log = pd.read_csv(LOG_PATH)
    log["id_num"] = pd.to_numeric(log["id"], errors="coerce").astype("Int64")
    return log


def log_info(log: pd.DataFrame, candidate_id: int) -> dict[str, Any]:
    rows = log[log["id_num"].eq(candidate_id)]
    if rows.empty:
        return {}
    row = rows.iloc[-1].to_dict()
    for key in ["local_cv_score_num", "public_lb_score_num", "local_cv_score", "public_lb_score"]:
        value = pd.to_numeric(row.get(key), errors="coerce")
        row[key] = None if pd.isna(value) else float(value)
    return row


def load_metadata(candidate_id: int) -> dict[str, Any]:
    paths = sorted(REPORTS_DIR.glob(f"experiment_{candidate_id:03d}_*.json"))
    if not paths:
        return {}
    return json.loads(paths[-1].read_text())


def projection(vec: np.ndarray, direction: np.ndarray) -> float:
    denom = float(np.dot(direction, direction))
    return float(np.dot(vec, direction) / denom) if denom > 0 else 0.0


def geometry(test_pred: pd.DataFrame, anchors: dict[int, pd.DataFrame]) -> dict[str, float]:
    arr = test_pred[TARGETS].astype(float).to_numpy()
    anchor_arrays = {candidate_id: frame[TARGETS].astype(float).to_numpy() for candidate_id, frame in anchors.items()}
    arr178 = anchor_arrays[178]
    failed_direction = (anchor_arrays[287] - arr178).reshape(-1)
    neutral_329 = (anchor_arrays[329] - arr178).reshape(-1)
    current_335 = (anchor_arrays[335] - arr178).reshape(-1)
    vec178 = (arr - arr178).reshape(-1)
    out: dict[str, float] = {}
    for anchor_id, anchor_arr in anchor_arrays.items():
        out[f"mean_delta_vs_{anchor_id}"] = float(np.mean(np.abs(arr - anchor_arr)))
        out[f"max_delta_vs_{anchor_id}"] = float(np.max(np.abs(arr - anchor_arr)))
    out["projection_failed"] = projection(vec178, failed_direction)
    out["projection_329"] = projection(vec178, neutral_329)
    out["projection_335"] = projection(vec178, current_335)
    out["prediction_mean"] = float(np.mean(arr))
    out["prediction_std"] = float(np.std(arr))
    return out


def source_flags(log_row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(
        [
            str(log_row.get("notes", "")),
            str(metadata.get("validation", "")),
            str(metadata.get("local_cv_name", "")),
            str(metadata.get("method", "")),
        ]
    ).lower()
    return {
        "full_oof_or_non_nested": any(token in text for token in ["full-oof", "full_oof", "not_nested", "non-strict"]),
        "strict_subject_hole_comparable": bool(metadata.get("strict_subject_hole_comparable", False)),
        "fold_nested_calibration": bool(metadata.get("fold_nested_calibration", False)),
        "mentions_orthogonal": "orthogonal" in text,
        "mentions_antivector": "antivector" in text,
        "mentions_failed_direction": "failed-direction" in text or "failed_direction" in text,
    }


def family_for(log_row: dict[str, Any], metadata: dict[str, Any]) -> str:
    text = f"{log_row.get('submission_file', '')} {metadata.get('slug', '')} {metadata.get('method', '')}".lower()
    if "orthogonal" in text or "antivector" in text or "failed" in text or "post303" in text or "post307" in text:
        return "failed_direction_recovery_family"
    if "sequence" in text or "random_conv" in text:
        return "sequence_temporal_family"
    if "combo" in text or "budget" in text:
        return "combo_family"
    return "other_family"


def verdict_for(row: dict[str, Any], anchors: dict[int, dict[str, Any]]) -> dict[str, Any]:
    candidate_id = int(row["id"])
    flags = row["source_flags"]
    local_gain_vs_179 = anchors[179]["overall"] - row["overall"]
    local_gain_vs_335 = anchors[335]["overall"] - row["overall"]
    efficient_gain = local_gain_vs_179 / max(row["geometry"]["mean_delta_vs_179"], 1e-12)
    if candidate_id in {178, 179, 329, 334, 335}:
        return {
            "decision": "reference",
            "reason": "Reference row for comparison, not a new failed-direction candidate.",
            "local_gain_vs_179": local_gain_vs_179,
            "local_gain_vs_335": local_gain_vs_335,
            "gain_per_mean_delta_vs_179": efficient_gain,
        }
    if flags["full_oof_or_non_nested"] and not flags["fold_nested_calibration"]:
        if local_gain_vs_179 > 0.004:
            decision = "freeze_full_oof_efficiency_trap"
            reason = (
                "Large local gain comes from non-nested/full-OOF recombination. ID329 already showed this family can "
                "preserve public plateau despite much better local CV."
            )
        elif row["geometry"]["projection_failed"] < -0.01 and local_gain_vs_179 <= 0:
            decision = "diagnostic_antivector_only"
            reason = "Negative failed-vector projection is informative, but it does not improve local CV versus ID179."
        else:
            decision = "diagnostic_only_not_promoted"
            reason = "Low movement is interesting, but non-nested full-OOF selection is not enough after ID329."
    else:
        decision = "review_later"
        reason = "Not immediately rejected by metadata flags, but still needs fold-nested validation before promotion."
    return {
        "decision": decision,
        "reason": reason,
        "local_gain_vs_179": local_gain_vs_179,
        "local_gain_vs_335": local_gain_vs_335,
        "gain_per_mean_delta_vs_179": efficient_gain,
    }


def review_candidate(
    candidate_id: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    log: pd.DataFrame,
    anchor_tests: dict[int, pd.DataFrame],
) -> dict[str, Any]:
    log_row = log_info(log, candidate_id)
    metadata = load_metadata(candidate_id)
    oof = align(load_artifact("oof", candidate_id), train)
    test_pred = align(load_artifact("test", candidate_id), test)
    overall, target_scores = local_scores(train, oof)
    slices = slice_summary(train, oof)
    return {
        "id": candidate_id,
        "submission_file": log_row.get("submission_file", ""),
        "family": family_for(log_row, metadata),
        "overall": overall,
        "target_scores": target_scores,
        "robust_score": robust_score(overall, slices),
        "slice_summary": {
            "worst_subject": slices["worst_subject"],
            "worst_subject_loss": slices["worst_subject_loss"],
            "worst_date_bin": slices["worst_date_bin"],
            "worst_date_loss": slices["worst_date_loss"],
            "worst_target": slices["worst_target"],
            "worst_target_loss": slices["worst_target_loss"],
            "late_minus_early": slices["late_minus_early"],
        },
        "geometry": geometry(test_pred, anchor_tests),
        "source_flags": source_flags(log_row, metadata),
        "metadata_summary": {
            "local_cv_name": metadata.get("local_cv_name", ""),
            "validation": metadata.get("validation", ""),
            "strict_subject_hole_comparable": metadata.get("strict_subject_hole_comparable"),
            "fold_nested_calibration": metadata.get("fold_nested_calibration"),
        },
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Failed-direction candidate audit — post ID335",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. No submission CSV, notebook, base artifact, submission-log row, or DACON submission was created.",
        "",
        "## Candidate decisions",
        "",
        "| id | decision | overall | robust | gain vs ID179 | mean Δ179 | max Δ178 | proj failed | gain/Δ179 | reason |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["candidate_rows"]:
        if row["id"] in [178, 179, 329, 334, 335]:
            continue
        verdict = row["verdict"]
        geom = row["geometry"]
        lines.append(
            f"| {row['id']} | `{verdict['decision']}` | {row['overall']:.6f} | {row['robust_score']:.6f} | "
            f"{verdict['local_gain_vs_179']:+.6f} | {geom['mean_delta_vs_179']:.6f} | "
            f"{geom['max_delta_vs_178']:.6f} | {geom['projection_failed']:+.4f} | "
            f"{verdict['gain_per_mean_delta_vs_179']:+.3f} | {verdict['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["decision"]["reason"],
            "",
        ]
    )
    return "\n".join(lines)


def run(output_json: Path) -> dict[str, Any]:
    train, test = load_train_test()
    log = load_log()
    anchor_ids = [178, 179, 287, 329, 335]
    anchor_tests = {candidate_id: align(load_artifact("test", candidate_id), test) for candidate_id in anchor_ids}
    rows = [review_candidate(candidate_id, train, test, log, anchor_tests) for candidate_id in REVIEW_IDS]
    by_id = {int(row["id"]): row for row in rows}
    for row in rows:
        row["verdict"] = verdict_for(row, by_id)
    promoted = [
        row
        for row in rows
        if row["verdict"]["decision"]
        not in {
            "reference",
            "freeze_full_oof_efficiency_trap",
            "diagnostic_only_not_promoted",
            "diagnostic_antivector_only",
        }
    ]
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": "failed_direction_candidates_post_id335",
        "review_ids": REVIEW_IDS,
        "anchor_ids": anchor_ids,
        "candidate_rows": rows,
        "decision": {
            "promote_generated_unsubmitted_artifact": False,
            "public_058_plausible": False,
            "promoted_rows": [row["id"] for row in promoted],
            "reason": (
                "The top failed-direction/orthogonal rows look attractive because they buy local CV with tiny test movement, "
                "but their gain is non-nested/full-OOF recombination. ID329 already demonstrated that this evidence does not "
                "convert into public improvement. Keep these rows diagnostic-only unless a fold-nested validation path is built."
            ),
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_json.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_DIR / "failed_direction_candidates_post_id335_20260617.json",
    )
    args = parser.parse_args()
    report = run(args.output_json)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
