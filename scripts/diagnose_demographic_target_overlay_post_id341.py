#!/usr/bin/env python3
"""Report-only demographic/static overlay scan on the ID341 control anchor.

The demographic branch was rejected on ID335, but ID341 is now the lowest-risk
control artifact. This diagnostic re-tests only the allowed target-specific
demographic hypotheses (Q2/S2/S4) as tiny overlays on ID341. It creates no
submission artifact and does not submit to DACON.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_demographic_target_overlay_post_id335 import (  # noqa: E402
    build_combined_predictions,
    candidate_lookup,
    demo_candidate_names,
    scan_target_candidate,
    select_target_rows,
    target_loss,
)
from scripts.diagnose_id179_vs_id335_sequence_probe_review import geometry, robust_score, slice_summary  # noqa: E402
from scripts.diagnose_id340_exact_sequence_full_geometry_gate import conservative_public_proxy  # noqa: E402
from scripts.diagnose_strict_demographic_feature_audit import (  # noqa: E402
    build_candidates,
    make_loso_folds,
    prepare_frames,
)
from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_strict_raw_feature_experiment import REPORTS_DIR, TARGETS  # noqa: E402
from src.utils.validation import make_subject_hole_folds  # noqa: E402

KST = timezone(timedelta(hours=9))
BASE_ID = 341
REFERENCE_PROXY_ID341 = 0.6038297835283394
DEFAULT_ALPHA_GRID = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]


def retarget_selected_rows(selected: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(selected))
    for row in out.values():
        row["baseline_id341_loss"] = row.pop("baseline_id335_loss")
        row["test_max_abs_delta_vs_id341"] = row.pop("test_max_abs_delta_vs_id335")
        row["test_mean_abs_delta_vs_id341"] = row.pop("test_mean_abs_delta_vs_id335")
    return out


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Demographic target overlay scan — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. No submission CSV, notebook, base artifact, submission-log row, or DACON submission was created.",
        "",
        "## Selected target decisions",
        "",
        "| target | decision | candidate | alpha | ID341 loss | SH blend | LOSO blend | max Δtest341 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for target, row in report["selected_target_rows"].items():
        lines.append(
            f"| {target} | `{row['decision']}` | `{row['candidate']}` | {row['alpha']:.4f} | "
            f"{row['baseline_id341_loss']:.6f} | {row['subject_hole_blend_loss']:.6f} | "
            f"{row['loso_blend_loss']:.6f} | {row['test_max_abs_delta_vs_id341']:.6f} |"
        )
    geom = report["combined"]["geometry"]
    lines.extend(
        [
            "",
            "## Combined guarded overlay",
            "",
            f"- Promoted targets: `{report['combined']['promoted_targets']}`",
            f"- Combined local OOF CV: `{report['combined']['local_cv']:.6f}`",
            f"- ID341 local OOF CV: `{report['base_local_cv']:.6f}`",
            f"- Conservative public-risk proxy: `{report['combined']['conservative_public_risk_proxy']:.6f}`",
            f"- Proxy Δ vs ID341: `{report['combined']['proxy_delta_vs_id341']:+.6f}`",
            f"- Geometry max Δ178: `{geom['max_delta_vs_178']:.6f}`",
            f"- Failed-vector projection: `{geom['projection_failed']:+.6f}`",
            f"- Robust score: `{report['combined']['robust_score']:.6f}`",
            "",
            "## Verdict",
            "",
            report["decision"]["reason"],
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    output_json: Path,
    gate_json: Path,
    audit_json: Path,
    alpha_grid: list[float],
    n_estimators: int,
) -> dict[str, Any]:
    train, test = prepare_frames()
    gate = json.loads(gate_json.read_text(encoding="utf-8"))
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    allowed_targets = [target for target in gate["allow_research_targets"] if target in TARGETS]
    base_oof, base_test, base_source = load_artifact_pair(train, test, BASE_ID)
    _id178_oof, id178_test, id178_source = load_artifact_pair(train, test, 178)
    _id287_oof, id287_test, id287_source = load_artifact_pair(train, test, 287)
    _id329_oof, id329_test, id329_source = load_artifact_pair(train, test, 329)
    _id335_oof, id335_test, id335_source = load_artifact_pair(train, test, 335)

    candidates = build_candidates(train, test, seed=42, n_estimators=n_estimators)
    candidates_by_name = candidate_lookup(candidates)
    subject_hole_folds = make_subject_hole_folds(train, n_folds=5)
    loso_folds = make_loso_folds(train)
    scans = []
    for target in allowed_targets:
        for candidate_name in demo_candidate_names(audit, target):
            scans.append(
                scan_target_candidate(
                    train=train,
                    test=test,
                    target=target,
                    candidate=candidates_by_name[candidate_name],
                    subject_hole_folds=subject_hole_folds,
                    loso_folds=loso_folds,
                    id335_oof=base_oof,
                    id335_test=base_test,
                    alpha_grid=alpha_grid,
                )
            )

    selected_raw = select_target_rows(scans)
    selected = retarget_selected_rows(selected_raw)
    combined_oof, combined_test, target_losses = build_combined_predictions(
        train=train,
        test=test,
        selected=selected_raw,
        candidate_by_name=candidates_by_name,
        subject_hole_folds=subject_hole_folds,
        id335_oof=base_oof,
        id335_test=base_test,
    )
    base_target_losses = {
        target: target_loss(train[target].astype(int).to_numpy(), base_oof[target].astype(float).to_numpy())
        for target in TARGETS
    }
    local_cv = float(np.mean(list(target_losses.values())))
    base_local_cv = float(np.mean(list(base_target_losses.values())))
    promoted_targets = [
        target for target, row in selected_raw.items() if row["decision"] == "allow_tiny_overlay_hypothesis"
    ]
    anchors = {178: id178_test, 287: id287_test, 329: id329_test, 335: id335_test}
    geom = geometry(combined_test, anchors)
    slices = slice_summary(train, combined_oof)
    proxy = conservative_public_proxy(geom, local_cv)
    promote = bool(
        promoted_targets
        and local_cv < base_local_cv - 0.0001
        and geom["max_delta_vs_178"] <= 0.01
        and geom["projection_failed"] <= 0.002
        and proxy < REFERENCE_PROXY_ID341
    )
    reason = (
        "A tiny ID341-anchored demographic overlay is reviewable, but it is still not Public 0.58-plausible and would need a separate artifact contract before any submission."
        if promote
        else "No ID341-anchored demographic overlay is promoted. The target-specific demographic signal remains too weak or too risky relative to the ID341 control proxy."
    )
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "demographic_target_overlay_post_id341",
        "sources": {
            "gate": str(gate_json.relative_to(ROOT)),
            "audit": str(audit_json.relative_to(ROOT)),
            "base_id341": base_source,
            "id178": id178_source,
            "id287": id287_source,
            "id329": id329_source,
            "id335": id335_source,
        },
        "allowed_targets": allowed_targets,
        "alpha_grid": alpha_grid,
        "n_estimators": n_estimators,
        "scans": scans,
        "selected_target_rows": selected,
        "base_target_losses": base_target_losses,
        "base_local_cv": base_local_cv,
        "combined": {
            "promoted_targets": promoted_targets,
            "local_cv": local_cv,
            "target_losses": target_losses,
            "geometry": geom,
            "robust_score": robust_score(local_cv, slices),
            "conservative_public_risk_proxy": proxy,
            "proxy_delta_vs_id341": float(proxy - REFERENCE_PROXY_ID341),
        },
        "decision": {
            "promote_generated_unsubmitted_artifact": promote,
            "public_058_plausible": False,
            "reason": reason,
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
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
        default=REPORTS_DIR / "demographic_target_overlay_post_id341_20260618.json",
    )
    parser.add_argument(
        "--gate-json",
        type=Path,
        default=REPORTS_DIR / "target_specific_demographic_gate_post_id335_20260617.json",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=REPORTS_DIR / "strict_demographic_feature_audit_kaggle_20260617.json",
    )
    parser.add_argument("--alpha-grid", default="0,0.0025,0.005,0.01,0.02,0.03,0.05")
    parser.add_argument("--n-estimators", type=int, default=120)
    args = parser.parse_args()
    report = run(
        output_json=args.output_json,
        gate_json=args.gate_json,
        audit_json=args.audit_json,
        alpha_grid=[float(value) for value in args.alpha_grid.split(",")],
        n_estimators=args.n_estimators,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(json.dumps(report["decision"] | report["combined"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
