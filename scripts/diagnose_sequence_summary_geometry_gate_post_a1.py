#!/usr/bin/env python3
"""Geometry gate for post-A1 sequence-summary candidates.

This diagnostic builds candidate OOF/test predictions in memory only. It does
not write a submission CSV, notebook/history file, base prediction artifact, or
submission-log row.
"""

from __future__ import annotations

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
    geometry,
    load_artifact,
    load_train_test,
    local_scores,
    robust_score,
    slice_summary,
)
from scripts.run_public_sequence_summary_embedding_overlay_experiment import (  # noqa: E402
    predict_sequence_summary_embedding_overlay,
)

REPORTS_DIR = ROOT / "reports"
OUTPUT_PATH = REPORTS_DIR / "sequence_summary_geometry_gate_post_a1_20260618.json"
KST = timezone(timedelta(hours=9))

FULL_ID178_SEQUENCE_CONFIGS: dict[str, dict[str, Any]] = {
    "Q1": {"kind": "blend", "model": "lr_c0_3", "weight": 0.10},
    "Q2": {"kind": "blend", "model": "lr_c0_3", "weight": 0.15},
    "Q3": {"kind": "blend", "model": "lr_c0_1", "weight": 0.10},
    "S1": {"kind": "blend", "model": "lr_c0_3", "weight": 0.05},
    "S2": {"kind": "base"},
    "S3": {"kind": "base"},
    "S4": {"kind": "base"},
}


def evaluate_candidate(
    *,
    name: str,
    oof: pd.DataFrame,
    test_pred: pd.DataFrame,
    train: pd.DataFrame,
    anchors: dict[int, pd.DataFrame],
    modified_targets: list[str],
    read: str,
) -> dict[str, Any]:
    overall, target_scores = local_scores(train, oof)
    slices = slice_summary(train, oof)
    geom = geometry(test_pred, anchors)
    base178 = anchors[178]
    deltas = np.abs(test_pred[TARGETS].astype(float).to_numpy() - base178[TARGETS].astype(float).to_numpy())
    return {
        "name": name,
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
        "geometry": geom,
        "max_delta_vs_178_by_target": {target: float(deltas[:, idx].max()) for idx, target in enumerate(TARGETS)},
        "modified_targets": modified_targets,
        "read": read,
    }


def build_candidates() -> dict[str, Any]:
    train, test = load_train_test()
    anchor_tests = {
        candidate_id: align(load_artifact("test", candidate_id), test) for candidate_id in [178, 287, 329, 335]
    }
    base178_oof = align(load_artifact("oof", 178), train)
    base178_test = align(load_artifact("test", 178), test)
    id337_oof = align(load_artifact("oof", 337), train)
    id337_test = align(load_artifact("test", 337), test)

    full_oof, full_test, _scores, selector_info, _sample_original = predict_sequence_summary_embedding_overlay(
        178,
        selected_configs=FULL_ID178_SEQUENCE_CONFIGS,
        diagnostic_reports=["reports/id178_sequence_summary_embedding_post_a1_20260618.json"],
    )
    full_oof = align(full_oof, train)
    full_test = align(full_test, test)

    q123_oof = full_oof.copy()
    q123_test = full_test.copy()
    q123_oof["S1"] = base178_oof["S1"]
    q123_test["S1"] = base178_test["S1"]

    q123_plus_id337_oof = q123_oof.copy()
    q123_plus_id337_test = q123_test.copy()
    q123_plus_id337_oof["S1"] = id337_oof["S1"]
    q123_plus_id337_test["S1"] = id337_test["S1"]

    candidates = [
        evaluate_candidate(
            name="id178_q123_sequence_summary_gate",
            oof=q123_oof,
            test_pred=q123_test,
            train=train,
            anchors=anchor_tests,
            modified_targets=["Q1", "Q2", "Q3"],
            read="public-anchor Q1/Q2/Q3 sequence-summary only",
        ),
        evaluate_candidate(
            name="id178_q123_sequence_summary_plus_id337_s1_safety_gate",
            oof=q123_plus_id337_oof,
            test_pred=q123_plus_id337_test,
            train=train,
            anchors=anchor_tests,
            modified_targets=["Q1", "Q2", "Q3", "S1"],
            read="public-anchor Q1/Q2/Q3 sequence-summary plus ID337 S1 low-movement random-conv",
        ),
        evaluate_candidate(
            name="id178_q123s1_sequence_summary_gate",
            oof=full_oof,
            test_pred=full_test,
            train=train,
            anchors=anchor_tests,
            modified_targets=["Q1", "Q2", "Q3", "S1"],
            read="public-anchor sequence-summary on Q1/Q2/Q3/S1",
        ),
    ]
    return {
        "selector_info": selector_info,
        "candidates": sorted(
            candidates,
            key=lambda row: (
                row["overall"],
                row["geometry"]["max_delta_vs_178"],
                row["geometry"]["projection_failed"],
            ),
        ),
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sequence-summary geometry gate — post A1",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. No CSV, notebook, base prediction artifact, submission-log row, or DACON submission was created.",
        "",
        "## Candidate gate",
        "",
        "| rank | candidate | local | robust | max Δ178 | mean Δ178 | failed proj | worst target | modified | read |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for rank, row in enumerate(report["candidates"], start=1):
        geom = row["geometry"]
        slices = row["slice_summary"]
        lines.append(
            f"| {rank} | `{row['name']}` | {row['overall']:.6f} | {row['robust_score']:.6f} | "
            f"{geom['max_delta_vs_178']:.6f} | {geom['mean_delta_vs_178']:.6f} | "
            f"{geom['projection_failed']:+.6f} | {slices['worst_target']} `{slices['worst_target_loss']:.6f}` | "
            f"`{row['modified_targets']}` | {row['read']} |"
        )
    lines.extend(["", "## Target movement vs ID178", ""])
    for row in report["candidates"]:
        lines.append(f"### `{row['name']}`")
        lines.append("")
        lines.append("| target | max Δ178 | loss |")
        lines.append("|---|---:|---:|")
        for target in TARGETS:
            lines.append(
                f"| `{target}` | {row['max_delta_vs_178_by_target'][target]:.6f} | {row['target_scores'][target]:.6f} |"
            )
        lines.append("")
    decision = report["decision"]
    lines.extend(["## Decision", "", decision["reason"], ""])
    return "\n".join(lines)


def main() -> int:
    built = build_candidates()
    candidates = built["candidates"]
    best = candidates[0]
    # A deliberately cautious gate: no auto-promotion if any changed target has
    # large movement from ID178, even when local CV improves.
    promote = (
        best["overall"] < 0.5370
        and best["geometry"]["max_delta_vs_178"] <= 0.12
        and best["geometry"]["projection_failed"] <= 0.002
    )
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "sequence_summary_geometry_gate_post_a1",
        "selector_info": built["selector_info"],
        "candidates": candidates,
        "decision": {
            "best_candidate": best["name"],
            "create_submission_artifact": False,
            "public_058_plausible": False,
            "passes_internal_promotion_screen": bool(promote),
            "reason": (
                "Sequence-summary gives a real local improvement over ID178, but Q2 movement reaches about "
                "0.12 and the branch changes multiple targets. Treat this as a stronger candidate-construction "
                "lead, not an automatic ID339 promotion."
            ),
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    OUTPUT_PATH.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_PATH.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
