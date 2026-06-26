#!/usr/bin/env python3
"""Cap sensitivity for post-A1 sequence-summary + ID337-S1 candidate.

This diagnostic searches for a lower-movement variant of the strongest
post-A1 sequence-summary lead. It is report-only and creates no DACON-facing
artifact.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_id179_vs_id335_sequence_probe_review import (  # noqa: E402
    EPS,
    align,
    geometry,
    load_artifact,
    load_train_test,
    local_scores,
    robust_score,
    slice_summary,
)
from scripts.diagnose_sequence_summary_geometry_gate_post_a1 import FULL_ID178_SEQUENCE_CONFIGS  # noqa: E402
from scripts.run_public_sequence_summary_embedding_overlay_experiment import (  # noqa: E402
    predict_sequence_summary_embedding_overlay,
)

REPORTS_DIR = ROOT / "reports"
OUTPUT_PATH = REPORTS_DIR / "sequence_summary_cap_sensitivity_post_a1_20260618.json"
KST = timezone(timedelta(hours=9))
SCALE_GRID = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
Q_TARGETS = ["Q1", "Q2", "Q3"]


def derive_raw_from_blend(base_values: np.ndarray, blended_values: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0:
        raise ValueError("Cannot derive raw prediction from zero-weight blend.")
    return np.clip((blended_values - (1.0 - weight) * base_values) / weight, EPS, 1.0 - EPS)


def evaluate(name: str, oof, test_pred, train, anchors, scale: float) -> dict[str, Any]:
    overall, target_scores = local_scores(train, oof)
    slices = slice_summary(train, oof)
    geom = geometry(test_pred, anchors)
    return {
        "name": name,
        "scale": scale,
        "overall": overall,
        "target_scores": target_scores,
        "robust_score": robust_score(overall, slices),
        "slice_summary": {
            "worst_subject": slices["worst_subject"],
            "worst_subject_loss": slices["worst_subject_loss"],
            "worst_target": slices["worst_target"],
            "worst_target_loss": slices["worst_target_loss"],
            "late_minus_early": slices["late_minus_early"],
        },
        "geometry": geom,
    }


def build_report() -> dict[str, Any]:
    train, test = load_train_test()
    anchors = {candidate_id: align(load_artifact("test", candidate_id), test) for candidate_id in [178, 287, 329, 335]}
    base_oof = align(load_artifact("oof", 178), train)
    base_test = align(load_artifact("test", 178), test)
    id337_oof = align(load_artifact("oof", 337), train)
    id337_test = align(load_artifact("test", 337), test)

    full_oof, full_test, _scores, _selector_info, _sample_original = predict_sequence_summary_embedding_overlay(
        178,
        selected_configs=FULL_ID178_SEQUENCE_CONFIGS,
        diagnostic_reports=["reports/id178_sequence_summary_embedding_post_a1_20260618.json"],
    )
    full_oof = align(full_oof, train)
    full_test = align(full_test, test)

    raw_oof = {}
    raw_test = {}
    for target in Q_TARGETS:
        weight = float(FULL_ID178_SEQUENCE_CONFIGS[target]["weight"])
        raw_oof[target] = derive_raw_from_blend(
            base_oof[target].astype(float).to_numpy(),
            full_oof[target].astype(float).to_numpy(),
            weight,
        )
        raw_test[target] = derive_raw_from_blend(
            base_test[target].astype(float).to_numpy(),
            full_test[target].astype(float).to_numpy(),
            weight,
        )

    rows = []
    for scale in SCALE_GRID:
        candidate_oof = base_oof.copy()
        candidate_test = base_test.copy()
        for target in Q_TARGETS:
            original_weight = float(FULL_ID178_SEQUENCE_CONFIGS[target]["weight"])
            scaled_weight = original_weight * scale
            candidate_oof[target] = np.clip(
                (1.0 - scaled_weight) * base_oof[target].astype(float).to_numpy() + scaled_weight * raw_oof[target],
                EPS,
                1.0 - EPS,
            )
            candidate_test[target] = np.clip(
                (1.0 - scaled_weight) * base_test[target].astype(float).to_numpy() + scaled_weight * raw_test[target],
                EPS,
                1.0 - EPS,
            )
        candidate_oof["S1"] = id337_oof["S1"]
        candidate_test["S1"] = id337_test["S1"]
        rows.append(
            evaluate(
                name=f"id178_q123_summary_scale{scale:.2f}_plus_id337_s1",
                oof=candidate_oof,
                test_pred=candidate_test,
                train=train,
                anchors=anchors,
                scale=scale,
            )
        )
    best_by_local = min(rows, key=lambda row: row["overall"])
    cap_rows = {
        "cap010": [row for row in rows if row["geometry"]["max_delta_vs_178"] <= 0.10 + 1e-12],
        "cap011": [row for row in rows if row["geometry"]["max_delta_vs_178"] <= 0.11 + 1e-12],
        "cap012": [row for row in rows if row["geometry"]["max_delta_vs_178"] <= 0.12 + 1e-12],
    }
    best_by_cap = {
        cap: min(items, key=lambda row: (row["overall"], row["robust_score"])) if items else None
        for cap, items in cap_rows.items()
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "sequence_summary_cap_sensitivity_post_a1",
        "scale_grid": SCALE_GRID,
        "rows": rows,
        "best_by_local": best_by_local,
        "best_by_cap": best_by_cap,
        "decision": {
            "create_submission_artifact": False,
            "public_058_plausible": False,
            "preferred_report_only_variant": best_by_cap["cap010"]["name"] if best_by_cap["cap010"] else None,
            "reason": (
                "Shrinking Q1/Q2/Q3 sequence-summary weights can bring max movement below 0.10 while keeping "
                "a meaningful local gain over ID337. This is now a cleaner candidate-construction target, but "
                "still needs explicit promotion and artifact-contract creation before any submission discussion."
            ),
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sequence-summary cap sensitivity — post A1",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. No CSV, notebook, base prediction artifact, submission-log row, or DACON submission was created.",
        "",
        "## Scale sweep",
        "",
        "| scale | local | robust | max Δ178 | mean Δ178 | failed proj | worst target |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        geom = row["geometry"]
        slices = row["slice_summary"]
        lines.append(
            f"| {row['scale']:.2f} | {row['overall']:.6f} | {row['robust_score']:.6f} | "
            f"{geom['max_delta_vs_178']:.6f} | {geom['mean_delta_vs_178']:.6f} | "
            f"{geom['projection_failed']:+.6f} | {slices['worst_target']} `{slices['worst_target_loss']:.6f}` |"
        )
    lines.extend(["", "## Best by cap", "", "| cap | variant | local | max Δ178 |", "|---|---|---:|---:|"])
    for cap, row in report["best_by_cap"].items():
        if row is None:
            lines.append(f"| `{cap}` | none |  |  |")
        else:
            lines.append(
                f"| `{cap}` | `{row['name']}` | {row['overall']:.6f} | {row['geometry']['max_delta_vs_178']:.6f} |"
            )
    lines.extend(["", "## Decision", "", report["decision"]["reason"], ""])
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    OUTPUT_PATH.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_PATH.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
