#!/usr/bin/env python3
"""Full-test geometry gate for the A1 q240-free S4 salvage lead.

The post-ID341 target salvage review found only S4 survives the q240-free A1
source replay. This diagnostic builds the corresponding in-memory test
prediction candidate:

    ID235 base, with S4 blended toward ID239 at prob alpha 0.55.

It then checks whether that target-only salvage is close enough to known public
anchors to justify a future generated-unsubmitted artifact. It is deliberately
report-only and never writes a submission CSV.
"""

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
    geometry,
    load_artifact,
    load_train_test,
    local_scores,
    robust_score,
    slice_summary,
)
from scripts.run_public_confirmed_anchor_bridge_probe_experiment import blend_values  # noqa: E402

REPORTS_DIR = ROOT / "reports"
TARGET_SALVAGE_PATH = REPORTS_DIR / "a1_q240_free_target_salvage_post_id341_20260618.json"
OUTPUT_PATH = REPORTS_DIR / "a1_s4_salvage_geometry_gate_post_id341_20260618.json"
KST = timezone(timedelta(hours=9))
BASE_ID = 235
SOURCE_ID = 239
TARGET = "S4"
METHOD = "prob"
ALPHA = 0.55
ANCHOR_IDS = [178, 287, 329, 335]


def build_candidate_frame(base: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out[TARGET] = blend_values(
        base[TARGET].astype(float).to_numpy(),
        source[TARGET].astype(float).to_numpy(),
        alpha=ALPHA,
        method=METHOD,
    )
    return out


def max_delta_by_target(candidate: pd.DataFrame, base: pd.DataFrame) -> dict[str, float]:
    return {
        target: float(np.max(np.abs(candidate[target].astype(float) - base[target].astype(float))))
        for target in TARGETS
    }


def mean_delta_by_target(candidate: pd.DataFrame, base: pd.DataFrame) -> dict[str, float]:
    return {
        target: float(np.mean(np.abs(candidate[target].astype(float) - base[target].astype(float))))
        for target in TARGETS
    }


def frame_delta(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    arr = candidate[TARGETS].astype(float).to_numpy()
    ref = reference[TARGETS].astype(float).to_numpy()
    return {
        "mean_abs_delta": float(np.mean(np.abs(arr - ref))),
        "max_abs_delta": float(np.max(np.abs(arr - ref))),
        "corr": float(np.corrcoef(arr.reshape(-1), ref.reshape(-1))[0, 1]),
    }


def public_risk_read(geom: dict[str, float]) -> dict[str, Any]:
    high_movement = geom["max_delta_vs_178"] > 0.18 or geom["mean_delta_vs_178"] > 0.08
    failed_direction = geom["projection_failed"] > 0.02
    reject = high_movement or failed_direction
    return {
        "risk_tier": "reject_high_public_risk" if reject else "reviewable_low_movement",
        "high_movement_vs_id178": bool(high_movement),
        "failed_direction_exposure": bool(failed_direction),
        "artifact_review_pass": bool(not reject),
        "public_058_plausible": False,
    }


def nested_hybrid_context() -> dict[str, Any] | None:
    if not TARGET_SALVAGE_PATH.exists():
        return None
    report = json.loads(TARGET_SALVAGE_PATH.read_text(encoding="utf-8"))
    return {
        "stable_salvage_targets": report["stable_salvage_targets"],
        "hybrid_rows": report["hybrid_rows"],
    }


def build_report() -> dict[str, Any]:
    train, test = load_train_test()
    base_oof = align(load_artifact("oof", BASE_ID), train)
    base_test = align(load_artifact("test", BASE_ID), test)
    source_oof = align(load_artifact("oof", SOURCE_ID), train)
    source_test = align(load_artifact("test", SOURCE_ID), test)
    candidate_oof = build_candidate_frame(base_oof, source_oof)
    candidate_test = build_candidate_frame(base_test, source_test)
    anchors = {candidate_id: align(load_artifact("test", candidate_id), test) for candidate_id in ANCHOR_IDS}

    local_cv, target_scores = local_scores(train, candidate_oof)
    base_local, base_target_scores = local_scores(train, base_oof)
    slices = slice_summary(train, candidate_oof)
    geom = geometry(candidate_test, anchors)
    risk = public_risk_read(geom)
    reference_deltas = {
        f"id{candidate_id}": frame_delta(candidate_test, anchors[candidate_id]) for candidate_id in ANCHOR_IDS
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "a1_s4_salvage_geometry_gate_post_id341",
        "candidate_design": {
            "name": "id235_s4_id239_prob_a055_salvage",
            "base_id": BASE_ID,
            "source_id": SOURCE_ID,
            "target": TARGET,
            "method": METHOD,
            "alpha": ALPHA,
            "artifact_status": "not_created",
        },
        "local_context_not_promotion_evidence": {
            "base_id235_local": base_local,
            "candidate_full_oof_local": local_cv,
            "candidate_delta_vs_base": local_cv - base_local,
            "base_target_scores": base_target_scores,
            "candidate_target_scores": target_scores,
            "nested_source_replay_context": nested_hybrid_context(),
            "warning": (
                "Candidate full-OOF local uses pre-existing ID235/ID239 artifacts. The nested replay context is "
                "stronger evidence for the S4 direction, but neither is public proof."
            ),
        },
        "slice_summary": {
            "worst_subject": slices["worst_subject"],
            "worst_subject_loss": slices["worst_subject_loss"],
            "worst_date_bin": slices["worst_date_bin"],
            "worst_date_loss": slices["worst_date_loss"],
            "worst_target": slices["worst_target"],
            "worst_target_loss": slices["worst_target_loss"],
            "late_minus_early": slices["late_minus_early"],
        },
        "robust_score": robust_score(local_cv, slices),
        "geometry": geom,
        "public_risk_read": risk,
        "candidate_vs_references": reference_deltas,
        "delta_vs_base_id235_by_target": {
            "max": max_delta_by_target(candidate_test, base_test),
            "mean": mean_delta_by_target(candidate_test, base_test),
        },
        "decision": {
            "promote_submission_candidate_now": False,
            "create_artifact_now": False,
            "submit_now": False,
            "artifact_review_pass": bool(risk["artifact_review_pass"]),
            "public_058_plausible": False,
            "read": (
                "S4 survives the q240-free source replay, but this gate must reject the candidate if it is too far "
                "from the confirmed ID178 public anchor or aligned with the failed ID287 direction."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    design = report["candidate_design"]
    local = report["local_context_not_promotion_evidence"]
    geom = report["geometry"]
    risk = report["public_risk_read"]
    lines = [
        "# A1 S4 salvage geometry gate — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no CSV, notebook/history artifact, base prediction artifact, submission-log row, or DACON submission.",
        "",
        "## Candidate design",
        "",
        f"- Name: `{design['name']}`",
        f"- Base: `ID{design['base_id']}`",
        f"- Source: `ID{design['source_id']}`",
        f"- Modified target: `{design['target']}` via `{design['method']}` alpha `{design['alpha']}`",
        "",
        "## Local context",
        "",
        f"- Base ID235 local: `{local['base_id235_local']:.6f}`",
        f"- Candidate full-OOF local: `{local['candidate_full_oof_local']:.6f}`",
        f"- Δ vs base: `{local['candidate_delta_vs_base']:+.6f}`",
        f"- Robust score: `{report['robust_score']:.6f}`",
        "",
        "## Public-risk geometry",
        "",
        f"- Mean Δ vs ID178: `{geom['mean_delta_vs_178']:.6f}`",
        f"- Max Δ vs ID178: `{geom['max_delta_vs_178']:.6f}`",
        f"- Mean Δ vs ID329: `{geom['mean_delta_vs_329']:.6f}`",
        f"- Max Δ vs ID329: `{geom['max_delta_vs_329']:.6f}`",
        f"- Failed-direction projection: `{geom['projection_failed']:+.6f}`",
        f"- Risk tier: `{risk['risk_tier']}`",
        "",
        "## Reference distances",
        "",
        "| reference | mean abs Δ | max abs Δ | corr |",
        "|---|---:|---:|---:|",
    ]
    for ref_name, row in report["candidate_vs_references"].items():
        lines.append(f"| `{ref_name}` | {row['mean_abs_delta']:.6f} | {row['max_abs_delta']:.6f} | {row['corr']:.6f} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Artifact review pass: `{report['decision']['artifact_review_pass']}`",
            "- Create artifact now: **no**",
            "- Submit now: **no**",
            "- Public 0.58 plausible: **no**",
            "",
            report["decision"]["read"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    output = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
