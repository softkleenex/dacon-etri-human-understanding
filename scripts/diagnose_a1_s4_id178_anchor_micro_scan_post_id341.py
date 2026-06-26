#!/usr/bin/env python3
"""ID178-anchored micro scan for the A1 q240-free S4 signal.

The A1 S4 salvage survives nested source replay, but the direct ID235-based
candidate is far from the confirmed public anchor. This report tests a safer
question: can the same S4 direction be imported into the ID178 public anchor at
very small weights without violating frozen split and geometry gates?

Report-only: no CSV, notebook/history artifact, base prediction artifact,
submission-log row, or DACON submission is created.
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
from sklearn.metrics import log_loss

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
)
from scripts.diagnose_id340_exact_sequence_full_geometry_gate import conservative_public_proxy  # noqa: E402
from scripts.diagnose_id340_sequence_summary_split_probe import load_split_manifest  # noqa: E402
from scripts.run_public_confirmed_anchor_bridge_probe_experiment import blend_values  # noqa: E402

REPORTS_DIR = ROOT / "reports"
SPLIT_MANIFEST_PATH = REPORTS_DIR / "id340_validation_split_manifest_20260618.json"
OUTPUT_PATH = REPORTS_DIR / "a1_s4_id178_anchor_micro_scan_post_id341_20260618.json"
KST = timezone(timedelta(hours=9))
BASE_ID = 178
SOURCE_ID = 239
TARGET = "S4"
METHOD = "prob"
ANCHOR_IDS = [178, 287, 329, 335]
WEIGHTS = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10]
PRIMARY_GATE_SPLITS = [
    "subject_hole_5",
    "actual_test_band_holdout",
    "test_boundary_neighbor2_holdout",
    "test_boundary_neighbor5_holdout",
    "loso_10",
]
REFERENCE_PROXIES = {
    "id341": 0.6038297835283394,
    "id340": 0.604062,
    "id337": 0.605022,
    "id339": 0.609547,
}


def build_candidate(base: pd.DataFrame, source: pd.DataFrame, weight: float) -> pd.DataFrame:
    out = base.copy()
    out[TARGET] = blend_values(
        base[TARGET].astype(float).to_numpy(),
        source[TARGET].astype(float).to_numpy(),
        alpha=weight,
        method=METHOD,
    )
    return out


def split_loss(train: pd.DataFrame, pred: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    target_losses = []
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        values = pred[target].astype(float).to_numpy()
        total = 0
        weighted = 0.0
        for _train_idx, valid_idx in folds:
            loss = float(log_loss(y[valid_idx], np.clip(values[valid_idx], 1e-6, 1 - 1e-6), labels=[0, 1]))
            weighted += loss * len(valid_idx)
            total += len(valid_idx)
        target_losses.append(weighted / total)
    return float(np.mean(target_losses))


def target_split_loss(
    train: pd.DataFrame, pred: pd.DataFrame, target: str, folds: list[tuple[np.ndarray, np.ndarray]]
) -> float:
    y = train[target].astype(int).to_numpy()
    values = pred[target].astype(float).to_numpy()
    total = 0
    weighted = 0.0
    for _train_idx, valid_idx in folds:
        loss = float(log_loss(y[valid_idx], np.clip(values[valid_idx], 1e-6, 1 - 1e-6), labels=[0, 1]))
        weighted += loss * len(valid_idx)
        total += len(valid_idx)
    return float(weighted / total)


def split_rows_for_candidate(train: pd.DataFrame, base: pd.DataFrame, candidate: pd.DataFrame) -> list[dict[str, Any]]:
    splits = load_split_manifest(SPLIT_MANIFEST_PATH)
    rows = []
    for split_name, folds in splits.items():
        base_loss = split_loss(train, base, folds)
        candidate_loss = split_loss(train, candidate, folds)
        base_target = target_split_loss(train, base, TARGET, folds)
        candidate_target = target_split_loss(train, candidate, TARGET, folds)
        rows.append(
            {
                "split": split_name,
                "is_primary": split_name in PRIMARY_GATE_SPLITS,
                "base_loss": base_loss,
                "candidate_loss": candidate_loss,
                "delta_vs_base": candidate_loss - base_loss,
                "base_target_loss": base_target,
                "candidate_target_loss": candidate_target,
                "target_delta_vs_base": candidate_target - base_target,
            }
        )
    return rows


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


def build_report() -> dict[str, Any]:
    train, test = load_train_test()
    base_oof = align(load_artifact("oof", BASE_ID), train)
    base_test = align(load_artifact("test", BASE_ID), test)
    source_oof = align(load_artifact("oof", SOURCE_ID), train)
    source_test = align(load_artifact("test", SOURCE_ID), test)
    anchors = {candidate_id: align(load_artifact("test", candidate_id), test) for candidate_id in ANCHOR_IDS}
    base_local, base_targets = local_scores(train, base_oof)

    variants = []
    for weight in WEIGHTS:
        candidate_oof = build_candidate(base_oof, source_oof, weight)
        candidate_test = build_candidate(base_test, source_test, weight)
        local_cv, target_scores = local_scores(train, candidate_oof)
        splits = split_rows_for_candidate(train, base_oof, candidate_oof)
        primary_pass = all(row["delta_vs_base"] < 0 for row in splits if row["is_primary"])
        target_primary_pass = all(row["target_delta_vs_base"] < 0 for row in splits if row["is_primary"])
        geom = geometry(candidate_test, anchors)
        proxy = conservative_public_proxy(
            geom, next(row["candidate_loss"] for row in splits if row["split"] == "subject_hole_5")
        )
        artifact_review_pass = (
            primary_pass
            and target_primary_pass
            and geom["max_delta_vs_178"] <= 0.05
            and geom["projection_failed"] <= 0.002
            and proxy < REFERENCE_PROXIES["id341"]
        )
        variants.append(
            {
                "name": f"id178_s4_id239_{METHOD}_w{weight:g}",
                "weight": weight,
                "local_cv": local_cv,
                "delta_vs_id178_local": local_cv - base_local,
                "target_scores": target_scores,
                "s4_delta_vs_id178_local": target_scores[TARGET] - base_targets[TARGET],
                "split_rows": splits,
                "primary_gate_pass": bool(primary_pass),
                "target_primary_gate_pass": bool(target_primary_pass),
                "geometry": geom,
                "conservative_public_risk_proxy": proxy,
                "proxy_delta_vs_id341": float(proxy - REFERENCE_PROXIES["id341"]),
                "artifact_review_pass": bool(artifact_review_pass),
                "public_058_plausible": bool(proxy <= 0.590),
                "delta_vs_id178_by_target": {
                    "max": max_delta_by_target(candidate_test, base_test),
                    "mean": mean_delta_by_target(candidate_test, base_test),
                },
            }
        )

    variants = sorted(
        variants,
        key=lambda row: (
            not row["artifact_review_pass"],
            not row["primary_gate_pass"],
            row["conservative_public_risk_proxy"],
            row["geometry"]["max_delta_vs_178"],
            row["weight"],
        ),
    )
    best = variants[0]
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "a1_s4_id178_anchor_micro_scan_post_id341",
        "purpose": "Test whether the A1 q240-free S4 direction can be safely imported into the confirmed ID178 public anchor at very small weights.",
        "base_id": BASE_ID,
        "source_id": SOURCE_ID,
        "target": TARGET,
        "method": METHOD,
        "weights": WEIGHTS,
        "reference_proxies": REFERENCE_PROXIES,
        "base_id178_local": base_local,
        "base_id178_target_scores": base_targets,
        "variants": variants,
        "best_variant": best,
        "decision": {
            "promote_submission_candidate_now": False,
            "create_artifact_now": False,
            "submit_now": False,
            "has_artifact_review_variant": any(row["artifact_review_pass"] for row in variants),
            "best_variant": best["name"],
            "best_artifact_review_pass": bool(best["artifact_review_pass"]),
            "best_public_058_plausible": bool(best["public_058_plausible"]),
            "read": (
                "This scan is still full-OOF-source assisted. A variant can only move to artifact review if it "
                "beats ID341's conservative proxy while passing frozen split and geometry gates."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# A1 S4 ID178-anchor micro scan — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no CSV, notebook/history artifact, base prediction artifact, submission-log row, or DACON submission.",
        "",
        f"- Base: `ID{report['base_id']}`",
        f"- Source: `ID{report['source_id']}`",
        f"- Target: `{report['target']}`",
        f"- Method: `{report['method']}`",
        f"- ID341 proxy reference: `{report['reference_proxies']['id341']:.6f}`",
        "",
        "## Top variants",
        "",
        "| rank | variant | artifact review | primary pass | target primary pass | local | S4 Δ local | proxy | Δ proxy vs ID341 | max Δ178 | failed proj |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(report["variants"], start=1):
        lines.append(
            f"| {rank} | `{row['name']}` | `{row['artifact_review_pass']}` | `{row['primary_gate_pass']}` | "
            f"`{row['target_primary_gate_pass']}` | {row['local_cv']:.6f} | "
            f"{row['s4_delta_vs_id178_local']:+.6f} | {row['conservative_public_risk_proxy']:.6f} | "
            f"{row['proxy_delta_vs_id341']:+.6f} | {row['geometry']['max_delta_vs_178']:.6f} | "
            f"{row['geometry']['projection_failed']:+.6f} |"
        )
    best = report["best_variant"]
    lines.extend(
        [
            "",
            "## Best variant split rows",
            "",
            f"Best variant: `{best['name']}`",
            "",
            "| split | primary | loss | ID178 | Δ | S4 loss | ID178 S4 | S4 Δ | pass |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in best["split_rows"]:
        lines.append(
            f"| `{row['split']}` | `{row['is_primary']}` | {row['candidate_loss']:.6f} | "
            f"{row['base_loss']:.6f} | {row['delta_vs_base']:+.6f} | "
            f"{row['candidate_target_loss']:.6f} | {row['base_target_loss']:.6f} | "
            f"{row['target_delta_vs_base']:+.6f} | `{row['delta_vs_base'] < 0 and row['target_delta_vs_base'] < 0}` |"
        )
    decision = report["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Has artifact-review variant: `{decision['has_artifact_review_variant']}`",
            f"- Best artifact-review pass: `{decision['best_artifact_review_pass']}`",
            f"- Best Public 0.58 plausible: `{decision['best_public_058_plausible']}`",
            "- Create artifact now: **no**",
            "- Submit now: **no**",
            "",
            decision["read"],
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
