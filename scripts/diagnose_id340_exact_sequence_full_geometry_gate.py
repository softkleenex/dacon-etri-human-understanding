#!/usr/bin/env python3
"""Full-train/test geometry gate for the exact ID340 sequence-summary lead.

This report trains the exact source design on the full training set, creates
candidate test predictions in memory, and evaluates movement/proxy geometry
against confirmed public anchors. It creates no submission CSV, base prediction
artifact, notebook/history artifact, submission-log row, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_id179_vs_id335_sequence_probe_review import geometry  # noqa: E402
from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    RAW_DIR,
    ModelSpec,
    align_table,
    build_daily_sequence_table,
    make_model,
    table_to_tensor,
)
from scripts.diagnose_public_sequence_summary_embedding import sequence_summary_features  # noqa: E402
from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_strict_raw_feature_experiment import TARGETS, build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
BASE_ID = 178
REFERENCE_IDS = [178, 287, 329, 335, 337, 339]
MODEL_SPEC = ModelSpec("lr_c0_3", "lr", c_value=0.3)
FIXED_BLEND_WEIGHT = 0.03
ID178_PUBLIC_SCORE = 0.6014113009
ID329_PUBLIC_SCORE = 0.6017908577
ID287_PUBLIC_SCORE = 0.6698787318
ID178_LOCAL_REFERENCE = 0.5386382667903545
PUBLIC_RERANK_PATH = ROOT / "reports" / "post_remote_parity_public_anchor_rerank_post_id339_20260618.json"
CANDIDATE_DECISION_PATH = ROOT / "reports" / "id340_sequence_summary_candidate_decision_20260618.json"


def target_losses(train: pd.DataFrame, values: pd.DataFrame) -> dict[str, float]:
    return {
        target: float(log_loss(train[target].astype(int), values[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }


def frame_like(keys: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "subject_id": keys["subject_id"].to_numpy(),
            "date": pd.to_datetime(keys["date"]).dt.date,
        }
    )
    for target_idx, target in enumerate(TARGETS):
        out[target] = np.clip(values[:, target_idx], 1e-6, 1 - 1e-6)
    return out


def train_full_sequence_source(
    x_train: np.ndarray,
    x_test: np.ndarray,
    train: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_train = np.zeros((len(train), len(TARGETS)), dtype=float)
    source_test = np.zeros((x_test.shape[0], len(TARGETS)), dtype=float)
    details: dict[str, Any] = {}
    for target_idx, target in enumerate(TARGETS):
        y = train[target].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = clone(make_model(MODEL_SPEC))
        model.fit(x_train, y)
        source_train[:, target_idx] = model.predict_proba(x_train)[:, 1]
        source_test[:, target_idx] = model.predict_proba(x_test)[:, 1]
        details[target] = {
            "positive_rate_train": float(np.mean(y)),
            "source_train_mean": float(np.mean(source_train[:, target_idx])),
            "source_test_mean": float(np.mean(source_test[:, target_idx])),
            "source_test_min": float(np.min(source_test[:, target_idx])),
            "source_test_max": float(np.max(source_test[:, target_idx])),
            "source_test_std": float(np.std(source_test[:, target_idx])),
        }
    return (
        np.clip(source_train, 1e-6, 1 - 1e-6),
        np.clip(source_test, 1e-6, 1 - 1e-6),
        details,
    )


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


def conservative_public_proxy(geom: dict[str, float], local_cv: float | None) -> float:
    base = min(ID178_PUBLIC_SCORE, ID329_PUBLIC_SCORE)
    movement = 0.22 * geom["mean_delta_vs_178"] + 0.025 * geom["max_delta_vs_178"]
    neutral = 0.08 * geom["mean_delta_vs_329"] + 0.010 * geom["max_delta_vs_329"]
    failed = 0.18 * max(0.0, geom["projection_failed"])
    if local_cv is None:
        local_penalty = 0.0
    else:
        low_local_pressure = max(0.0, ID178_LOCAL_REFERENCE - local_cv)
        local_penalty = 0.25 * max(0.0, low_local_pressure - 0.02)
    return float(base + movement + neutral + failed + local_penalty)


def risk_tier(geom: dict[str, float]) -> str:
    if geom["projection_failed"] > 0.06 or geom["max_delta_vs_178"] > 0.18 or geom["max_delta_vs_329"] > 0.22:
        return "reject_high_public_risk"
    if geom["max_delta_vs_178"] <= 0.12 and geom["max_delta_vs_329"] <= 0.17 and geom["projection_failed"] <= 0.02:
        return "reviewable_low_movement"
    return "diagnostic_only"


def rerank_proxy_by_id(candidate_id: int) -> float | None:
    if not PUBLIC_RERANK_PATH.exists():
        return None
    report = json.loads(PUBLIC_RERANK_PATH.read_text(encoding="utf-8"))
    for row in report.get("top_all_by_proxy", []):
        if int(row["id"]) == candidate_id:
            return float(row["conservative_public_risk_proxy"])
    return None


def split_decision_rows() -> list[dict[str, Any]]:
    if not CANDIDATE_DECISION_PATH.exists():
        return []
    return json.loads(CANDIDATE_DECISION_PATH.read_text(encoding="utf-8")).get("split_rows", [])


def build_report() -> dict[str, Any]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    reference_oof: dict[int, pd.DataFrame] = {}
    reference_test: dict[int, pd.DataFrame] = {}
    reference_source: dict[int, str] = {}
    for candidate_id in REFERENCE_IDS:
        oof, test_pred, source = load_artifact_pair(train, test, candidate_id)
        reference_oof[candidate_id] = oof
        reference_test[candidate_id] = test_pred
        reference_source[candidate_id] = source

    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    x_train, feature_names = sequence_summary_features(train_tensor, channel_names)
    x_test, _ = sequence_summary_features(test_tensor, channel_names)
    source_train_values, source_test_values, source_details = train_full_sequence_source(x_train, x_test, train)

    base_oof = reference_oof[BASE_ID]
    base_test = reference_test[BASE_ID]
    candidate_train_values = (1.0 - FIXED_BLEND_WEIGHT) * base_oof[TARGETS].astype(
        float
    ).to_numpy() + FIXED_BLEND_WEIGHT * source_train_values
    candidate_test_values = (1.0 - FIXED_BLEND_WEIGHT) * base_test[TARGETS].astype(
        float
    ).to_numpy() + FIXED_BLEND_WEIGHT * source_test_values
    source_train_frame = frame_like(train, source_train_values)
    candidate_train_frame = frame_like(train, candidate_train_values)
    candidate_test_frame = frame_like(test, candidate_test_values)

    anchors = {candidate_id: reference_test[candidate_id] for candidate_id in [178, 287, 329, 335]}
    geom = geometry(candidate_test_frame, anchors)
    split_rows = split_decision_rows()
    subject_hole = next((row for row in split_rows if row["split"] == "subject_hole_5"), None)
    split_subject_hole_loss = None if subject_hole is None else float(subject_hole["mean_loss"])
    split_subject_hole_delta = None if subject_hole is None else float(subject_hole["delta_vs_id178"])
    proxy = conservative_public_proxy(geom, split_subject_hole_loss)
    id337_proxy = rerank_proxy_by_id(337)
    id339_proxy = rerank_proxy_by_id(339)

    candidate_vs = {}
    for candidate_id in [337, 339, 329, 287]:
        arr = candidate_test_frame[TARGETS].astype(float).to_numpy()
        ref = reference_test[candidate_id][TARGETS].astype(float).to_numpy()
        candidate_vs[f"id{candidate_id}"] = {
            "mean_abs_delta": float(np.mean(np.abs(arr - ref))),
            "max_abs_delta": float(np.max(np.abs(arr - ref))),
        }

    in_sample_losses = target_losses(train, candidate_train_frame)
    base_losses = target_losses(train, base_oof)
    source_in_sample_losses = target_losses(train, source_train_frame)
    geometry_pass = (
        geom["max_delta_vs_178"] <= 0.05
        and geom["projection_failed"] <= 0.002
        and (id337_proxy is None or proxy < id337_proxy)
        and (id339_proxy is None or proxy < id339_proxy)
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "id340_exact_sequence_full_geometry_gate",
        "candidate_design": {
            "name": "blend_id178_sequence_summary_lr_c0_3_w0.03",
            "base_id": BASE_ID,
            "source_model": asdict(MODEL_SPEC),
            "fixed_blend_weight": FIXED_BLEND_WEIGHT,
            "modified_targets": TARGETS,
            "artifact_status": "not_created",
        },
        "sequence_feature_summary": {
            "n_sequence_rows": int(len(seq_table)),
            "n_total_channels": int(len(channel_names)),
            "n_summary_features": int(x_train.shape[1]),
            "feature_family_count": int(len(feature_names) // max(len(channel_names), 1)),
        },
        "reference_sources": reference_source,
        "split_probe_context": {
            "source": str(CANDIDATE_DECISION_PATH.relative_to(ROOT)),
            "split_rows": split_rows,
            "subject_hole_loss": split_subject_hole_loss,
            "subject_hole_delta_vs_id178": split_subject_hole_delta,
            "primary_gate_pass_from_probe": all(
                row.get("delta_vs_id178", 1.0) < 0 for row in split_rows if row.get("is_primary")
            )
            if split_rows
            else None,
        },
        "geometry": geom,
        "risk_tier": risk_tier(geom),
        "conservative_public_risk_proxy": proxy,
        "proxy_comparison": {
            "id337_proxy": id337_proxy,
            "id339_proxy": id339_proxy,
            "delta_vs_id337_proxy": None if id337_proxy is None else float(proxy - id337_proxy),
            "delta_vs_id339_proxy": None if id339_proxy is None else float(proxy - id339_proxy),
            "naive_id178_gap_projected_public": None
            if split_subject_hole_loss is None
            else float(split_subject_hole_loss + (ID178_PUBLIC_SCORE - ID178_LOCAL_REFERENCE)),
        },
        "candidate_vs_reference_test_delta": candidate_vs,
        "test_delta_vs_id178_by_target": {
            "max": max_delta_by_target(candidate_test_frame, base_test),
            "mean": mean_delta_by_target(candidate_test_frame, base_test),
        },
        "source_prediction_details": source_details,
        "loss_context_not_cv": {
            "warning": (
                "These losses blend ID178 OOF with full-train in-sample source predictions. They are diagnostic "
                "only and must not be used as local CV evidence."
            ),
            "base_id178_oof_losses": base_losses,
            "source_full_train_in_sample_losses": source_in_sample_losses,
            "candidate_mixed_in_sample_losses": in_sample_losses,
            "candidate_mixed_in_sample_mean": float(np.mean(list(in_sample_losses.values()))),
        },
        "decision": {
            "passes_full_geometry_gate_for_artifact_review": bool(geometry_pass),
            "promote_submission_candidate_now": False,
            "create_id340_artifact_now": False,
            "submit_now": False,
            "public_058_plausible": proxy <= 0.590,
            "read": (
                "The exact design is safe enough for artifact-review if movement/proxy are below ID337/ID339, "
                "but it remains report-only here. Public 0.58 requires stronger evidence than this proxy gate."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    geom = report["geometry"]
    proxy = report["proxy_comparison"]
    decision = report["decision"]
    lines = [
        "# ID340 exact sequence-summary full geometry gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It trains the exact source on the full train set and creates candidate test predictions in memory only. No CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission was created.",
        "",
        "## Candidate design",
        "",
        f"- Name: `{report['candidate_design']['name']}`",
        f"- Base: `ID{report['candidate_design']['base_id']}`",
        f"- Source model: `{report['candidate_design']['source_model']}`",
        f"- Fixed blend weight: `{report['candidate_design']['fixed_blend_weight']}`",
        f"- Modified targets: `{report['candidate_design']['modified_targets']}`",
        "",
        "## Geometry",
        "",
        f"- Mean Δ vs ID178: `{geom['mean_delta_vs_178']:.6f}`",
        f"- Max Δ vs ID178: `{geom['max_delta_vs_178']:.6f}`",
        f"- Mean Δ vs ID329: `{geom['mean_delta_vs_329']:.6f}`",
        f"- Max Δ vs ID329: `{geom['max_delta_vs_329']:.6f}`",
        f"- Failed-direction projection: `{geom['projection_failed']:+.6f}`",
        f"- Conservative public-risk proxy: `{report['conservative_public_risk_proxy']:.6f}`",
        f"- Proxy Δ vs ID337: `{proxy['delta_vs_id337_proxy']:+.6f}`",
        f"- Proxy Δ vs ID339: `{proxy['delta_vs_id339_proxy']:+.6f}`",
        f"- Naive ID178-gap projected public: `{proxy['naive_id178_gap_projected_public']:.6f}`",
        f"- Risk tier: `{report['risk_tier']}`",
        "",
        "## Target movement vs ID178 test",
        "",
        "| target | mean Δ | max Δ | source test mean | source test std |",
        "|---|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        details = report["source_prediction_details"][target]
        lines.append(
            f"| `{target}` | {report['test_delta_vs_id178_by_target']['mean'][target]:.6f} | "
            f"{report['test_delta_vs_id178_by_target']['max'][target]:.6f} | "
            f"{details['source_test_mean']:.6f} | {details['source_test_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Split-probe context",
            "",
            "| split | primary | candidate loss | ID178 loss | Δ vs ID178 | pass |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in report["split_probe_context"]["split_rows"]:
        delta = float(row["delta_vs_id178"])
        lines.append(
            f"| `{row['split']}` | `{row['is_primary']}` | {float(row['mean_loss']):.6f} | "
            f"{float(row['id178_loss']):.6f} | {delta:+.6f} | `{delta < 0}` |"
        )
    lines.extend(
        [
            "",
            "## Loss context warning",
            "",
            report["loss_context_not_cv"]["warning"],
            "",
            "## Decision",
            "",
            f"- Passes full geometry gate for artifact review: `{decision['passes_full_geometry_gate_for_artifact_review']}`",
            f"- Public `0.58` plausible: `{decision['public_058_plausible']}`",
            "- Promote submission candidate now: **no**",
            "- Create ID340 artifact now: **no**",
            "- Submit now: **no**",
            "",
            decision["read"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "id340_exact_sequence_full_geometry_gate_20260618.json",
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
