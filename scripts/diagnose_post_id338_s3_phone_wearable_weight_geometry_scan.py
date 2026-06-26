#!/usr/bin/env python3
"""Weight geometry scan for ID338 S3 phone/wearable overlays.

Report-only diagnostic. It compares small S3-only phone/wearable overlay weights
against ID338 using existing fold-internal probe evidence plus in-memory test
geometry. It creates no submission artifact.
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

from scripts.diagnose_post_id338_material_q3s3_geometry_gate import (  # noqa: E402
    TRAIN_PATH,
    TEST_PATH,
    fit_predict_test_source,
    load_base_test_submission,
    load_source_train_test,
    metric_row_for_submission,
)
from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402
    EPS,
    load_train_sorted,
)
from scripts.diagnose_public_joint_distribution_risk_post_id341 import train_reference_stats  # noqa: E402

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
PROBE_REPORT = REPORTS_DIR / "post_id338_material_q3s3_probe_20260623.json"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id338_s3_phone_wearable_weight_geometry_scan_20260624.json"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
TARGET = "S3"
SOURCE_GROUP = "phone_wearable_discordance"
MODEL_SPEC = {"name": "lr_c0_20", "kind": "lr", "c_value": 0.20}
WEIGHTS = [0.0025, 0.005, 0.01, 0.02, 0.035, 0.05]
PUBLIC_ID338 = 0.5960832519


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def candidate_name(weight: float) -> str:
    return f"blend_id338_{SOURCE_GROUP}_{MODEL_SPEC['name']}_w{weight:g}"


def probe_summary_by_weight() -> dict[float, dict[str, Any]]:
    report = json.loads(PROBE_REPORT.read_text(encoding="utf-8"))
    rows = report.get("top_rows_by_target", {}).get(TARGET, [])
    out: dict[float, dict[str, Any]] = {}
    for row in rows:
        if (
            row.get("source_group") == SOURCE_GROUP
            and row.get("model_spec", {}).get("name") == MODEL_SPEC["name"]
            and row.get("weight") is not None
        ):
            out[float(row["weight"])] = row
    # top_rows may be truncated in older reports; selected config is always present.
    selected = report.get("selected_target_configs", {}).get(TARGET)
    if selected and selected.get("source_group") == SOURCE_GROUP:
        out[float(selected["weight"])] = selected
    return out


def risk_flags_for(row: dict[str, Any]) -> list[str]:
    flags = []
    if row["mean_abs_all_targets_vs_id338"] > 0.00075:
        flags.append("mean_all_delta_vs_id338_gt_0.00075")
    if row["max_abs_all_targets_vs_id338"] > 0.020:
        flags.append("max_all_delta_vs_id338_gt_0.020")
    if row["joint_deltas_vs_id338"].get("target_mean_l1_to_train", 0.0) > 0.001:
        flags.append("target_mean_l1_worse_gt_0.001")
    if row["joint_deltas_vs_id338"].get("pair11_l1_to_train", 0.0) > 0.001:
        flags.append("pair11_l1_worse_gt_0.001")
    if row["joint_deltas_vs_id338"].get("extreme_rate_05_95", 0.0) > 0.005:
        flags.append("extreme_rate_worse_gt_0.005")
    probe = row.get("probe_summary") or {}
    if probe.get("primary_wins_vs_id338", 0) < 4:
        flags.append("primary_wins_lt_4")
    if probe.get("mean_delta_vs_id338", 0.0) >= -0.00005:
        flags.append("mean_delta_not_below_gate")
    if probe.get("max_delta_vs_id338", 999.0) > 0.00060:
        flags.append("max_delta_above_gate")
    return flags


def build_report() -> dict[str, Any]:
    train_sorted = load_train_sorted()
    train_processed = pd.read_parquet(TRAIN_PATH)
    test_processed = pd.read_parquet(TEST_PATH)
    reference_key = test_processed[["subject_id", "sleep_date", "lifelog_date"]].copy()
    train_stats = train_reference_stats(train_processed)
    id338 = load_base_test_submission(338, reference_key)
    id337 = load_base_test_submission(337, reference_key)
    id340 = load_base_test_submission(340, reference_key)
    id178 = load_base_test_submission(178, reference_key)
    id287 = load_base_test_submission(287, reference_key)
    anchor_predictions = {
        178: id178[TARGETS].astype(float).to_numpy(),
        287: id287[TARGETS].astype(float).to_numpy(),
        337: id337[TARGETS].astype(float).to_numpy(),
        338: id338[TARGETS].astype(float).to_numpy(),
        340: id340[TARGETS].astype(float).to_numpy(),
    }
    features, feature_metadata = load_source_train_test(SOURCE_GROUP, train_sorted, test_processed)
    feature_cols = [col for col in features.columns if col.startswith(f"{SOURCE_GROUP}__")]
    y = train_sorted[TARGET].astype(int).to_numpy()
    source_pred = fit_predict_test_source(
        features=features,
        feature_cols=feature_cols,
        y=y,
        model_spec=MODEL_SPEC,
        n_train=len(train_sorted),
    )
    id338_metrics = metric_row_for_submission(338, id338, train_stats, anchor_predictions, public_lb=PUBLIC_ID338)
    probe_by_weight = probe_summary_by_weight()
    rows = []
    for weight in WEIGHTS:
        candidate = id338.copy()
        candidate[TARGET] = np.clip(
            (1.0 - weight) * id338[TARGET].astype(float).to_numpy() + weight * source_pred,
            EPS,
            1.0 - EPS,
        )
        delta = np.abs(candidate[TARGETS].astype(float).to_numpy() - id338[TARGETS].astype(float).to_numpy())
        metrics = metric_row_for_submission(900338, candidate, train_stats, anchor_predictions)
        joint_deltas = {
            key: safe_float(metrics.get(key)) - safe_float(id338_metrics.get(key))
            for key in [
                "target_mean_l1_to_train",
                "target_mean_max_abs_to_train",
                "pair11_l1_to_train",
                "temporal_volatility_ratio_to_train",
                "prediction_entropy",
                "prediction_sharpness",
                "extreme_rate_05_95",
                "extreme_rate_01_99",
                "mean_abs_delta_vs_id338",
                "max_abs_delta_vs_id338",
            ]
            if safe_float(metrics.get(key)) is not None and safe_float(id338_metrics.get(key)) is not None
        }
        row = {
            "target": TARGET,
            "candidate": candidate_name(weight),
            "source_group": SOURCE_GROUP,
            "model_spec": MODEL_SPEC,
            "weight": weight,
            "probe_summary": probe_by_weight.get(weight),
            "source_prediction_mean": float(np.mean(source_pred)),
            "source_prediction_min": float(np.min(source_pred)),
            "source_prediction_max": float(np.max(source_pred)),
            "target_mean_shift_vs_id338": float(candidate[TARGET].mean() - id338[TARGET].mean()),
            "mean_abs_target_vs_id338": float(delta[:, TARGETS.index(TARGET)].mean()),
            "max_abs_target_vs_id338": float(delta[:, TARGETS.index(TARGET)].max()),
            "mean_abs_all_targets_vs_id338": float(delta.mean()),
            "max_abs_all_targets_vs_id338": float(delta.max()),
            "joint_deltas_vs_id338": joint_deltas,
        }
        row["risk_flags"] = risk_flags_for(row)
        row["geometry_pass"] = len([flag for flag in row["risk_flags"] if not flag.startswith("mean_delta")]) == 0
        rows.append(row)
    reviewable = [
        row
        for row in rows
        if row["probe_summary"]
        and row["probe_summary"].get("primary_wins_vs_id338", 0) >= 4
        and row["probe_summary"].get("mean_delta_vs_id338", 0.0) < -0.00005
        and row["probe_summary"].get("max_delta_vs_id338", 999.0) <= 0.00060
        and not row["risk_flags"]
    ]
    selected = sorted(
        reviewable,
        key=lambda row: (
            row["probe_summary"]["mean_delta_vs_id338"],
            -row["weight"],
            row["mean_abs_all_targets_vs_id338"],
        ),
    )[0] if reviewable else None
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id338_s3_phone_wearable_weight_geometry_scan",
        "purpose": "Compare ID338 S3-only phone/wearable overlay weights before any bolder reserve artifact promotion.",
        "inputs": {
            "probe_report": rel(PROBE_REPORT),
            "source_group": SOURCE_GROUP,
            "model_spec": MODEL_SPEC,
            "feature_metadata": feature_metadata,
        },
        "rows": rows,
        "selected_config": selected,
        "decision": {
            "scan_completed": True,
            "selected_weight": selected["weight"] if selected else None,
            "selected_candidate": selected["candidate"] if selected else None,
            "public_058_evidence": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": (
                "promotion_contract_review_for_bolder_id338_s3_micro_overlay"
                if selected
                else "keep_existing_id343_only"
            ),
            "next_action": (
                "Selected weight is still no-submit; create an artifact only through a separate promotion script and audit."
                if selected
                else "No bolder weight clears the geometry/probe gate; keep ID343 as the only S3 reserve."
            ),
        },
        "no_full_test_predictions_written": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_made": True,
        "no_kaggle_run_made": True,
        "no_dacon_submission_made": True,
    }


def fmt(value: Any, digits: int = 6) -> str:
    value = safe_float(value)
    return "" if value is None else f"{value:.{digits}f}"


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# ID338 S3 phone/wearable weight geometry scan",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It writes no full-test predictions, submission CSV, base artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.",
        "",
        "## Weight Rows",
        "",
        "| weight | candidate | wins | mean Δ338 | max Δ338 | mean move | max move | mean all move | flags |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        probe = row.get("probe_summary") or {}
        flags = ",".join(row["risk_flags"])
        lines.append(
            f"| {row['weight']:.4f} | `{row['candidate']}` | {probe.get('primary_wins_vs_id338', '')} | "
            f"{fmt(probe.get('mean_delta_vs_id338'))} | {fmt(probe.get('max_delta_vs_id338'))} | "
            f"{fmt(row['mean_abs_target_vs_id338'])} | {fmt(row['max_abs_target_vs_id338'])} | "
            f"{fmt(row['mean_abs_all_targets_vs_id338'])} | `{flags}` |"
        )
    selected = report["selected_config"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Scan completed: `{decision['scan_completed']}`",
            f"- Selected candidate: `{decision['selected_candidate']}`",
            f"- Selected weight: `{decision['selected_weight']}`",
            f"- Public `0.58` evidence: `{decision['public_058_evidence']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Next local allowed step: `{decision['next_local_allowed_step']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )
    if selected:
        lines.extend(
            [
                "## Selected Config",
                "",
                "```json",
                json.dumps(selected, indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
