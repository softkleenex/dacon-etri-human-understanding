#!/usr/bin/env python3
"""No-artifact geometry gate for the ID338 material Q3/S3 probe."""

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

from scripts.diagnose_post_id338_material_q3s3_probe import (  # noqa: E402
    ANCHOR_ID,
    SOURCE_MATRICES,
    TARGETS,
    rel,
)
from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402
    EPS,
    find_oof,
    fold_matrix,
    load_train_sorted,
    make_model,
)
from scripts.diagnose_public_joint_distribution_risk_post_id341 import (  # noqa: E402
    candidate_metrics,
    train_reference_stats,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
BASE_DIR = ROOT / "artifacts" / "base_predictions"
PROBE_REPORT = REPORTS_DIR / "post_id338_material_q3s3_probe_20260623.json"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id338_material_q3s3_geometry_gate_20260623.json"
TRAIN_PATH = ROOT / "data" / "processed" / "df_train_raw.parquet"
TEST_PATH = ROOT / "data" / "processed" / "df_test_raw.parquet"

PUBLIC_ID338 = 0.5960832519
PUBLIC_ID337 = 0.6006206286219425
PUBLIC_ID340 = 0.601379175253277


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def find_base_test(candidate_id: int) -> Path:
    paths = sorted(BASE_DIR.glob(f"base_test_id_{candidate_id}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing base test ID{candidate_id}")
    return paths[0]


def load_base_test_submission(candidate_id: int, reference_key: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(find_base_test(candidate_id))
    pred = pred.copy()
    pred["lifelog_date"] = pd.to_datetime(pred["date"]).dt.date
    ref = reference_key.copy()
    ref["lifelog_date"] = pd.to_datetime(ref["lifelog_date"]).dt.date
    ref["sleep_date"] = pd.to_datetime(ref["sleep_date"]).dt.date
    merged = ref.merge(
        pred.drop(columns=["date"]),
        on=["subject_id", "lifelog_date"],
        how="left",
        validate="one_to_one",
    )
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"Base test ID{candidate_id} does not align with reference test rows.")
    return merged[["subject_id", "sleep_date", "lifelog_date", *TARGETS]]


def load_oof_train(candidate_id: int, train: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(find_oof(candidate_id))
    pred = pred.copy()
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    aligned = train[["subject_id", "date"]].merge(pred, on=["subject_id", "date"], how="left", validate="one_to_one")
    if aligned[TARGETS].isna().any().any():
        raise ValueError(f"OOF ID{candidate_id} does not align with train rows.")
    return aligned[["subject_id", "date", *TARGETS]]


def load_source_train_test(source_group: str, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = SOURCE_MATRICES[source_group]
    frame = pd.read_parquet(path).copy()
    frame["date"] = pd.to_datetime(frame["lifelog_date"]).dt.date
    feature_cols = [
        col
        for col in frame.columns
        if col not in {"subject_id", "sleep_date", "lifelog_date", "date", "split"}
        and pd.api.types.is_numeric_dtype(frame[col])
    ]
    renamed = frame[["subject_id", "date", "split", *feature_cols]].rename(
        columns={col: f"{source_group}__{col}" for col in feature_cols}
    )
    prefixed = [f"{source_group}__{col}" for col in feature_cols]
    train_key = train[["subject_id", "date"]].copy()
    test_key = test[["subject_id", "lifelog_date"]].copy()
    test_key["date"] = pd.to_datetime(test_key["lifelog_date"]).dt.date
    test_key = test_key[["subject_id", "date"]]
    train_aligned = train_key.merge(
        renamed[renamed["split"].eq("train")].drop(columns=["split"]),
        on=["subject_id", "date"],
        how="left",
        validate="one_to_one",
    )
    test_aligned = test_key.merge(
        renamed[renamed["split"].eq("test")].drop(columns=["split"]),
        on=["subject_id", "date"],
        how="left",
        validate="one_to_one",
    )
    train_aligned["_row_split"] = "train"
    test_aligned["_row_split"] = "test"
    combined = pd.concat([train_aligned, test_aligned], axis=0, ignore_index=True)
    metadata = {
        "source_group": source_group,
        "path": rel(path),
        "feature_columns": len(prefixed),
        "train_rows": int(len(train_aligned)),
        "test_rows": int(len(test_aligned)),
        "train_missing_rate": float(train_aligned[prefixed].isna().mean().mean()),
        "test_missing_rate": float(test_aligned[prefixed].isna().mean().mean()),
        "all_train_rows_aligned": bool(train_aligned[prefixed].notna().any(axis=1).all()),
        "all_test_rows_aligned": bool(test_aligned[prefixed].notna().any(axis=1).all()),
    }
    return combined, metadata


def fit_predict_test_source(
    features: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    model_spec: dict[str, Any],
    n_train: int,
) -> np.ndarray:
    fit_idx = np.arange(n_train)
    test_idx = np.arange(n_train, len(features))
    fit_x, test_x = fold_matrix(features, feature_cols, fit_idx, test_idx)
    model = make_model(type("Spec", (), model_spec)())
    model.fit(fit_x, y)
    return np.clip(model.predict_proba(test_x)[:, 1], EPS, 1 - EPS)


def metric_row_for_submission(
    candidate_id: int,
    submission: pd.DataFrame,
    train_stats: dict[str, Any],
    anchors: dict[int, np.ndarray],
    local_cv: float | None = None,
    public_lb: float | None = None,
) -> dict[str, Any]:
    row = pd.Series(
        {
            "id_num": candidate_id,
            "submission_file": f"in_memory_id{candidate_id}",
            "local_cv_score_num": local_cv,
            "public_lb_score_num": public_lb,
            "status_clean": "in_memory_geometry_reference",
        }
    )
    return candidate_metrics(row, submission, train_stats, anchors)


def build_report(probe_report_path: Path = PROBE_REPORT) -> dict[str, Any]:
    probe = json.loads(probe_report_path.read_text(encoding="utf-8"))
    selected = probe.get("selected_target_configs", {})
    if not selected:
        raise ValueError("No selected target configs in probe report.")

    train_sorted = load_train_sorted()
    train_processed = pd.read_parquet(TRAIN_PATH)
    test_processed = pd.read_parquet(TEST_PATH)
    reference_key = test_processed[["subject_id", "sleep_date", "lifelog_date"]].copy()
    train_stats = train_reference_stats(train_processed)
    id338_test = load_base_test_submission(338, reference_key)
    id337_test = load_base_test_submission(337, reference_key)
    id340_test = load_base_test_submission(340, reference_key)
    id178_test = load_base_test_submission(178, reference_key)
    id287_test = load_base_test_submission(287, reference_key)
    anchor_predictions = {
        178: id178_test[TARGETS].astype(float).to_numpy(),
        287: id287_test[TARGETS].astype(float).to_numpy(),
        337: id337_test[TARGETS].astype(float).to_numpy(),
        338: id338_test[TARGETS].astype(float).to_numpy(),
        340: id340_test[TARGETS].astype(float).to_numpy(),
    }

    candidate = id338_test.copy()
    target_reports: dict[str, Any] = {}
    for target, config in selected.items():
        source_group = config["source_group"]
        features, feature_metadata = load_source_train_test(source_group, train_sorted, test_processed)
        feature_cols = [col for col in features.columns if col.startswith(f"{source_group}__")]
        y = train_sorted[target].astype(int).to_numpy()
        source_pred = fit_predict_test_source(
            features=features,
            feature_cols=feature_cols,
            y=y,
            model_spec=config["model_spec"],
            n_train=len(train_sorted),
        )
        weight = float(config["weight"])
        target_idx = TARGETS.index(target)
        before = candidate[target].astype(float).to_numpy()
        candidate[target] = np.clip((1.0 - weight) * before + weight * source_pred, EPS, 1 - EPS)
        movement = np.abs(candidate[target].astype(float).to_numpy() - id338_test[target].astype(float).to_numpy())
        target_reports[target] = {
            "config": config,
            "feature_metadata": feature_metadata,
            "source_prediction_mean": float(np.mean(source_pred)),
            "source_prediction_min": float(np.min(source_pred)),
            "source_prediction_max": float(np.max(source_pred)),
            "id338_target_mean": float(id338_test[target].mean()),
            "candidate_target_mean": float(candidate[target].mean()),
            "target_mean_shift": float(candidate[target].mean() - id338_test[target].mean()),
            "mean_abs_movement_vs_id338": float(movement.mean()),
            "max_abs_movement_vs_id338": float(movement.max()),
            "target_index": target_idx,
        }

    candidate_arr = candidate[TARGETS].astype(float).to_numpy()
    id338_arr = id338_test[TARGETS].astype(float).to_numpy()
    delta = np.abs(candidate_arr - id338_arr)
    metrics = {
        "id338": metric_row_for_submission(338, id338_test, train_stats, anchor_predictions, public_lb=PUBLIC_ID338),
        "id337": metric_row_for_submission(337, id337_test, train_stats, anchor_predictions, public_lb=PUBLIC_ID337),
        "id340": metric_row_for_submission(340, id340_test, train_stats, anchor_predictions, public_lb=PUBLIC_ID340),
        "candidate_in_memory": metric_row_for_submission(900338, candidate, train_stats, anchor_predictions),
    }
    candidate_metrics_row = metrics["candidate_in_memory"]
    id338_metrics_row = metrics["id338"]
    joint_deltas_vs_id338 = {
        key: safe_float(candidate_metrics_row.get(key)) - safe_float(id338_metrics_row.get(key))
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
        if safe_float(candidate_metrics_row.get(key)) is not None and safe_float(id338_metrics_row.get(key)) is not None
    }
    risk_flags = []
    if float(delta.mean()) > 0.001:
        risk_flags.append("mean_delta_vs_id338_gt_0.001")
    if float(delta.max()) > 0.020:
        risk_flags.append("max_delta_vs_id338_gt_0.020")
    if joint_deltas_vs_id338.get("target_mean_l1_to_train", 0.0) > 0.001:
        risk_flags.append("target_mean_l1_worse_gt_0.001")
    if joint_deltas_vs_id338.get("pair11_l1_to_train", 0.0) > 0.001:
        risk_flags.append("pair11_l1_worse_gt_0.001")
    if joint_deltas_vs_id338.get("extreme_rate_05_95", 0.0) > 0.005:
        risk_flags.append("extreme_rate_worse_gt_0.005")

    gate_pass = not risk_flags and bool(target_reports)
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id338_material_q3s3_geometry_gate",
        "purpose": (
            "Reconstruct the selected ID338 material Q3/S3 overlay in memory on test rows and compare its "
            "distribution geometry against ID338. No artifact is written."
        ),
        "probe_report": rel(probe_report_path),
        "selected_targets": sorted(target_reports),
        "target_reports": target_reports,
        "movement_vs_id338": {
            "mean_abs_all_targets": float(delta.mean()),
            "max_abs_all_targets": float(delta.max()),
            "target_mean_abs": {
                target: float(delta[:, idx].mean()) for idx, target in enumerate(TARGETS) if float(delta[:, idx].mean()) > 0
            },
            "target_max_abs": {
                target: float(delta[:, idx].max()) for idx, target in enumerate(TARGETS) if float(delta[:, idx].max()) > 0
            },
        },
        "joint_metrics": metrics,
        "joint_deltas_vs_id338": joint_deltas_vs_id338,
        "risk_flags": risk_flags,
        "decision": {
            "geometry_gate_completed": True,
            "geometry_gate_pass": gate_pass,
            "public_058_evidence": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": (
                "promotion_contract_review_for_id338_s3_micro_overlay"
                if gate_pass
                else "close_id338_material_q3s3_geometry_gate"
            ),
            "next_action": (
                "Gate is distribution-safe but too small for automatic promotion. A separate promotion contract "
                "would be required before creating CSV/base/notebook/log artifacts."
                if gate_pass
                else "Do not promote the selected overlay; geometry flags require a different source or smaller move."
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
    move = report["movement_vs_id338"]
    lines = [
        "# ID338 material Q3/S3 geometry gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It reconstructs the selected overlay in memory and writes no full-test predictions, submission CSV, base artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.",
        "",
        "## Movement vs ID338",
        "",
        f"- Mean absolute movement, all targets: `{move['mean_abs_all_targets']:.8f}`",
        f"- Max absolute movement, all targets: `{move['max_abs_all_targets']:.8f}`",
        "",
        "| target | mean abs movement | max abs movement | mean shift | source | candidate |",
        "|---|---:|---:|---:|---|---|",
    ]
    for target, row in report["target_reports"].items():
        config = row["config"]
        lines.append(
            f"| `{target}` | {row['mean_abs_movement_vs_id338']:.8f} | "
            f"{row['max_abs_movement_vs_id338']:.8f} | {row['target_mean_shift']:+.8f} | "
            f"`{config['source_group']}` | `{config['candidate']}` |"
        )
    lines.extend(
        [
            "",
            "## Joint Deltas vs ID338",
            "",
            "| metric | delta |",
            "|---|---:|",
        ]
    )
    for key, value in report["joint_deltas_vs_id338"].items():
        lines.append(f"| `{key}` | {value:+.8f} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Geometry gate completed: `{decision['geometry_gate_completed']}`",
            f"- Geometry gate pass: `{decision['geometry_gate_pass']}`",
            f"- Risk flags: `{report['risk_flags']}`",
            f"- Public `0.58` evidence: `{decision['public_058_evidence']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Next local allowed step: `{decision['next_local_allowed_step']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-report", type=Path, default=PROBE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    probe_report = args.probe_report if args.probe_report.is_absolute() else ROOT / args.probe_report
    report = build_report(probe_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
