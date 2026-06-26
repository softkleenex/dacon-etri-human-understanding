#!/usr/bin/env python3
"""Fold-nested Q3/S3 sibling-target bridge smoke after ID341.

The target anatomy report found Q3 is materially related to Q2 and S3 is
materially related to S2. Directly using same-row sibling labels would leak.
This diagnostic therefore builds sibling predictions inside each validation
fold, using inner out-of-fold sibling predictions for the target-model training
rows and a sibling model fitted only on the outer training rows for validation
rows.

It is report-only: no full-test predictions, submission CSV, base prediction
artifact, notebook/history artifact, submission-log row, Kaggle push/run, or
DACON submission is created.
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
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_q2q3s3_feature_generation_metadata_smoke import (  # noqa: E402
    DEFAULT_OUTPUT as FEATURE_METADATA_REPORT,
)
from scripts.diagnose_post_id341_q2q3s3_feature_generation_metadata_smoke import (  # noqa: E402
    feature_columns,
)
from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402
    ANCHOR_IDS,
    BLEND_WEIGHTS,
    CONFIG_MAX_DELTA_CEILING,
    CONFIG_MEAN_DELTA_CEILING,
    CONFIG_WIN_FLOOR,
    EPS,
    MODEL_SPECS,
    PRIMARY_GATE_SPLITS,
    TARGETS,
    ModelSpec,
    aligned_feature_frame,
    find_oof,
    fold_matrix,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
    make_model,
    rel,
    target_loss_on_split,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q3s3_fold_nested_sibling_bridge_smoke_20260619.json"
SIBLING_DESIGN = REPORTS_DIR / "post_id341_q3s3_sibling_target_bridge_design_20260619.json"
NEGATIVE_MAP = REPORTS_DIR / "post_id341_q3s3_negative_evidence_map_20260618.json"

TARGET_FOCUS = ["Q3", "S3"]
SIBLING_TARGET = {"Q3": "Q2", "S3": "S2"}
PUBLIC_058 = 0.58
ID178_PUBLIC = 0.6014113009


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "parse_ok": False, "decision": {}, "path": rel(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"exists": True, "parse_ok": False, "decision": {}, "path": rel(path), "error": str(exc)}
    payload.setdefault("path", rel(path))
    payload["exists"] = True
    payload["parse_ok"] = True
    return payload


def fit_predict_binary(model, fit_x: pd.DataFrame, fit_y: np.ndarray, query_x: pd.DataFrame) -> np.ndarray:
    if len(np.unique(fit_y)) < 2:
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(np.zeros((len(fit_y), 1)), fit_y)
        return np.full(len(query_x), float(np.mean(fit_y)), dtype=float)
    model.fit(fit_x, fit_y)
    return np.clip(model.predict_proba(query_x)[:, 1], EPS, 1 - EPS)


def inner_folds_for_fit(
    features: pd.DataFrame, fit_idx: np.ndarray, y: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    subjects = features.iloc[fit_idx]["subject_id"].astype(str).to_numpy()
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) >= 3:
        n_splits = min(5, len(unique_subjects))
        splitter = GroupKFold(n_splits=n_splits)
        return [
            (fit_idx[inner_fit], fit_idx[inner_valid])
            for inner_fit, inner_valid in splitter.split(fit_idx, y, subjects)
        ]

    counts = np.bincount(y.astype(int), minlength=2)
    min_count = int(counts.min())
    if min_count >= 2:
        n_splits = min(5, min_count, len(fit_idx))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=341)
        return [(fit_idx[inner_fit], fit_idx[inner_valid]) for inner_fit, inner_valid in splitter.split(fit_idx, y)]

    midpoint = max(1, len(fit_idx) // 2)
    return [(fit_idx[:midpoint], fit_idx[midpoint:]), (fit_idx[midpoint:], fit_idx[:midpoint])]


def nested_sibling_predictions(
    features: pd.DataFrame,
    feature_cols: list[str],
    sibling_y: np.ndarray,
    outer_fit_idx: np.ndarray,
    outer_valid_idx: np.ndarray,
    spec: ModelSpec,
) -> tuple[np.ndarray, np.ndarray, int]:
    fit_sibling_pred = np.full(len(outer_fit_idx), np.nan, dtype=float)
    inner_folds = inner_folds_for_fit(features, outer_fit_idx, sibling_y[outer_fit_idx])
    for inner_fit_idx, inner_valid_idx in inner_folds:
        if len(inner_valid_idx) == 0:
            continue
        inner_fit_x, inner_valid_x = fold_matrix(features, feature_cols, inner_fit_idx, inner_valid_idx)
        inner_pred = fit_predict_binary(make_model(spec), inner_fit_x, sibling_y[inner_fit_idx], inner_valid_x)
        positions = {int(abs_idx): pos for pos, abs_idx in enumerate(outer_fit_idx)}
        for abs_idx, pred in zip(inner_valid_idx, inner_pred, strict=False):
            fit_sibling_pred[positions[int(abs_idx)]] = pred

    missing = np.isnan(fit_sibling_pred)
    if missing.any():
        fallback = float(np.mean(sibling_y[outer_fit_idx]))
        fit_sibling_pred[missing] = fallback

    outer_fit_x, outer_valid_x = fold_matrix(features, feature_cols, outer_fit_idx, outer_valid_idx)
    valid_sibling_pred = fit_predict_binary(make_model(spec), outer_fit_x, sibling_y[outer_fit_idx], outer_valid_x)
    return fit_sibling_pred, valid_sibling_pred, len(inner_folds)


def predict_target_with_nested_sibling(
    features: pd.DataFrame,
    feature_cols: list[str],
    target_y: np.ndarray,
    sibling_y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: ModelSpec,
) -> tuple[np.ndarray, int, int]:
    pred = np.full(len(target_y), np.nan, dtype=float)
    covered = 0
    inner_fold_total = 0
    for fit_idx, valid_idx in folds:
        fit_sibling_pred, valid_sibling_pred, inner_count = nested_sibling_predictions(
            features, feature_cols, sibling_y, fit_idx, valid_idx, spec
        )
        fit_x, valid_x = fold_matrix(features, feature_cols, fit_idx, valid_idx)
        fit_x = fit_x.copy()
        valid_x = valid_x.copy()
        fit_x["nested_sibling_pred"] = fit_sibling_pred
        valid_x["nested_sibling_pred"] = valid_sibling_pred
        pred[valid_idx] = fit_predict_binary(make_model(spec), fit_x, target_y[fit_idx], valid_x)
        covered += len(valid_idx)
        inner_fold_total += inner_count
    return pred, covered, inner_fold_total


def scan_rows_for_target(
    train: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    references: dict[str, np.ndarray],
    target: str,
    model_specs: tuple[ModelSpec, ...],
    split_names: list[str],
) -> list[dict[str, Any]]:
    target_idx = TARGETS.index(target)
    target_y = train[target].astype(int).to_numpy()
    sibling = SIBLING_TARGET[target]
    sibling_y = train[sibling].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for spec in model_specs:
        for split_name in split_names:
            folds = splits[split_name]
            pred, covered, inner_fold_total = predict_target_with_nested_sibling(
                features, feature_cols, target_y, sibling_y, folds, spec
            )
            id178_loss = target_loss_on_split(train, folds, target, references["id178"])
            id341_loss = target_loss_on_split(train, folds, target, references["id341"])
            source_values_by_anchor = {
                anchor_name: anchor_values.copy() for anchor_name, anchor_values in references.items()
            }
            for values in source_values_by_anchor.values():
                values[:, target_idx] = np.where(np.isnan(pred), values[:, target_idx], pred)
            source_loss = target_loss_on_split(train, folds, target, source_values_by_anchor["id341"])
            rows.append(
                {
                    "target": target,
                    "sibling_target": sibling,
                    "split_name": split_name,
                    "candidate": f"q3s3_nested_sibling_{target.lower()}_from_{sibling.lower()}_{spec.name}",
                    "model_spec": asdict(spec),
                    "anchor": None,
                    "weight": None,
                    "valid_rows": int(covered),
                    "inner_fold_total": int(inner_fold_total),
                    "loss": source_loss,
                    "id178_loss": id178_loss,
                    "id341_loss": id341_loss,
                    "delta_vs_id178": float(source_loss - id178_loss),
                    "delta_vs_id341": float(source_loss - id341_loss),
                }
            )
            for anchor_name, anchor_values in references.items():
                source_values = source_values_by_anchor[anchor_name]
                for weight in BLEND_WEIGHTS:
                    blended = anchor_values.copy()
                    blended[:, target_idx] = np.clip(
                        (1.0 - weight) * anchor_values[:, target_idx] + weight * source_values[:, target_idx],
                        EPS,
                        1 - EPS,
                    )
                    loss = target_loss_on_split(train, folds, target, blended)
                    anchor_loss = id178_loss if anchor_name == "id178" else id341_loss
                    rows.append(
                        {
                            "target": target,
                            "sibling_target": sibling,
                            "split_name": split_name,
                            "candidate": f"blend_{anchor_name}_nested_{target.lower()}_from_{sibling.lower()}_{spec.name}_w{weight:g}",
                            "model_spec": asdict(spec),
                            "anchor": anchor_name,
                            "weight": weight,
                            "valid_rows": int(covered),
                            "inner_fold_total": int(inner_fold_total),
                            "loss": loss,
                            "id178_loss": id178_loss,
                            "id341_loss": id341_loss,
                            "delta_vs_anchor": float(loss - anchor_loss),
                            "delta_vs_id178": float(loss - id178_loss),
                            "delta_vs_id341": float(loss - id341_loss),
                        }
                    )
    return rows


def summarize_candidate_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if not rows:
        return out
    frame = pd.DataFrame(rows)
    for target in TARGET_FOCUS:
        target_frame = frame[frame["target"].eq(target)].copy()
        summaries: list[dict[str, Any]] = []
        for candidate, group in target_frame.groupby("candidate", sort=False):
            deltas = group.set_index("split_name")["delta_vs_id341"].to_dict()
            losses = group.set_index("split_name")["loss"].to_dict()
            delta_values = [float(v) for v in deltas.values()]
            row0 = group.iloc[0].to_dict()
            summaries.append(
                {
                    "target": target,
                    "sibling_target": row0.get("sibling_target"),
                    "candidate": str(candidate),
                    "model_spec": row0.get("model_spec"),
                    "anchor": row0.get("anchor"),
                    "weight": row0.get("weight"),
                    "valid_rows": int(group["valid_rows"].sum()),
                    "inner_fold_total": int(group["inner_fold_total"].sum()),
                    "primary_wins_vs_id341": int(sum(delta < 0 for delta in delta_values)),
                    "mean_delta_vs_id341": float(np.mean(delta_values)),
                    "min_delta_vs_id341": float(np.min(delta_values)),
                    "max_delta_vs_id341": float(np.max(delta_values)),
                    "split_deltas_vs_id341": {str(k): float(v) for k, v in deltas.items()},
                    "split_losses": {str(k): float(v) for k, v in losses.items()},
                }
            )
        out[target] = sorted(
            summaries,
            key=lambda row: (
                -row["primary_wins_vs_id341"],
                row["mean_delta_vs_id341"],
                row["max_delta_vs_id341"],
                row["candidate"],
            ),
        )
    return out


def select_target_configs(summaries: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for target, rows in summaries.items():
        for row in rows:
            if (
                row["primary_wins_vs_id341"] >= CONFIG_WIN_FLOOR
                and row["mean_delta_vs_id341"] <= CONFIG_MEAN_DELTA_CEILING
                and row["max_delta_vs_id341"] <= CONFIG_MAX_DELTA_CEILING
            ):
                selected[target] = row
                break
    return selected


def build_report(
    model_specs: tuple[ModelSpec, ...] = MODEL_SPECS,
    split_names: list[str] | None = None,
    negative_evidence_path: Path = NEGATIVE_MAP,
) -> dict[str, Any]:
    split_names = split_names or PRIMARY_GATE_SPLITS
    negative_evidence_path = (
        negative_evidence_path if negative_evidence_path.is_absolute() else ROOT / negative_evidence_path
    )
    sibling_design = load_json(SIBLING_DESIGN)
    negative_map = load_json(negative_evidence_path)
    train = load_train_sorted()
    features, feature_metadata = aligned_feature_frame(train)
    cols = feature_columns(features)
    cols = [col for col in cols if pd.api.types.is_numeric_dtype(features[col]) and features[col].notna().any()]
    splits = load_split_manifest()
    references = {f"id{candidate_id}": load_reference_values(train, candidate_id) for candidate_id in ANCHOR_IDS}
    scan_rows: list[dict[str, Any]] = []
    for target in TARGET_FOCUS:
        scan_rows.extend(
            scan_rows_for_target(train, features, cols, splits, references, target, model_specs, split_names)
        )
    summaries = summarize_candidate_rows(scan_rows)
    selected = select_target_configs(summaries)
    selected_targets = sorted(selected)
    q3s3_covered = selected_targets == TARGET_FOCUS
    best_proxy = ID178_PUBLIC + min(
        [row["mean_delta_vs_id341"] for rows in summaries.values() for row in rows[:1]] or [0.0]
    )
    checks = {
        "sibling_design_ready": sibling_design.get("decision", {}).get("sibling_bridge_design_ready") is True,
        "negative_map_closed": negative_map.get("decision", {}).get("all_local_q3s3_lanes_closed") is True,
        "features_aligned": feature_metadata.get("aligned_train_rows") == 450,
        "anchors_loaded": all(find_oof(candidate_id).exists() for candidate_id in ANCHOR_IDS),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q3s3_fold_nested_sibling_bridge_smoke",
        "purpose": (
            "Evaluate leakage-controlled Q3<-Q2 and S3<-S2 sibling prediction bridges using inner OOF "
            "sibling predictions inside each frozen validation fold."
        ),
        "target_focus": TARGET_FOCUS,
        "sibling_targets": SIBLING_TARGET,
        "inputs": {
            "feature_metadata_report": rel(FEATURE_METADATA_REPORT),
            "sibling_design": rel(SIBLING_DESIGN),
            "negative_evidence_map": rel(negative_evidence_path),
            "id178_oof": rel(find_oof(178)),
            "id341_oof": rel(find_oof(341)),
        },
        "feature_metadata": feature_metadata,
        "config": {
            "model_specs": [asdict(spec) for spec in model_specs],
            "split_names": split_names,
            "blend_weights": BLEND_WEIGHTS,
            "selection_win_floor": CONFIG_WIN_FLOOR,
            "selection_mean_delta_ceiling": CONFIG_MEAN_DELTA_CEILING,
            "selection_max_delta_ceiling": CONFIG_MAX_DELTA_CEILING,
        },
        "checks": checks,
        "selected_target_configs": selected,
        "top_rows_by_target": {target: rows[:8] for target, rows in summaries.items()},
        "scan_row_count": int(len(scan_rows)),
        "decision": {
            "fold_nested_sibling_smoke_completed": True,
            "target_focus": TARGET_FOCUS,
            "selected_targets": selected_targets,
            "q3_signal_found": "Q3" in selected,
            "s3_signal_found": "S3" in selected,
            "all_q3s3_targets_covered": q3s3_covered,
            "best_proxy": float(best_proxy),
            "beats_id178_public_anchor_proxy": bool(best_proxy < ID178_PUBLIC),
            "public_058_evidence": bool(q3s3_covered and best_proxy <= PUBLIC_058),
            "local_artifact_allowed": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "If both Q3 and S3 are selected, feed this only into a no-artifact transfer/joint-risk gate. "
                "Otherwise keep sibling bridges as upstream design context and do not create ID342."
            ),
        },
        "no_full_test_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    feature_metadata = report["feature_metadata"]
    lines = [
        "# Post-ID341 Q3/S3 fold-nested sibling bridge smoke",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.",
        "",
        "## Leakage control",
        "",
        "- Q3 uses only fold-nested predictions of Q2, not Q2 labels from the same row.",
        "- S3 uses only fold-nested predictions of S2, not S2 labels from the same row.",
        "- Inner OOF sibling predictions are used for outer training rows; sibling models fitted on outer training rows predict outer validation rows.",
        "",
        "## Feature input",
        "",
        f"- Aligned train rows: `{feature_metadata['aligned_train_rows']}`",
        f"- Feature columns used: `{feature_metadata['feature_columns']}`",
        f"- Mean missing rate: `{feature_metadata['mean_missing_rate']:.4f}`",
        "",
        "## Selected target configs",
        "",
        "| target | sibling | candidate | wins | mean Δ341 | max Δ341 |",
        "|---|---|---|---:|---:|---:|",
    ]
    if not report["selected_target_configs"]:
        lines.append("| _(none)_ |  |  |  |  |  |")
    for target, row in report["selected_target_configs"].items():
        lines.append(
            f"| `{target}` | `{row['sibling_target']}` | `{row['candidate']}` | "
            f"{row['primary_wins_vs_id341']} | {row['mean_delta_vs_id341']:+.6f} | "
            f"{row['max_delta_vs_id341']:+.6f} |"
        )
    lines.extend(["", "## Top rows by target", ""])
    for target, rows in report["top_rows_by_target"].items():
        lines.extend(
            [
                f"### `{target}` from `{report['sibling_targets'][target]}`",
                "",
                "| rank | candidate | wins | mean Δ341 | max Δ341 |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(rows[:5], start=1):
            lines.append(
                f"| {rank} | `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
                f"{row['mean_delta_vs_id341']:+.6f} | {row['max_delta_vs_id341']:+.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            f"- Fold-nested sibling smoke completed: `{decision['fold_nested_sibling_smoke_completed']}`",
            f"- Selected targets: `{decision['selected_targets']}`",
            f"- Q3 signal found: `{decision['q3_signal_found']}`",
            f"- S3 signal found: `{decision['s3_signal_found']}`",
            f"- All Q3/S3 targets covered: `{decision['all_q3s3_targets_covered']}`",
            f"- Best proxy: `{decision['best_proxy']:.6f}`",
            f"- Beats ID178 public anchor proxy: `{decision['beats_id178_public_anchor_proxy']}`",
            f"- Public `0.58` evidence: `{decision['public_058_evidence']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Next action: {decision['next_action']}",
            "",
            "## Checks",
            "",
        ]
    )
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--negative-evidence-map",
        type=Path,
        default=NEGATIVE_MAP,
        help="Path to the Q3/S3 negative-evidence map used by this smoke.",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(negative_evidence_path=args.negative_evidence_map)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
