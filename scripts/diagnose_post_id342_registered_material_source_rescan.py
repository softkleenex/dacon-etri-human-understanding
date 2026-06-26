#!/usr/bin/env python3
"""Registered material-source rescan against ID342.

This report-only diagnostic revisits the registered post-ID341 feature matrices
after ID342 became the public anchor. It asks whether any independent raw-source
matrix has a fold-internal target-level signal strong enough to justify a later
artifact/Kaggle review.

It creates no full-test predictions, submission CSV, base-prediction artifact,
notebook-history artifact, submission-log row, Kaggle push/run, or DACON
submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402
    BLEND_WEIGHTS,
    MODEL_SPECS,
    PRIMARY_GATE_SPLITS,
    TARGETS,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
    predict_target_for_split,
    target_loss_on_split,
)
from scripts.diagnose_post_id342_feature_similarity_source_decomposition import (  # noqa: E402
    ID342_PUBLIC,
    PUBLIC_055,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_registered_material_source_rescan_20260624.json"
BASE_ANCHOR_ID = 342
REFERENCE_IDS = {
    "id178": 178,
    "id221": 221,
    "id342": 342,
    "id287": 287,
    "id329": 329,
}

SOURCE_MATRICES: dict[str, Path] = {
    "circadian_hr_light_phase": ROOT / "data/processed/post_id341_circadian_hr_light_phase_features_v1.parquet",
    "gps_path_topology": ROOT / "data/processed/post_id341_gps_path_topology_features_v1.parquet",
    "phone_wearable_discordance": ROOT / "data/processed/post_id341_phone_wearable_discordance_features_v1.parquet",
    "proximity_ambience_cooccurrence": ROOT
    / "data/processed/post_id341_proximity_ambience_cooccurrence_features_v1.parquet",
    "usage_semantic_burst_markov": ROOT / "data/processed/post_id341_usage_semantic_burst_markov_features_v1.parquet",
}

TARGET_FOCUS = ["Q1", "Q2", "Q3", "S2", "S3", "S4"]
FORBIDDEN_COLUMN_FRAGMENTS = ("target", "oof", "pred", "prediction", "submission", "public", "leaderboard")
KEY_COLUMNS = {"subject_id", "date", "lifelog_date", "sleep_date", "split"}

MIN_PRIMARY_WINS_VS_ID342 = 4
MAX_PRIMARY_MEAN_DELTA_VS_ID342 = -0.00005
MAX_PRIMARY_WORST_DELTA_VS_ID342 = 0.00025
MAX_PUBLIC_PROXY_FOR_REVIEW = 0.5920
MAX_GAP_TO_PUBLIC_055_FOR_REVIEW = 0.0400

RESEARCH_PRIORS = [
    {
        "source": "ETRI Lifelog Dataset 2024 paper",
        "url": "https://arxiv.org/abs/2508.03698",
        "read": (
            "Q1/S3 were strongest with 3-hour step-count features, Q2/Q3 with daily mobile app usage, and "
            "demographics/weekday helped across models."
        ),
    },
    {
        "source": "DACON 4th ETRI data page",
        "url": "https://dacon.io/en/competitions/official/236468/data",
        "read": (
            "The challenge exposes multimodal smartphone, wearable, proximity, GPS, light, app, and sleep-related "
            "lifelog streams for the seven Q/S targets."
        ),
    },
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def binary_loss(y_true: np.ndarray, values: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(values, 1e-6, 1.0 - 1e-6), labels=[0, 1]))


def load_aligned_matrix(source_name: str, matrix_path: Path, train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix = pd.read_parquet(matrix_path)
    if "split" not in matrix.columns or "lifelog_date" not in matrix.columns:
        raise ValueError(f"{matrix_path} must contain split and lifelog_date columns.")
    train_matrix = matrix[matrix["split"].eq("train")].copy()
    train_matrix["date"] = pd.to_datetime(train_matrix["lifelog_date"]).dt.date
    feature_cols = [
        column
        for column in train_matrix.columns
        if column not in KEY_COLUMNS
        and pd.api.types.is_numeric_dtype(train_matrix[column])
        and train_matrix[column].notna().any()
        and not any(fragment in column.lower() for fragment in FORBIDDEN_COLUMN_FRAGMENTS)
    ]
    aligned = train[["subject_id", "date"]].merge(
        train_matrix[["subject_id", "date", *feature_cols]],
        on=["subject_id", "date"],
        how="left",
        validate="one_to_one",
    )
    if len(aligned) != len(train):
        raise ValueError(f"{source_name} matrix does not align to sorted train rows.")
    metadata = {
        "source_name": source_name,
        "matrix_path": rel(matrix_path),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "split_counts": {str(key): int(value) for key, value in matrix["split"].value_counts().to_dict().items()},
        "aligned_train_rows": int(len(aligned)),
        "feature_columns": int(len(feature_cols)),
        "missing_rate": float(aligned[feature_cols].isna().mean().mean()) if feature_cols else None,
        "feature_column_sample": feature_cols[:12],
        "forbidden_feature_columns": sorted(
            column for column in train_matrix.columns if any(fragment in column.lower() for fragment in FORBIDDEN_COLUMN_FRAGMENTS)
        ),
    }
    return aligned, metadata


def scan_split_rows_for_source(
    train: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    references: dict[str, np.ndarray],
    source_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGET_FOCUS:
        target_idx = TARGETS.index(target)
        y = train[target].astype(int).to_numpy()
        for spec in MODEL_SPECS:
            for split_name in PRIMARY_GATE_SPLITS:
                folds = splits[split_name]
                source_pred, covered = predict_target_for_split(features, feature_cols, y, folds, spec)
                id342_target_loss = target_loss_on_split(train, folds, target, references["id342"])
                source_values = references["id342"].copy()
                source_values[:, target_idx] = np.where(np.isnan(source_pred), references["id342"][:, target_idx], source_pred)
                source_loss = target_loss_on_split(train, folds, target, source_values)
                rows.append(
                    {
                        "source": source_name,
                        "target": target,
                        "split_name": split_name,
                        "candidate": f"{source_name}_{target.lower()}_{spec.name}_replace",
                        "model_spec": spec.name,
                        "weight": 1.0,
                        "valid_rows": int(covered),
                        "target_loss": source_loss,
                        "delta_target_vs_id342": float(source_loss - id342_target_loss),
                        "delta_all_vs_id342": float((source_loss - id342_target_loss) / len(TARGETS)),
                    }
                )
                for weight in BLEND_WEIGHTS:
                    blended = references["id342"].copy()
                    blended[:, target_idx] = np.clip(
                        (1.0 - weight) * references["id342"][:, target_idx]
                        + weight * source_values[:, target_idx],
                        1e-6,
                        1.0 - 1e-6,
                    )
                    loss = target_loss_on_split(train, folds, target, blended)
                    rows.append(
                        {
                            "source": source_name,
                            "target": target,
                            "split_name": split_name,
                            "candidate": f"{source_name}_{target.lower()}_{spec.name}_w{weight:g}",
                            "model_spec": spec.name,
                            "weight": weight,
                            "valid_rows": int(covered),
                            "target_loss": loss,
                            "delta_target_vs_id342": float(loss - id342_target_loss),
                            "delta_all_vs_id342": float((loss - id342_target_loss) / len(TARGETS)),
                        }
                    )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["candidate"]].append(row)
    summary = []
    for candidate, candidate_rows in grouped.items():
        split_rows = {row["split_name"]: row for row in candidate_rows}
        primary_rows = [split_rows[name] for name in PRIMARY_GATE_SPLITS if name in split_rows]
        deltas = [row["delta_all_vs_id342"] for row in primary_rows]
        target_deltas = [row["delta_target_vs_id342"] for row in primary_rows]
        first = candidate_rows[0]
        mean_delta = float(np.mean(deltas)) if deltas else 0.0
        public_proxy = float(ID342_PUBLIC + mean_delta)
        checks = {
            "primary_wins_vs_id342": int(sum(delta < 0 for delta in deltas)) >= MIN_PRIMARY_WINS_VS_ID342,
            "primary_mean_delta_material": mean_delta <= MAX_PRIMARY_MEAN_DELTA_VS_ID342,
            "primary_worst_delta_within_ceiling": (float(np.max(deltas)) if deltas else 0.0)
            <= MAX_PRIMARY_WORST_DELTA_VS_ID342,
            "public_proxy_low_enough": public_proxy <= MAX_PUBLIC_PROXY_FOR_REVIEW,
            "public_055_gap_low_enough": (public_proxy - PUBLIC_055) <= MAX_GAP_TO_PUBLIC_055_FOR_REVIEW,
        }
        summary.append(
            {
                "candidate": candidate,
                "source": first["source"],
                "target": first["target"],
                "model_spec": first["model_spec"],
                "weight": first["weight"],
                "primary_wins_vs_id342": int(sum(delta < 0 for delta in deltas)),
                "primary_mean_delta_all_vs_id342": mean_delta,
                "primary_max_delta_all_vs_id342": float(np.max(deltas)) if deltas else 0.0,
                "primary_mean_delta_target_vs_id342": float(np.mean(target_deltas)) if target_deltas else 0.0,
                "primary_max_delta_target_vs_id342": float(np.max(target_deltas)) if target_deltas else 0.0,
                "projected_public_proxy": public_proxy,
                "gap_to_public_055": float(public_proxy - PUBLIC_055),
                "checks": checks,
                "local_gate_pass": bool(all(checks.values())),
                "split_rows": primary_rows,
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            not row["local_gate_pass"],
            row["primary_mean_delta_all_vs_id342"],
            row["primary_max_delta_all_vs_id342"],
        ),
    )


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    splits = load_split_manifest()
    references = {name: load_reference_values(train, candidate_id) for name, candidate_id in REFERENCE_IDS.items()}
    all_rows: list[dict[str, Any]] = []
    source_metadata = {}
    for source_name, matrix_path in SOURCE_MATRICES.items():
        aligned, metadata = load_aligned_matrix(source_name, matrix_path, train)
        feature_cols = [
            column
            for column in aligned.columns
            if column not in {"subject_id", "date"}
            and pd.api.types.is_numeric_dtype(aligned[column])
            and aligned[column].notna().any()
        ]
        source_metadata[source_name] = metadata | {"used_feature_columns": feature_cols}
        all_rows.extend(
            scan_split_rows_for_source(
                train=train,
                features=aligned,
                feature_cols=feature_cols,
                splits=splits,
                references=references,
                source_name=source_name,
            )
        )
    summary = summarize_rows(all_rows)
    passing = [row for row in summary if row["local_gate_pass"]]
    best = summary[0] if summary else None
    by_source = {}
    for source_name in SOURCE_MATRICES:
        source_rows = [row for row in summary if row["source"] == source_name]
        by_source[source_name] = source_rows[:5]
    decision = {
        "reviewed_split_rows": int(len(all_rows)),
        "reviewed_candidates": int(len(summary)),
        "local_gate_pass_count": int(len(passing)),
        "best_candidate": best["candidate"] if best else None,
        "best_source": best["source"] if best else None,
        "best_target": best["target"] if best else None,
        "best_primary_mean_delta_all_vs_id342": best["primary_mean_delta_all_vs_id342"] if best else None,
        "best_projected_public_proxy": best["projected_public_proxy"] if best else ID342_PUBLIC,
        "best_gap_to_public_055": best["gap_to_public_055"] if best else ID342_PUBLIC - PUBLIC_055,
        "artifact_review_allowed": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "create_submission_artifact": False,
        "submit_now": False,
        "reason": (
            "At least one row passed the mechanical local gate, but this report is discovery-only; require a stricter "
            "artifact/test-geometry gate before Kaggle or DACON action."
            if passing
            else "No registered material-source row cleared the strict ID342/Public-0.55 local gate."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_registered_material_source_rescan",
        "purpose": "Rescan registered post-ID341 material feature matrices against the ID342 public anchor.",
        "research_priors": RESEARCH_PRIORS,
        "base_anchor_id": BASE_ANCHOR_ID,
        "target_focus": TARGET_FOCUS,
        "model_specs": [spec.name for spec in MODEL_SPECS],
        "blend_weights": BLEND_WEIGHTS,
        "primary_gate_splits": PRIMARY_GATE_SPLITS,
        "thresholds": {
            "min_primary_wins_vs_id342": MIN_PRIMARY_WINS_VS_ID342,
            "max_primary_mean_delta_vs_id342": MAX_PRIMARY_MEAN_DELTA_VS_ID342,
            "max_primary_worst_delta_vs_id342": MAX_PRIMARY_WORST_DELTA_VS_ID342,
            "max_public_proxy_for_review": MAX_PUBLIC_PROXY_FOR_REVIEW,
            "max_gap_to_public_055_for_review": MAX_GAP_TO_PUBLIC_055_FOR_REVIEW,
        },
        "source_metadata": source_metadata,
        "top_rows": summary[:40],
        "top_rows_by_source": by_source,
        "decision": decision,
        "no_prediction_file_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Post-ID342 registered material-source rescan",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.",
        "",
        "## Research Priors",
        "",
    ]
    for prior in report["research_priors"]:
        lines.append(f"- [{prior['source']}]({prior['url']}): {prior['read']}")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Reviewed candidates: `{decision['reviewed_candidates']}`",
            f"- Local gate pass count: `{decision['local_gate_pass_count']}`",
            f"- Best candidate: `{decision['best_candidate']}`",
            f"- Best source/target: `{decision['best_source']}` / `{decision['best_target']}`",
            f"- Best mean all-target delta vs ID342: `{decision['best_primary_mean_delta_all_vs_id342']:+.9f}`",
            f"- Best projected public proxy: `{decision['best_projected_public_proxy']:.9f}`",
            f"- Best gap to Public 0.55: `{decision['best_gap_to_public_055']:+.9f}`",
            f"- Artifact review allowed: `{decision['artifact_review_allowed']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Reason: {decision['reason']}",
            "",
            "## Top Rows",
            "",
            "| rank | pass | candidate | source | target | weight | wins | mean all Δ342 | max all Δ342 | target mean Δ342 | proxy | gap 0.55 |",
            "|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(report["top_rows"][:25], start=1):
        lines.append(
            "| {rank} | `{passed}` | `{candidate}` | `{source}` | `{target}` | {weight} | {wins} | {mean:+.6f} | {max_delta:+.6f} | {target_mean:+.6f} | {proxy:.6f} | {gap:+.6f} |".format(
                rank=rank,
                passed=row["local_gate_pass"],
                candidate=row["candidate"],
                source=row["source"],
                target=row["target"],
                weight=row["weight"],
                wins=row["primary_wins_vs_id342"],
                mean=row["primary_mean_delta_all_vs_id342"],
                max_delta=row["primary_max_delta_all_vs_id342"],
                target_mean=row["primary_mean_delta_target_vs_id342"],
                proxy=row["projected_public_proxy"],
                gap=row["gap_to_public_055"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    print(f"Wrote {rel(args.output)} and {rel(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
