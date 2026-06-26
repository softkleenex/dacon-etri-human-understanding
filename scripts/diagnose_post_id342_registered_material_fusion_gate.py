#!/usr/bin/env python3
"""Registered material-source fusion gate against ID342.

This report-only diagnostic follows the individual registered material-source
rescan by concatenating all registered post-ID341 feature matrices. It tests
whether a lightweight multimodal fusion model has a stronger fold-internal
signal than any single source family.

It creates no full-test predictions, submission CSV, base-prediction artifact,
notebook-history artifact, submission-log row, Kaggle push/run, or DACON
submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402
    BLEND_WEIGHTS,
    MODEL_SPECS,
    PRIMARY_GATE_SPLITS,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
)
from scripts.diagnose_post_id342_feature_similarity_source_decomposition import (  # noqa: E402
    ID342_PUBLIC,
    PUBLIC_055,
)
from scripts.diagnose_post_id342_registered_material_source_rescan import (  # noqa: E402
    BASE_ANCHOR_ID,
    FORBIDDEN_COLUMN_FRAGMENTS,
    KEY_COLUMNS,
    MAX_GAP_TO_PUBLIC_055_FOR_REVIEW,
    MAX_PRIMARY_MEAN_DELTA_VS_ID342,
    MAX_PRIMARY_WORST_DELTA_VS_ID342,
    MAX_PUBLIC_PROXY_FOR_REVIEW,
    MIN_PRIMARY_WINS_VS_ID342,
    REFERENCE_IDS,
    RESEARCH_PRIORS,
    SOURCE_MATRICES,
    TARGET_FOCUS,
    rel,
    scan_split_rows_for_source,
    summarize_rows,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_registered_material_fusion_gate_20260624.json"


def load_prefixed_source_features(
    source_name: str, matrix_path: Path, train: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix = pd.read_parquet(matrix_path)
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
    renamed = {column: f"{source_name}__{column}" for column in feature_cols}
    aligned = train[["subject_id", "date"]].merge(
        train_matrix[["subject_id", "date", *feature_cols]].rename(columns=renamed),
        on=["subject_id", "date"],
        how="left",
        validate="one_to_one",
    )
    if len(aligned) != len(train):
        raise ValueError(f"{source_name} matrix does not align to sorted train rows.")
    metadata = {
        "source_name": source_name,
        "matrix_path": rel(matrix_path),
        "feature_columns": int(len(feature_cols)),
        "prefixed_feature_sample": list(renamed.values())[:8],
        "missing_rate": float(aligned[list(renamed.values())].isna().mean().mean()) if feature_cols else None,
    }
    return aligned.drop(columns=["subject_id", "date"]), metadata


def build_fusion_features(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    fused = train[["subject_id", "date"]].copy()
    source_metadata = {}
    for source_name, matrix_path in SOURCE_MATRICES.items():
        aligned, metadata = load_prefixed_source_features(source_name, matrix_path, train)
        fused = pd.concat([fused, aligned.reset_index(drop=True)], axis=1)
        source_metadata[source_name] = metadata
    feature_cols = [
        column
        for column in fused.columns
        if column not in {"subject_id", "date"}
        and pd.api.types.is_numeric_dtype(fused[column])
        and fused[column].notna().any()
    ]
    metadata = {
        "source_metadata": source_metadata,
        "aligned_train_rows": int(len(fused)),
        "feature_columns": int(len(feature_cols)),
        "missing_rate": float(fused[feature_cols].isna().mean().mean()) if feature_cols else None,
        "feature_column_sample": feature_cols[:16],
    }
    return fused, metadata


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    splits = load_split_manifest()
    references = {name: load_reference_values(train, candidate_id) for name, candidate_id in REFERENCE_IDS.items()}
    features, feature_metadata = build_fusion_features(train)
    feature_cols = [
        column
        for column in features.columns
        if column not in {"subject_id", "date"}
        and pd.api.types.is_numeric_dtype(features[column])
        and features[column].notna().any()
    ]
    split_rows = scan_split_rows_for_source(
        train=train,
        features=features,
        feature_cols=feature_cols,
        splits=splits,
        references=references,
        source_name="registered_material_fusion",
    )
    summary = summarize_rows(split_rows)
    passing = [row for row in summary if row["local_gate_pass"]]
    best = summary[0] if summary else None
    decision = {
        "reviewed_split_rows": int(len(split_rows)),
        "reviewed_candidates": int(len(summary)),
        "local_gate_pass_count": int(len(passing)),
        "best_candidate": best["candidate"] if best else None,
        "best_target": best["target"] if best else None,
        "best_primary_mean_delta_all_vs_id342": best["primary_mean_delta_all_vs_id342"] if best else None,
        "best_primary_max_delta_all_vs_id342": best["primary_max_delta_all_vs_id342"] if best else None,
        "best_projected_public_proxy": best["projected_public_proxy"] if best else ID342_PUBLIC,
        "best_gap_to_public_055": best["gap_to_public_055"] if best else ID342_PUBLIC - PUBLIC_055,
        "artifact_review_allowed": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "create_submission_artifact": False,
        "submit_now": False,
        "reason": (
            "At least one fusion row passed the mechanical local gate, but this report is discovery-only; require "
            "a stricter artifact/test-geometry gate before Kaggle or DACON action."
            if passing
            else "Registered material fusion did not clear the strict ID342/Public-0.55 local gate."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_registered_material_fusion_gate",
        "purpose": "Test lightweight multimodal fusion of all registered post-ID341 material matrices against ID342.",
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
        "feature_metadata": feature_metadata,
        "top_rows": summary[:40],
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
        "# Post-ID342 registered material fusion gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Reviewed candidates: `{decision['reviewed_candidates']}`",
        f"- Local gate pass count: `{decision['local_gate_pass_count']}`",
        f"- Best candidate: `{decision['best_candidate']}`",
        f"- Best target: `{decision['best_target']}`",
        f"- Best mean/max all-target delta vs ID342: `{decision['best_primary_mean_delta_all_vs_id342']:+.9f}` / `{decision['best_primary_max_delta_all_vs_id342']:+.9f}`",
        f"- Best projected public proxy: `{decision['best_projected_public_proxy']:.9f}`",
        f"- Best gap to Public 0.55: `{decision['best_gap_to_public_055']:+.9f}`",
        f"- Artifact review allowed: `{decision['artifact_review_allowed']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Reason: {decision['reason']}",
        "",
        "## Feature Frame",
        "",
        f"- Feature columns: `{report['feature_metadata']['feature_columns']}`",
        f"- Mean missing rate: `{report['feature_metadata']['missing_rate']:.6f}`",
        "",
        "## Top Rows",
        "",
        "| rank | pass | candidate | target | weight | wins | mean all Δ342 | max all Δ342 | target mean Δ342 | proxy | gap 0.55 |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(report["top_rows"][:25], start=1):
        lines.append(
            "| {rank} | `{passed}` | `{candidate}` | `{target}` | {weight} | {wins} | {mean:+.6f} | {max_delta:+.6f} | {target_mean:+.6f} | {proxy:.6f} | {gap:+.6f} |".format(
                rank=rank,
                passed=row["local_gate_pass"],
                candidate=row["candidate"],
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
