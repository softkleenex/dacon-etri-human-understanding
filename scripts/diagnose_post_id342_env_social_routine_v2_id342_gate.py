#!/usr/bin/env python3
"""ID342-aware env/social/routine v2 fold-internal gate.

This report-only diagnostic retests the post-ID341 Q2/Q3/S3
env-social-routine v2 feature lane against the current ID342 anchor. It uses
only fold-internal models on existing raw-derived feature metadata.

It creates no full-test prediction file, base prediction artifact, notebook
artifact, submission CSV, submission-log row, Kaggle push/run, or DACON
submission.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_q2q3s3_feature_generation_metadata_smoke import (  # noqa: E402,I001
    DEFAULT_OUTPUT as FEATURE_METADATA_REPORT,
)
from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402,I001
    BLEND_WEIGHTS,
    DEFAULT_OUTPUT as POST_ID341_MODEL_SMOKE_PATH,
    MODEL_SPECS,
    PRIMARY_GATE_SPLITS,
    TARGETS,
    aligned_feature_frame,
    feature_columns,
    find_oof,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
    predict_target_for_split,
    target_loss_on_split,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
SPLIT_MANIFEST_PATH = REPORTS_DIR / "id340_validation_split_manifest_20260618.json"
VALIDATION_LENS_REPORT = REPORTS_DIR / "post_id342_public_alignment_validation_lens_20260624.json"
STABLE_LENS_TRIAGE_REPORT = REPORTS_DIR / "post_id342_stable_lens_existing_candidate_triage_20260624.json"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_env_social_routine_v2_id342_gate_20260624.json"

ANCHOR_ID = 342
ID342_PUBLIC = 0.5960832519
PUBLIC_055 = 0.55
TARGET_FOCUS = ["Q2", "Q3", "S3"]
CONFIG_WIN_FLOOR = 4
CONFIG_MEAN_DELTA_CEILING = -0.00005
CONFIG_MAX_DELTA_CEILING = 0.00075
PUBLIC_PROXY_REVIEW_CEILING = 0.5920

RESEARCH_SOURCES = [
    {
        "title": "ETRI Lifelog Dataset 2024",
        "url": "https://arxiv.org/html/2508.03698v1",
        "relevance": "Dataset paper frames the task around continuous sensing for sleep quality and stress.",
    },
    {
        "title": "Human Understanding AI Paper Challenge 2024 - Dataset Design",
        "url": "https://cfiles.dacon.co.kr/competitions/236690/Human%20Understanding%20AI%20Paper%20Challenge%202024%20-%20Dataset%20Design.pdf",
        "relevance": "DACON/ETRI design note emphasizes daily experiences, sleep patterns, emotional states, and stress.",
    },
    {
        "title": "Monitoring Daily Sleep, Mood, and Affect Using Digital Technologies and Wearables",
        "url": "https://www.mdpi.com/1424-8220/24/14/4701",
        "relevance": "Systematic review motivates day-to-day sleep/affect dynamics under ambulatory monitoring.",
    },
    {
        "title": "Detection and monitoring of stress using wearables: a systematic review",
        "url": "https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2024.1478851/full",
        "relevance": "Wearable stress review supports low-level feature computation and cautious validation.",
    },
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def scan_rows_for_target(
    train: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    anchor_values: np.ndarray,
    target: str,
    split_names: list[str],
) -> list[dict[str, Any]]:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        for split_name in split_names:
            folds = splits[split_name]
            pred, covered = predict_target_for_split(features, feature_cols, y, folds, spec)
            anchor_loss = target_loss_on_split(train, folds, target, anchor_values)
            source_values = anchor_values.copy()
            source_values[:, target_idx] = np.where(np.isnan(pred), anchor_values[:, target_idx], pred)
            source_loss = target_loss_on_split(train, folds, target, source_values)
            rows.append(
                {
                    "target": target,
                    "split_name": split_name,
                    "candidate": f"source_envsocial_{spec.name}",
                    "model_spec": asdict(spec),
                    "weight": None,
                    "valid_rows": int(covered),
                    "loss": source_loss,
                    "anchor_loss": anchor_loss,
                    "delta_vs_id342": float(source_loss - anchor_loss),
                }
            )
            for weight in BLEND_WEIGHTS:
                blended = anchor_values.copy()
                blended[:, target_idx] = np.clip(
                    (1.0 - weight) * anchor_values[:, target_idx] + weight * source_values[:, target_idx],
                    1e-6,
                    1.0 - 1e-6,
                )
                loss = target_loss_on_split(train, folds, target, blended)
                rows.append(
                    {
                        "target": target,
                        "split_name": split_name,
                        "candidate": f"blend_id342_envsocial_{spec.name}_w{weight:g}",
                        "model_spec": asdict(spec),
                        "weight": weight,
                        "valid_rows": int(covered),
                        "loss": loss,
                        "anchor_loss": anchor_loss,
                        "delta_vs_id342": float(loss - anchor_loss),
                    }
                )
    return rows


def summarize_candidate_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    frame = pd.DataFrame(rows)
    for target in TARGET_FOCUS:
        summaries: list[dict[str, Any]] = []
        target_frame = frame[frame["target"].eq(target)].copy()
        for candidate, group in target_frame.groupby("candidate", sort=False):
            deltas = group.set_index("split_name")["delta_vs_id342"].to_dict()
            losses = group.set_index("split_name")["loss"].to_dict()
            delta_values = [float(value) for value in deltas.values()]
            row0 = group.iloc[0].to_dict()
            summaries.append(
                {
                    "target": target,
                    "candidate": str(candidate),
                    "model_spec": row0.get("model_spec"),
                    "weight": row0.get("weight"),
                    "primary_wins_vs_id342": int(sum(delta < 0 for delta in delta_values)),
                    "mean_delta_vs_id342": float(np.mean(delta_values)),
                    "min_delta_vs_id342": float(np.min(delta_values)),
                    "max_delta_vs_id342": float(np.max(delta_values)),
                    "split_deltas_vs_id342": {str(k): float(v) for k, v in deltas.items()},
                    "split_losses": {str(k): float(v) for k, v in losses.items()},
                }
            )
        out[target] = sorted(
            summaries,
            key=lambda row: (
                -row["primary_wins_vs_id342"],
                row["mean_delta_vs_id342"],
                row["max_delta_vs_id342"],
                row["candidate"],
            ),
        )
    return out


def select_target_configs(summaries: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for target, rows in summaries.items():
        for row in rows:
            if (
                row["primary_wins_vs_id342"] >= CONFIG_WIN_FLOOR
                and row["mean_delta_vs_id342"] <= CONFIG_MEAN_DELTA_CEILING
                and row["max_delta_vs_id342"] <= CONFIG_MAX_DELTA_CEILING
            ):
                selected[target] = row
                break
    return selected


def anchor_target_table(
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    anchor_values: np.ndarray,
    split_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGET_FOCUS:
        for split_name in split_names:
            rows.append(
                {
                    "target": target,
                    "split_name": split_name,
                    "id342_loss": target_loss_on_split(train, splits[split_name], target, anchor_values),
                }
            )
    return rows


def build_report(split_names: list[str] | None = None) -> dict[str, Any]:
    split_names = split_names or PRIMARY_GATE_SPLITS
    train = load_train_sorted()
    features, feature_metadata = aligned_feature_frame(train)
    cols = feature_columns(features)
    cols = [col for col in cols if pd.api.types.is_numeric_dtype(features[col]) and features[col].notna().any()]
    splits = load_split_manifest()
    anchor_values = load_reference_values(train, ANCHOR_ID)
    scan_rows: list[dict[str, Any]] = []
    for target in TARGET_FOCUS:
        scan_rows.extend(
            scan_rows_for_target(
                train=train,
                features=features,
                feature_cols=cols,
                splits=splits,
                anchor_values=anchor_values,
                target=target,
                split_names=split_names,
            )
        )
    summaries = summarize_candidate_rows(scan_rows)
    selected = select_target_configs(summaries)
    selected_targets = sorted(selected)
    estimated_all_target_delta = (
        float(sum(row["mean_delta_vs_id342"] for row in selected.values()) / len(TARGETS))
        if selected
        else 0.0
    )
    projected_public_proxy = float(ID342_PUBLIC + estimated_all_target_delta)
    public_proxy_reviewable = bool(selected and projected_public_proxy <= PUBLIC_PROXY_REVIEW_CEILING)
    decision = {
        "model_smoke_completed": True,
        "target_focus": TARGET_FOCUS,
        "selected_targets": selected_targets,
        "selected_target_count": int(len(selected_targets)),
        "any_env_social_routine_signal_found": bool(selected_targets),
        "all_focus_targets_covered": selected_targets == TARGET_FOCUS,
        "estimated_all_target_delta_vs_id342": estimated_all_target_delta,
        "projected_public_proxy": projected_public_proxy,
        "gap_to_public_055_proxy": float(projected_public_proxy - PUBLIC_055),
        "public_proxy_reviewable": public_proxy_reviewable,
        "public_055_evidence": False,
        "create_submission_artifact": False,
        "submit_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "next_action": (
            "Selected rows are too small or too narrow for promotion; require a separate geometry/artifact gate before any remote action."
            if selected
            else "Close this ID342-aware env/social/routine v2 local gate; no Kaggle or DACON action is justified."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_env_social_routine_v2_id342_gate",
        "purpose": "Retest the Q2/Q3/S3 env-social-routine v2 feature lane against ID342 under fold-internal validation.",
        "research_sources": RESEARCH_SOURCES,
        "inputs": {
            "feature_metadata_report": rel(FEATURE_METADATA_REPORT),
            "post_id341_model_smoke": rel(POST_ID341_MODEL_SMOKE_PATH),
            "split_manifest": rel(SPLIT_MANIFEST_PATH),
            "validation_lens_report": rel(VALIDATION_LENS_REPORT),
            "stable_lens_triage_report": rel(STABLE_LENS_TRIAGE_REPORT),
            "id342_oof": rel(find_oof(ANCHOR_ID)),
        },
        "feature_metadata": feature_metadata,
        "config": {
            "model_specs": [asdict(spec) for spec in MODEL_SPECS],
            "split_names": split_names,
            "blend_weights": BLEND_WEIGHTS,
            "config_win_floor": CONFIG_WIN_FLOOR,
            "config_mean_delta_ceiling": CONFIG_MEAN_DELTA_CEILING,
            "config_max_delta_ceiling": CONFIG_MAX_DELTA_CEILING,
            "public_proxy_review_ceiling": PUBLIC_PROXY_REVIEW_CEILING,
        },
        "anchor_rows": anchor_target_table(train, splits, anchor_values, split_names),
        "selected_target_configs": selected,
        "top_rows_by_target": {target: rows[:8] for target, rows in summaries.items()},
        "scan_row_count": int(len(scan_rows)),
        "decision": decision,
        "no_full_test_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_made": True,
        "no_kaggle_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    feature_metadata = report["feature_metadata"]
    lines = [
        "# Post-ID342 env/social/routine v2 ID342 gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Research basis",
        "",
    ]
    for source in report["research_sources"]:
        lines.append(f"- [{source['title']}]({source['url']}): {source['relevance']}")
    lines.extend(
        [
            "",
            "## Feature input",
            "",
            f"- Aligned train rows: `{feature_metadata['aligned_train_rows']}`",
            f"- Feature columns used: `{feature_metadata['feature_columns']}`",
            f"- Mean missing rate: `{feature_metadata['mean_missing_rate']:.4f}`",
            "",
            "## Selected target configs",
            "",
            "| target | candidate | wins | mean Δ342 | max Δ342 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    if not report["selected_target_configs"]:
        lines.append("| _(none)_ |  |  |  |  |")
    for target, row in report["selected_target_configs"].items():
        lines.append(
            f"| `{target}` | `{row['candidate']}` | {row['primary_wins_vs_id342']} | "
            f"{row['mean_delta_vs_id342']:+.6f} | {row['max_delta_vs_id342']:+.6f} |"
        )
    lines.extend(["", "## Top rows by target", ""])
    for target, rows in report["top_rows_by_target"].items():
        lines.extend(
            [
                f"### `{target}`",
                "",
                "| rank | candidate | wins | mean Δ342 | max Δ342 |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(rows[:5], start=1):
            lines.append(
                f"| {rank} | `{row['candidate']}` | {row['primary_wins_vs_id342']} | "
                f"{row['mean_delta_vs_id342']:+.6f} | {row['max_delta_vs_id342']:+.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            f"- Model smoke completed: `{decision['model_smoke_completed']}`",
            f"- Selected targets: `{decision['selected_targets']}`",
            f"- Any env/social/routine signal found: `{decision['any_env_social_routine_signal_found']}`",
            f"- Estimated all-target Δ vs ID342: `{decision['estimated_all_target_delta_vs_id342']:+.9f}`",
            f"- Projected public proxy: `{decision['projected_public_proxy']:.9f}`",
            f"- Gap to Public 0.55 proxy: `{decision['gap_to_public_055_proxy']:+.9f}`",
            f"- Public-proxy reviewable: `{decision['public_proxy_reviewable']}`",
            f"- Public 0.55 evidence: `{decision['public_055_evidence']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Next action: {decision['next_action']}",
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
    md_path = output.with_suffix(".md")
    md_path.write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {rel(output)} and {rel(md_path)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
