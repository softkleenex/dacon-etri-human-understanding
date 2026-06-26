#!/usr/bin/env python3
"""Report-only S3/Q3 raw sequence representation gate against ID342.

This diagnostic reuses the existing raw half-hour sequence tensor builder and
ROCKET-style random convolution features, but evaluates them only in memory on
frozen ID342 validation lenses. It writes JSON/Markdown reports only.

It does not persist feature matrices, predictions, submission CSVs,
base-prediction artifacts, notebook-history artifacts, submission-log rows,
Kaggle actions, or DACON submissions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402
    BLEND_WEIGHTS,
    PRIMARY_GATE_SPLITS,
    TARGETS,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
    target_loss_on_split,
)
from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    N_BINS,
    align_table,
    build_daily_sequence_table,
    make_kernels,
    random_conv_features,
    table_to_tensor,
)
from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    RAW_DIR as RAW_ITEMS_DIR,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_s3q3_raw_sequence_representation_gate_20260624.json"
ANCHOR_ID = 342
ID342_PUBLIC = 0.5960832519
PUBLIC_055 = 0.55
FOCUS_TARGETS = ["Q3", "S3"]
EPS = 1e-6

MIN_PRIMARY_WINS_VS_ID342 = 4
MAX_PRIMARY_MEAN_DELTA_VS_ID342 = -0.00005
MAX_PRIMARY_WORST_DELTA_VS_ID342 = 0.00025
MAX_PUBLIC_PROXY_FOR_REVIEW = 0.5920
MAX_GAP_TO_PUBLIC_055_FOR_REVIEW = 0.0400


@dataclass(frozen=True)
class ModelSpec:
    name: str
    c_value: float


@dataclass(frozen=True)
class SequenceFamily:
    name: str
    channel_keywords: tuple[str, ...]
    bin_ranges: tuple[tuple[int, int], ...]
    domain_read: str


MODEL_SPECS = (
    ModelSpec("lr_c0_02", 0.02),
    ModelSpec("lr_c0_08", 0.08),
    ModelSpec("lr_c0_20", 0.20),
)

SEQUENCE_FAMILIES = (
    SequenceFamily(
        name="all_channels",
        channel_keywords=(),
        bin_ranges=((0, N_BINS),),
        domain_read="All raw half-hour channels and missingness channels.",
    ),
    SequenceFamily(
        name="q3_usage_routine",
        channel_keywords=("usage", "screen", "activity", "gps", "wifi", "ble", "amb_"),
        bin_ranges=((0, 32),),
        domain_read="Q3 stress/fatigue is screened through app use, phone state, mobility, and context routine.",
    ),
    SequenceFamily(
        name="s3_sleep_boundary",
        channel_keywords=("screen", "charging", "light", "hr", "step", "pedo", "activity", "still", "active"),
        bin_ranges=((12, 40),),
        domain_read="S3 sleep-onset-latency is screened through evening-to-morning light, motion, HR, and phone state.",
    ),
    SequenceFamily(
        name="usage_evening",
        channel_keywords=("usage", "screen", "wifi", "ble", "gps", "amb_"),
        bin_ranges=((8, 32),),
        domain_read="Evening usage, location/proximity, and ambience may capture recovery load and routine disruption.",
    ),
    SequenceFamily(
        name="physio_light_sleep",
        channel_keywords=("hr", "light", "step", "pedo", "activity", "still", "charging", "screen"),
        bin_ranges=((14, 42),),
        domain_read="Physiology, light, movement, and charging/screen context across the pre-sleep and wake boundary.",
    ),
)

RESEARCH_PRIORS = [
    {
        "source": "ETRI Lifelog Dataset 2024 paper",
        "read": (
            "Q2/Q3 are expected to respond to daily mobile-app and routine signals, while S3 is more tied to "
            "step, light, and time-window features."
        ),
    },
    {
        "source": "existing local domain notes",
        "path": "docs/domain/nsf_sleep_guidelines.md",
        "read": "S3 is treated as sleep-onset-latency related; pre-sleep screen, light, movement, and HR are prioritized.",
    },
    {
        "source": "post-ID342 negative evidence",
        "path": "reports/post_id342_raw_subject_state_feature_gate_20260624.md",
        "read": (
            "Daily/window aggregates closed as negative evidence, so this gate keeps temporal ordering through "
            "random convolution features rather than adding another aggregate blend."
        ),
    },
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_test_keys() -> pd.DataFrame:
    test = pd.read_csv(RAW_DIR / "ch2026_submission_sample.csv")
    test["date"] = pd.to_datetime(test["lifelog_date"]).dt.date
    return test.sort_values(["subject_id", "date"]).reset_index(drop=True)


def parse_int_list(text: str) -> list[int]:
    values = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        values.append(int(raw))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def requested_families(text: str) -> list[SequenceFamily]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names or names == ["all"]:
        return list(SEQUENCE_FAMILIES)
    by_name = {family.name: family for family in SEQUENCE_FAMILIES}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown sequence family names: {missing}")
    return [by_name[name] for name in names]


def make_model(spec: ModelSpec):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=spec.c_value, solver="liblinear", max_iter=500, random_state=46),
    )


def fold_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(y_pred, EPS, 1 - EPS), labels=[0, 1]))


def weighted_split_loss(y: np.ndarray, pred: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    weighted = 0.0
    total = 0
    for _fit_idx, valid_idx in folds:
        weighted += fold_loss(y[valid_idx], pred[valid_idx]) * len(valid_idx)
        total += len(valid_idx)
    return float(weighted / total)


def predict_for_split(
    x: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: ModelSpec,
) -> tuple[np.ndarray, int]:
    pred = np.full(len(y), np.nan, dtype=float)
    covered = 0
    for fit_idx, valid_idx in folds:
        if x.shape[1] == 0:
            proba = np.full(len(valid_idx), float(np.mean(y[fit_idx])), dtype=float)
        elif len(np.unique(y[fit_idx])) < 2:
            model = DummyClassifier(strategy="prior")
            model.fit(np.zeros((len(fit_idx), 1)), y[fit_idx])
            proba = np.full(len(valid_idx), float(np.mean(y[fit_idx])), dtype=float)
        else:
            model = make_model(spec)
            model.fit(x[fit_idx], y[fit_idx])
            proba = model.predict_proba(x[valid_idx])[:, 1]
        pred[valid_idx] = np.clip(proba, EPS, 1 - EPS)
        covered += len(valid_idx)
    if np.isnan(pred).any():
        fill_value = float(np.nanmean(pred)) if np.isfinite(np.nanmean(pred)) else float(np.mean(y))
        pred = np.where(np.isnan(pred), fill_value, pred)
    return np.clip(pred, EPS, 1 - EPS), covered


def family_bin_mask(family: SequenceFamily) -> np.ndarray:
    mask = np.zeros(N_BINS, dtype=bool)
    for start, end in family.bin_ranges:
        safe_start = max(0, int(start))
        safe_end = min(N_BINS, int(end))
        mask[safe_start:safe_end] = True
    return mask


def selected_channel_indices(channel_names: list[str], family: SequenceFamily) -> list[int]:
    if not family.channel_keywords:
        return list(range(len(channel_names)))
    indices = []
    for index, name in enumerate(channel_names):
        base_name = name[len("present_") :] if name.startswith("present_") else name
        low = base_name.lower()
        if any(keyword in low for keyword in family.channel_keywords):
            indices.append(index)
    return indices


def family_tensor(
    train_tensor: np.ndarray,
    channel_names: list[str],
    family: SequenceFamily,
) -> tuple[np.ndarray, list[str], list[int]]:
    indices = selected_channel_indices(channel_names, family)
    mask = family_bin_mask(family)
    if indices:
        x = train_tensor[:, indices, :].copy()
        x[:, :, ~mask] = 0.0
    else:
        x = np.zeros((train_tensor.shape[0], 0, N_BINS), dtype=np.float32)
    return x, [channel_names[index] for index in indices], np.where(mask)[0].astype(int).tolist()


def summarize_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["candidate"], []).append(row)
    summary = []
    for candidate, candidate_rows in grouped.items():
        split_rows = {row["split_name"]: row for row in candidate_rows}
        primary_rows = [split_rows[name] for name in PRIMARY_GATE_SPLITS if name in split_rows]
        deltas = [row["delta_all_vs_id342"] for row in primary_rows]
        target_deltas = [row["delta_target_vs_id342"] for row in primary_rows]
        first = candidate_rows[0]
        mean_delta = float(np.mean(deltas)) if deltas else 0.0
        max_delta = float(np.max(deltas)) if deltas else 0.0
        projected_public = float(ID342_PUBLIC + mean_delta)
        checks = {
            "primary_wins_vs_id342": int(sum(delta < 0 for delta in deltas)) >= MIN_PRIMARY_WINS_VS_ID342,
            "primary_mean_delta_material": mean_delta <= MAX_PRIMARY_MEAN_DELTA_VS_ID342,
            "primary_worst_delta_within_ceiling": max_delta <= MAX_PRIMARY_WORST_DELTA_VS_ID342,
            "public_proxy_low_enough": projected_public <= MAX_PUBLIC_PROXY_FOR_REVIEW,
            "public_055_gap_low_enough": (projected_public - PUBLIC_055) <= MAX_GAP_TO_PUBLIC_055_FOR_REVIEW,
        }
        summary.append(
            {
                "candidate": candidate,
                "family": first["family"],
                "target": first["target"],
                "seed": first["seed"],
                "model_spec": first["model_spec"],
                "weight": first["weight"],
                "primary_wins_vs_id342": int(sum(delta < 0 for delta in deltas)),
                "primary_mean_delta_all_vs_id342": mean_delta,
                "primary_max_delta_all_vs_id342": max_delta,
                "primary_mean_delta_target_vs_id342": float(np.mean(target_deltas)) if target_deltas else 0.0,
                "primary_max_delta_target_vs_id342": float(np.max(target_deltas)) if target_deltas else 0.0,
                "projected_public_proxy": projected_public,
                "gap_to_public_055": float(projected_public - PUBLIC_055),
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
            row["candidate"],
        ),
    )


def scan_sequence_representations(
    train: pd.DataFrame,
    train_tensor: np.ndarray,
    channel_names: list[str],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    anchor_values: np.ndarray,
    families: list[SequenceFamily],
    seeds: list[int],
    n_kernels: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    family_metadata: list[dict[str, Any]] = []
    for family in families:
        family_x, selected_channels, selected_bins = family_tensor(train_tensor, channel_names, family)
        metadata = {
            "name": family.name,
            "domain_read": family.domain_read,
            "selected_channels": int(len(selected_channels)),
            "selected_bins": selected_bins,
            "selected_channel_sample": selected_channels[:20],
            "skipped": not selected_channels,
        }
        family_metadata.append(metadata)
        if not selected_channels:
            continue
        for seed in seeds:
            kernels = make_kernels(family_x.shape[1], n_kernels=n_kernels, seed=seed)
            x = random_conv_features(family_x, kernels)
            for target in FOCUS_TARGETS:
                target_idx = TARGETS.index(target)
                y = train[target].astype(int).to_numpy()
                for spec in MODEL_SPECS:
                    split_source_predictions = {
                        split_name: predict_for_split(x, y, splits[split_name], spec)
                        for split_name in PRIMARY_GATE_SPLITS
                    }
                    for weight in BLEND_WEIGHTS:
                        for split_name in PRIMARY_GATE_SPLITS:
                            folds = splits[split_name]
                            source_pred, covered = split_source_predictions[split_name]
                            id342_target_loss = target_loss_on_split(train, folds, target, anchor_values)
                            source_values = anchor_values.copy()
                            source_values[:, target_idx] = np.clip(
                                (1.0 - weight) * anchor_values[:, target_idx] + weight * source_pred,
                                EPS,
                                1.0 - EPS,
                            )
                            candidate_loss = target_loss_on_split(train, folds, target, source_values)
                            delta_target = float(candidate_loss - id342_target_loss)
                            rows.append(
                                {
                                    "family": family.name,
                                    "target": target,
                                    "split_name": split_name,
                                    "candidate": (
                                        f"raw_sequence_representation_{family.name}_{target.lower()}_"
                                        f"{spec.name}_seed{seed}_w{weight:g}"
                                    ),
                                    "seed": int(seed),
                                    "model_spec": spec.name,
                                    "weight": weight,
                                    "valid_rows": int(covered),
                                    "target_loss": candidate_loss,
                                    "delta_target_vs_id342": delta_target,
                                    "delta_all_vs_id342": float(delta_target / len(TARGETS)),
                                }
                            )
    return rows, family_metadata


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    test = read_test_keys()
    seq_table = build_daily_sequence_table(RAW_ITEMS_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    del test_tensor

    splits = load_split_manifest()
    missing_splits = [split for split in PRIMARY_GATE_SPLITS if split not in splits]
    if missing_splits:
        raise ValueError(f"Missing required split names: {missing_splits}")
    anchor_values = load_reference_values(train, ANCHOR_ID)
    families = requested_families(args.families)
    seeds = parse_int_list(args.seeds)
    split_rows, family_metadata = scan_sequence_representations(
        train=train,
        train_tensor=train_tensor,
        channel_names=channel_names,
        splits=splits,
        anchor_values=anchor_values,
        families=families,
        seeds=seeds,
        n_kernels=args.n_kernels,
    )
    summary = summarize_candidate_rows(split_rows)
    passing = [row for row in summary if row["local_gate_pass"]]
    passing_targets = sorted({row["target"] for row in passing})
    best = summary[0] if summary else None
    best_by_target = {}
    for target in FOCUS_TARGETS:
        target_rows = [row for row in summary if row["target"] == target]
        best_by_target[target] = target_rows[0] if target_rows else None
    if passing:
        mean_passing_delta = float(np.mean([row["primary_mean_delta_all_vs_id342"] for row in passing]))
        projected_public_proxy = float(ID342_PUBLIC + mean_passing_delta)
    else:
        mean_passing_delta = 0.0
        projected_public_proxy = best["projected_public_proxy"] if best else ID342_PUBLIC
    final_gate_pass = bool(
        set(passing_targets) == set(FOCUS_TARGETS)
        and mean_passing_delta < -0.0005
        and projected_public_proxy <= MAX_PUBLIC_PROXY_FOR_REVIEW
    )
    next_action = (
        "Both S3 and Q3 passed the strict no-submit gate. Do not create artifacts in this script; run a separate "
        "artifact-contract review before any candidate generation."
        if final_gate_pass
        else "No strict S3/Q3 sequence-representation row passed; keep this as negative evidence, not a candidate."
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_s3q3_raw_sequence_representation_gate",
        "purpose": (
            "Gate raw half-hour sequence representation families against ID342 for S3/Q3 without creating artifacts."
        ),
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "public_target": {"score": PUBLIC_055, "gap_from_anchor": ID342_PUBLIC - PUBLIC_055},
        "research_priors": RESEARCH_PRIORS,
        "target_focus": FOCUS_TARGETS,
        "model_specs": [asdict(spec) for spec in MODEL_SPECS],
        "blend_weights": BLEND_WEIGHTS,
        "primary_gate_splits": PRIMARY_GATE_SPLITS,
        "thresholds": {
            "min_primary_wins_vs_id342": MIN_PRIMARY_WINS_VS_ID342,
            "max_primary_mean_delta_vs_id342": MAX_PRIMARY_MEAN_DELTA_VS_ID342,
            "max_primary_worst_delta_vs_id342": MAX_PRIMARY_WORST_DELTA_VS_ID342,
            "max_public_proxy_for_review": MAX_PUBLIC_PROXY_FOR_REVIEW,
            "max_gap_to_public_055_for_review": MAX_GAP_TO_PUBLIC_055_FOR_REVIEW,
        },
        "sequence_metadata": {
            "raw_items_dir": rel(RAW_ITEMS_DIR),
            "sequence_rows": int(len(seq_table)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "raw_sequence_feature_columns": int(len(feature_cols)),
            "value_channels": int(len(channel_names) // 2),
            "total_channels_with_missingness": int(len(channel_names)),
            "n_bins": N_BINS,
            "n_kernels": int(args.n_kernels),
            "seeds": seeds,
            "families": family_metadata,
        },
        "top_rows": summary[:50],
        "best_by_target": best_by_target,
        "decision": {
            "reviewed_split_rows": int(len(split_rows)),
            "reviewed_candidates": int(len(summary)),
            "local_gate_pass_count": int(len(passing)),
            "passing_targets": passing_targets,
            "all_focus_targets_passing": set(passing_targets) == set(FOCUS_TARGETS),
            "best_candidate": best["candidate"] if best else None,
            "best_target": best["target"] if best else None,
            "best_primary_mean_delta_all_vs_id342": best["primary_mean_delta_all_vs_id342"] if best else None,
            "best_primary_max_delta_all_vs_id342": best["primary_max_delta_all_vs_id342"] if best else None,
            "best_projected_public_proxy": projected_public_proxy,
            "best_gap_to_public_055": float(projected_public_proxy - PUBLIC_055),
            "final_gate_pass": final_gate_pass,
            "artifact_review_allowed_now": False,
            "artifact_review_required_if_gate_passes": final_gate_pass,
            "create_feature_matrix_artifact": False,
            "create_prediction_artifact": False,
            "create_submission_artifact": False,
            "modify_submission_log": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "submit_now": False,
            "dacon_submit_now": False,
            "next_action": next_action,
        },
        "in_memory_feature_matrix_only": True,
        "no_feature_matrix_created": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action_made_by_this_script": True,
        "no_dacon_submission_made": True,
    }


def format_optional_float(value: float | None, digits: int = 9) -> str:
    return "None" if value is None else f"{value:.{digits}f}"


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    metadata = report["sequence_metadata"]
    lines = [
        "# Post-ID342 S3/Q3 raw sequence representation gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It keeps random-convolution sequence features in memory and creates no feature matrix, "
        "prediction, submission artifact, Kaggle action, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Reviewed candidates: `{decision['reviewed_candidates']}`",
        f"- Local gate pass count: `{decision['local_gate_pass_count']}`",
        f"- Passing targets: `{decision['passing_targets']}`",
        f"- Final gate pass: `{decision['final_gate_pass']}`",
        f"- Best candidate: `{decision['best_candidate']}`",
        f"- Best target: `{decision['best_target']}`",
        (
            "- Best mean/max all-target delta vs ID342: "
            f"`{format_optional_float(decision['best_primary_mean_delta_all_vs_id342'])}` / "
            f"`{format_optional_float(decision['best_primary_max_delta_all_vs_id342'])}`"
        ),
        f"- Best projected public proxy: `{decision['best_projected_public_proxy']:.9f}`",
        f"- Best gap to Public 0.55: `{decision['best_gap_to_public_055']:+.9f}`",
        f"- Artifact review allowed now: `{decision['artifact_review_allowed_now']}`",
        f"- Create feature matrix artifact: `{decision['create_feature_matrix_artifact']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Sequence Frame",
        "",
        f"- Sequence rows: `{metadata['sequence_rows']}`",
        f"- Train/test rows: `{metadata['train_rows']}` / `{metadata['test_rows']}`",
        f"- Raw sequence feature columns: `{metadata['raw_sequence_feature_columns']}`",
        f"- Value/total channels: `{metadata['value_channels']}` / `{metadata['total_channels_with_missingness']}`",
        f"- Bins: `{metadata['n_bins']}`",
        f"- Kernels: `{metadata['n_kernels']}`",
        f"- Seeds: `{metadata['seeds']}`",
        "",
        "## Families",
        "",
        "| family | skipped | channels | bins | read |",
        "|---|---:|---:|---:|---|",
    ]
    for family in metadata["families"]:
        lines.append(
            "| {name} | `{skipped}` | {channels} | {bins} | {read} |".format(
                name=family["name"],
                skipped=family["skipped"],
                channels=family["selected_channels"],
                bins=len(family["selected_bins"]),
                read=family["domain_read"],
            )
        )
    lines.extend(
        [
            "",
            "## Top Rows",
            "",
            "| rank | pass | candidate | target | wins | mean all delta | max all delta | proxy | gap 0.55 |",
            "|---:|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(report["top_rows"][:25], start=1):
        lines.append(
            "| {rank} | `{passed}` | `{candidate}` | `{target}` | {wins} | "
            "{mean:+.6f} | {max_delta:+.6f} | {proxy:.6f} | {gap:+.6f} |".format(
                rank=rank,
                passed=row["local_gate_pass"],
                candidate=row["candidate"],
                target=row["target"],
                wins=row["primary_wins_vs_id342"],
                mean=row["primary_mean_delta_all_vs_id342"],
                max_delta=row["primary_max_delta_all_vs_id342"],
                proxy=row["projected_public_proxy"],
                gap=row["gap_to_public_055"],
            )
        )
    lines.extend(["", "## Guardrails", ""])
    for key in [
        "no_feature_matrix_created",
        "no_predictions_created",
        "no_submission_created",
        "no_base_prediction_artifact_created",
        "no_notebook_history_artifact_created",
        "no_submission_log_row_created",
        "no_kaggle_action_made_by_this_script",
        "no_dacon_submission_made",
    ]:
        lines.append(f"- {key}: `{report[key]}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-kernels", type=int, default=256)
    parser.add_argument("--seeds", default="45502,45602")
    parser.add_argument(
        "--families",
        default="all",
        help="Comma-separated sequence family names, or 'all'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_path.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    print(f"Wrote {rel(output_path)}")
    print(f"Wrote {rel(output_path.with_suffix('.md'))}")
    print(f"Final gate pass: {report['decision']['final_gate_pass']}")
    print(f"Create submission artifact: {report['decision']['create_submission_artifact']}")
    print(f"DACON submit now: {report['decision']['dacon_submit_now']}")


if __name__ == "__main__":
    main()
