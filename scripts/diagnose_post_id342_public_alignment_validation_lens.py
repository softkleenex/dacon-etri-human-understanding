#!/usr/bin/env python3
"""Post-ID342 public-alignment validation-lens diagnostic.

This report-only diagnostic asks which local validation lenses best explain the
confirmed public scores after the ID342 correction. It uses only existing OOF
artifacts and recorded public scores.

It creates no submission CSV, base prediction artifact, notebook-history
artifact, submission-log row, Kaggle push/run, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw"
BASE_PREDICTION_DIR = ROOT / "artifacts" / "base_predictions"
SPLIT_MANIFEST_PATH = REPORTS_DIR / "id340_validation_split_manifest_20260618.json"
THREE_SLOT_RESULT_PATH = REPORTS_DIR / "dacon_three_slot_submission_result_20260623.json"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_public_alignment_validation_lens_20260624.json"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
EPS = 1e-6
PUBLIC_055 = 0.55
CURRENT_ANCHOR_ID = 342
SCORED_IDS = [178, 287, 329, 337, 338, 339, 340, 342]
OPTIONAL_UNSCORED_IDS = [221]
RECENT_SLATE_IDS = [342, 339, 338]
FAILED_FRONTIER_IDS = [287, 329]
PRIMARY_SPLITS = [
    "subject_hole_5",
    "actual_test_band_holdout",
    "test_boundary_neighbor2_holdout",
    "test_boundary_neighbor5_holdout",
    "loso_10",
]

RECORDED_PUBLIC_SCORES = {
    178: {
        "public_score": 0.6014113009,
        "source": "recorded_control_public_anchor",
        "note": "Former public anchor.",
    },
    287: {
        "public_score": 0.6698787318,
        "source": "recorded_failed_frontier",
        "note": "Severe local-frontier public failure.",
    },
    329: {
        "public_score": 0.6017908577,
        "source": "recorded_guarded_probe",
        "note": "Practical guarded probe that did not beat ID178.",
    },
    337: {
        "public_score": 0.6006206286219425,
        "source": "recorded_micro_probe",
        "note": "Low-movement S1 random-conv safety probe.",
    },
    340: {
        "public_score": 0.601379175253277,
        "source": "recorded_micro_probe",
        "note": "Q2 sequence-summary safety probe.",
    },
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def finite_or_none(value: float | np.floating[Any]) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def load_train_sorted() -> pd.DataFrame:
    train = pd.read_csv(RAW_DIR / "ch2026_metrics_train.csv")
    train["date"] = pd.to_datetime(train["lifelog_date"]).dt.date
    return train.sort_values(["subject_id", "date"]).reset_index(drop=True)


def load_split_manifest(path: Path = SPLIT_MANIFEST_PATH) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for row in manifest["folds"]:
        splits.setdefault(row["split_name"], []).append(
            (
                np.asarray(row["train_idx"], dtype=int),
                np.asarray(row["valid_idx"], dtype=int),
            )
        )
    return splits


def find_oof(candidate_id: int) -> Path:
    paths = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{candidate_id}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing base OOF ID{candidate_id}")
    if len(paths) > 1:
        # OOF IDs are expected to be immutable. Keep deterministic behavior if
        # legacy duplicates exist and record the chosen path in the report.
        return paths[0]
    return paths[0]


def load_oof_values(train: pd.DataFrame, candidate_id: int) -> tuple[np.ndarray, Path]:
    path = find_oof(candidate_id)
    pred = pd.read_csv(path)
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    key = train[["subject_id", "date"]].copy()
    merged = key.merge(pred, on=["subject_id", "date"], how="left", validate="one_to_one")
    if merged[TARGETS].isna().any().any():
        missing = merged[TARGETS].isna().any(axis=1).sum()
        raise ValueError(f"OOF ID{candidate_id} does not align with sorted train rows; missing rows={missing}")
    values = merged[TARGETS].astype(float).to_numpy()
    return np.clip(values, EPS, 1.0 - EPS), path


def target_loss(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(pred, EPS, 1.0 - EPS), labels=[0, 1]))


def full_target_losses(train: pd.DataFrame, values: np.ndarray) -> dict[str, float]:
    return {
        target: target_loss(train[target].to_numpy(), values[:, target_idx])
        for target_idx, target in enumerate(TARGETS)
    }


def target_loss_on_split(
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    target: str,
    values: np.ndarray,
) -> float:
    target_idx = TARGETS.index(target)
    y = train[target].astype(int).to_numpy()
    weighted = 0.0
    total = 0
    for _fit_idx, valid_idx in folds:
        loss = target_loss(y[valid_idx], values[valid_idx, target_idx])
        weighted += loss * len(valid_idx)
        total += len(valid_idx)
    if total == 0:
        raise ValueError(f"Split for target {target} has no validation rows.")
    return float(weighted / total)


def split_target_losses(
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    values: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        split_name: {
            target: target_loss_on_split(train, folds, target, values)
            for target in TARGETS
        }
        for split_name, folds in splits.items()
    }


def load_public_scores() -> dict[int, dict[str, Any]]:
    public_scores = {candidate_id: dict(info) for candidate_id, info in RECORDED_PUBLIC_SCORES.items()}
    result = json.loads(THREE_SLOT_RESULT_PATH.read_text(encoding="utf-8"))
    for row in result.get("api_attempts", []):
        candidate_id = int(row["id"])
        public_scores[candidate_id] = {
            "public_score": float(row["public_score"]),
            "source": "dacon_three_slot_submission_result",
            "dacon_submission_id": row.get("dacon_submission_id"),
            "leaderboard_c_time": row.get("leaderboard_c_time"),
            "memo": row.get("memo"),
        }
    return public_scores


def mean_loss(losses: dict[str, float]) -> float:
    return float(np.mean([losses[target] for target in TARGETS]))


def candidate_row(
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    public_scores: dict[int, dict[str, Any]],
    candidate_id: int,
) -> dict[str, Any]:
    values, path = load_oof_values(train, candidate_id)
    target_losses = full_target_losses(train, values)
    split_losses_by_target = split_target_losses(train, splits, values)
    split_losses = {
        split_name: mean_loss(losses)
        for split_name, losses in split_losses_by_target.items()
    }
    primary_values = [split_losses[name] for name in PRIMARY_SPLITS if name in split_losses]
    validation_lens_losses = {
        "full_oof": mean_loss(target_losses),
        "primary_mean": float(np.mean(primary_values)),
        "primary_worst": float(np.max(primary_values)),
        **split_losses,
    }
    public_info = public_scores[candidate_id]
    public_score = float(public_info["public_score"])
    return {
        "id": candidate_id,
        "oof_path": rel(path),
        "public_score": public_score,
        "public_source": public_info.get("source"),
        "dacon_submission_id": public_info.get("dacon_submission_id"),
        "leaderboard_c_time": public_info.get("leaderboard_c_time"),
        "note": public_info.get("note"),
        "full_oof_loss": validation_lens_losses["full_oof"],
        "public_minus_full_oof": float(public_score - validation_lens_losses["full_oof"]),
        "target_losses": target_losses,
        "split_losses": split_losses,
        "split_target_losses": split_losses_by_target,
        "validation_lens_losses": validation_lens_losses,
    }


def pairwise_agreement(public_scores: pd.Series, lens_scores: pd.Series) -> float | None:
    total = 0
    agree = 0
    ids = list(public_scores.index)
    for left_pos, left_id in enumerate(ids):
        for right_id in ids[left_pos + 1 :]:
            public_sign = np.sign(public_scores[left_id] - public_scores[right_id])
            lens_sign = np.sign(lens_scores[left_id] - lens_scores[right_id])
            if public_sign == 0 or lens_sign == 0:
                continue
            total += 1
            agree += int(public_sign == lens_sign)
    return float(agree / total) if total else None


def rank_summary(frame: pd.DataFrame, lens_name: str, ids: list[int]) -> dict[str, Any]:
    subset = frame.loc[ids, ["public_score", lens_name]].copy()
    public_ranks = subset["public_score"].rank(method="average", ascending=True)
    lens_ranks = subset[lens_name].rank(method="average", ascending=True)
    spearman = subset["public_score"].corr(subset[lens_name], method="spearman")
    pearson = subset["public_score"].corr(subset[lens_name], method="pearson")
    public_order = [int(idx) for idx in subset.sort_values("public_score").index]
    lens_order = [int(idx) for idx in subset.sort_values(lens_name).index]
    return {
        "lens": lens_name,
        "n": int(len(subset)),
        "spearman": finite_or_none(spearman),
        "pearson": finite_or_none(pearson),
        "rank_mae": float(np.mean(np.abs(public_ranks - lens_ranks))),
        "pairwise_agreement": pairwise_agreement(subset["public_score"], subset[lens_name]),
        "top1_match": bool(public_order[0] == lens_order[0]),
        "bottom1_match": bool(public_order[-1] == lens_order[-1]),
        "public_order": public_order,
        "lens_order": lens_order,
        "best_by_lens": lens_order[0],
        "best_by_public": public_order[0],
    }


def rank_summaries(frame: pd.DataFrame, lens_names: list[str], groups: dict[str, list[int]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for group_name, ids in groups.items():
        rows = [rank_summary(frame, lens_name, ids) for lens_name in lens_names]
        rows.sort(
            key=lambda row: (
                row["spearman"] if row["spearman"] is not None else -2.0,
                row["pairwise_agreement"] if row["pairwise_agreement"] is not None else -1.0,
                -row["rank_mae"],
            ),
            reverse=True,
        )
        out[group_name] = rows
    return out


def target_alignment_summaries(candidate_rows: dict[int, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    target_frame = pd.DataFrame(
        {
            candidate_id: {
                "public_score": row["public_score"],
                **{f"full_{target}": row["target_losses"][target] for target in TARGETS},
                **{
                    f"primary_mean_{target}": float(
                        np.mean([row["split_target_losses"][split_name][target] for split_name in PRIMARY_SPLITS])
                    )
                    for target in TARGETS
                },
            }
            for candidate_id, row in candidate_rows.items()
        }
    ).T
    groups = {
        "all_scored": SCORED_IDS,
        "exclude_id287": [candidate_id for candidate_id in SCORED_IDS if candidate_id != 287],
        "exclude_failed_frontiers": [
            candidate_id for candidate_id in SCORED_IDS if candidate_id not in FAILED_FRONTIER_IDS
        ],
    }
    summaries: dict[str, list[dict[str, Any]]] = {}
    for group_name, ids in groups.items():
        rows: list[dict[str, Any]] = []
        for target in TARGETS:
            for lens_prefix in ["full", "primary_mean"]:
                lens_name = f"{lens_prefix}_{target}"
                summary = rank_summary(target_frame, lens_name, ids)
                summary["target"] = target
                summary["target_lens_type"] = lens_prefix
                rows.append(summary)
        rows.sort(
            key=lambda row: (
                row["spearman"] if row["spearman"] is not None else -2.0,
                row["pairwise_agreement"] if row["pairwise_agreement"] is not None else -1.0,
                -row["rank_mae"],
            ),
            reverse=True,
        )
        summaries[group_name] = rows
    return summaries


def build_lens_frame(candidate_rows: dict[int, dict[str, Any]], lens_names: list[str]) -> pd.DataFrame:
    rows = {}
    for candidate_id, row in candidate_rows.items():
        rows[candidate_id] = {
            "public_score": row["public_score"],
            **{lens_name: row["validation_lens_losses"][lens_name] for lens_name in lens_names},
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def optional_oof_status(train: pd.DataFrame) -> dict[int, dict[str, Any]]:
    status: dict[int, dict[str, Any]] = {}
    for candidate_id in OPTIONAL_UNSCORED_IDS:
        try:
            values, path = load_oof_values(train, candidate_id)
        except FileNotFoundError:
            status[candidate_id] = {"oof_present": False, "public_score_present": False}
            continue
        losses = full_target_losses(train, values)
        status[candidate_id] = {
            "oof_present": True,
            "public_score_present": False,
            "oof_path": rel(path),
            "full_oof_loss": mean_loss(losses),
            "note": "Excluded from public-alignment correlations because it has no confirmed public score.",
        }
    return status


def derive_decision(
    lens_summaries: dict[str, list[dict[str, Any]]],
    target_summaries: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    best_all = lens_summaries["all_scored"][0]
    best_no287 = lens_summaries["exclude_id287"][0]
    best_stable = lens_summaries["exclude_failed_frontiers"][0]
    trusted_lenses = [
        row
        for row in lens_summaries["exclude_failed_frontiers"]
        if (row["spearman"] is not None and row["spearman"] >= 0.60)
        and (row["pairwise_agreement"] is not None and row["pairwise_agreement"] >= 0.70)
    ][:3]
    slate_public_order = lens_summaries["recent_20260623_slate"][0]["public_order"]
    slate_exact_matches = [
        row["lens"]
        for row in lens_summaries["recent_20260623_slate"]
        if row["public_order"] == row["lens_order"]
    ]
    target_leads = target_summaries["exclude_failed_frontiers"][:5]
    all_spearman = best_all["spearman"]
    no287_spearman = best_no287["spearman"]
    stable_spearman = best_stable["spearman"]
    return {
        "best_lens_all_scored": best_all,
        "best_lens_exclude_id287": best_no287,
        "best_lens_exclude_failed_frontiers": best_stable,
        "trusted_lens_candidates": trusted_lenses,
        "recent_slate_public_order": slate_public_order,
        "recent_slate_exact_match_lenses": slate_exact_matches,
        "target_lens_leaders_exclude_failed_frontiers": target_leads,
        "public_055_gap_for_id342": 0.5960832519 - PUBLIC_055,
        "validation_repair_required": True,
        "create_submission_artifact": False,
        "submit_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "recommended_next_gate_policy": (
            "Do not trust full local CV alone. Require any future candidate to beat ID342 on the best stable "
            "public-alignment lens, preserve the recent slate order where relevant, and explicitly report behavior "
            "with ID287 and ID329 excluded so the gate is not driven by failed-frontier artifacts."
        ),
        "reason": (
            "The diagnostic is calibration-only. It explains which local lenses are less misleading, but it does not "
            "create a new independent signal or reduce ID342's public gap to 0.55."
        ),
        "summary_scores": {
            "best_all_scored_spearman": all_spearman,
            "best_exclude_id287_spearman": no287_spearman,
            "best_exclude_failed_frontiers_spearman": stable_spearman,
            "recent_slate_exact_match_count": int(len(slate_exact_matches)),
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    splits = load_split_manifest()
    public_scores = load_public_scores()
    missing_scores = [candidate_id for candidate_id in SCORED_IDS if candidate_id not in public_scores]
    if missing_scores:
        raise ValueError(f"Missing recorded public scores for IDs: {missing_scores}")
    candidate_rows = {
        candidate_id: candidate_row(train, splits, public_scores, candidate_id)
        for candidate_id in SCORED_IDS
    }
    lens_names = list(next(iter(candidate_rows.values()))["validation_lens_losses"].keys())
    lens_frame = build_lens_frame(candidate_rows, lens_names)
    groups = {
        "all_scored": SCORED_IDS,
        "exclude_id287": [candidate_id for candidate_id in SCORED_IDS if candidate_id != 287],
        "exclude_failed_frontiers": [
            candidate_id for candidate_id in SCORED_IDS if candidate_id not in FAILED_FRONTIER_IDS
        ],
        "recent_20260623_slate": RECENT_SLATE_IDS,
    }
    lens_summaries = rank_summaries(lens_frame, lens_names, groups)
    target_summaries = target_alignment_summaries(candidate_rows)
    decision = derive_decision(lens_summaries, target_summaries)
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_public_alignment_validation_lens",
        "purpose": "Measure which local validation lenses explain confirmed public-scored candidates after ID342.",
        "public_score_source": {
            "three_slot_result": rel(THREE_SLOT_RESULT_PATH),
            "recorded_public_scores": RECORDED_PUBLIC_SCORES,
        },
        "split_manifest": rel(SPLIT_MANIFEST_PATH),
        "scored_ids": SCORED_IDS,
        "current_anchor_id": CURRENT_ANCHOR_ID,
        "failed_frontier_ids": FAILED_FRONTIER_IDS,
        "recent_slate_ids": RECENT_SLATE_IDS,
        "lens_names": lens_names,
        "candidate_rows": candidate_rows,
        "lens_alignment_summaries": lens_summaries,
        "target_alignment_summaries": target_summaries,
        "optional_unscored_oof_status": optional_oof_status(train),
        "decision": decision,
        "no_prediction_file_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def format_optional(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    rows = report["candidate_rows"]
    lines = [
        "# Post-ID342 public-alignment validation lens",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.",
        "",
        "## Candidate Scores",
        "",
        "| id | full OOF | public | public-full gap | source | note |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for candidate_id in report["scored_ids"]:
        row = rows[candidate_id]
        lines.append(
            "| {id} | {local:.6f} | {public:.9f} | {gap:+.6f} | `{source}` | {note} |".format(
                id=candidate_id,
                local=row["full_oof_loss"],
                public=row["public_score"],
                gap=row["public_minus_full_oof"],
                source=row["public_source"],
                note=row.get("note") or "",
            )
        )
    lines.extend(
        [
            "",
            "## Lens Alignment",
            "",
            "| group | best lens | spearman | pearson | pairwise | rank MAE | order |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for group_name in ["all_scored", "exclude_id287", "exclude_failed_frontiers", "recent_20260623_slate"]:
        row = report["lens_alignment_summaries"][group_name][0]
        lines.append(
            "| {group} | `{lens}` | {spearman} | {pearson} | {pairwise} | {rank_mae:.3f} | `{order}` |".format(
                group=group_name,
                lens=row["lens"],
                spearman=format_optional(row["spearman"]),
                pearson=format_optional(row["pearson"]),
                pairwise=format_optional(row["pairwise_agreement"]),
                rank_mae=row["rank_mae"],
                order=row["lens_order"],
            )
        )
    lines.extend(
        [
            "",
            "## Recent Slate",
            "",
            f"- Public order: `{decision['recent_slate_public_order']}`",
            f"- Exact-match local lenses: `{decision['recent_slate_exact_match_lenses']}`",
            "",
            "## Target Lens Leaders",
            "",
            "| rank | target | lens type | spearman | pairwise | order |",
            "|---:|---|---|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(decision["target_lens_leaders_exclude_failed_frontiers"], start=1):
        lines.append(
            "| {rank} | `{target}` | `{lens_type}` | {spearman} | {pairwise} | `{order}` |".format(
                rank=rank,
                target=row["target"],
                lens_type=row["target_lens_type"],
                spearman=format_optional(row["spearman"]),
                pairwise=format_optional(row["pairwise_agreement"]),
                order=row["lens_order"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Best all-scored lens: `{decision['best_lens_all_scored']['lens']}`",
            f"- Best lens excluding ID287: `{decision['best_lens_exclude_id287']['lens']}`",
            f"- Best stable lens excluding ID287/ID329: `{decision['best_lens_exclude_failed_frontiers']['lens']}`",
            f"- Trusted lens candidates: `{[row['lens'] for row in decision['trusted_lens_candidates']]}`",
            f"- ID342 gap to Public 0.55: `{decision['public_055_gap_for_id342']:+.9f}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Gate policy: {decision['recommended_next_gate_policy']}",
            f"- Reason: {decision['reason']}",
            "",
        ]
    )
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
