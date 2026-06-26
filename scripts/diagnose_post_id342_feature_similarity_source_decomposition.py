#!/usr/bin/env python3
"""Post-ID342 ID221 feature-similarity source decomposition.

This report-only diagnostic decomposes the feature-similarity source behind
ID342. ID342 is a 50/50 blend of ID178 and historical ID221 on every target.
Here we scan target-specific alpha changes around 0.50 and evaluate them against
ID342 under frozen primary split lenses plus low-movement geometry checks.

It creates no submission CSV, base prediction artifact, notebook-history
artifact, submission-log row, Kaggle push/run, or DACON submission.
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

from scripts.diagnose_id179_vs_id335_sequence_probe_review import align, load_artifact  # noqa: E402
from scripts.diagnose_post_id342_raw_temporal_q3s3_sequence_image_smoke import PRIMARY_SPLITS  # noqa: E402
from scripts.run_strict_raw_feature_experiment import TARGETS, build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_feature_similarity_source_decomposition_20260624.json"
PUBLIC_RESPONSE_PATH = REPORTS_DIR / "post_id342_public_response_geometry_20260624.json"
SPLIT_MANIFEST_PATH = REPORTS_DIR / "id340_validation_split_manifest_20260618.json"
EPS = 1e-6

BASE_ID = 178
SOURCE_ID = 221
ANCHOR_ID = 342
ID342_PUBLIC = 0.5960832519
PUBLIC_055 = 0.55
BASE_ALPHA = 0.50
ALPHA_GRID = [0.0, 0.25, 0.35, 0.50, 0.65, 0.80, 1.0]
GROUPS: dict[str, list[str]] = {
    "q1_s4_public_response": ["Q1", "S4"],
    "q1_q3_s4_local_gain": ["Q1", "Q3", "S4"],
    "q123_subjective": ["Q1", "Q2", "Q3"],
    "sleep_objective": ["S1", "S2", "S3", "S4"],
}
PRIMARY_WIN_FLOOR = 4
MEAN_DELTA_CEILING = -0.00005
MAX_DELTA_CEILING = 0.00075
MAX_TEST_MOVEMENT_VS_ID342 = 0.035


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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


def target_loss(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(pred, EPS, 1.0 - EPS), labels=[0, 1]))


def target_losses(train: pd.DataFrame, values: np.ndarray) -> dict[str, float]:
    return {
        target: target_loss(train[target].astype(int).to_numpy(), values[:, target_idx])
        for target_idx, target in enumerate(TARGETS)
    }


def average_loss(losses: dict[str, float]) -> float:
    return float(np.mean([losses[target] for target in TARGETS]))


def split_mean_loss(train: pd.DataFrame, values: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    total = 0
    weighted = 0.0
    y_by_target = {target: train[target].astype(int).to_numpy() for target in TARGETS}
    for _fit_idx, valid_idx in folds:
        losses = [
            target_loss(y_by_target[target][valid_idx], values[valid_idx, target_idx])
            for target_idx, target in enumerate(TARGETS)
        ]
        weighted += float(np.mean(losses)) * len(valid_idx)
        total += len(valid_idx)
    return float(weighted / total)


def blend_values(base_values: np.ndarray, source_values: np.ndarray, alpha_by_target: dict[str, float]) -> np.ndarray:
    out = base_values.copy()
    for target_idx, target in enumerate(TARGETS):
        alpha = float(alpha_by_target.get(target, BASE_ALPHA))
        out[:, target_idx] = np.clip(
            (1.0 - alpha) * base_values[:, target_idx] + alpha * source_values[:, target_idx],
            EPS,
            1.0 - EPS,
        )
    return out


def candidate_name(changed_targets: list[str], alpha: float) -> str:
    target_slug = "".join(t.lower() for t in changed_targets)
    return f"id178_id221_alpha{alpha:.2f}_{target_slug}_rest050"


def build_candidate_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for target in TARGETS:
        for alpha in ALPHA_GRID:
            if alpha == BASE_ALPHA:
                continue
            alpha_by_target = dict.fromkeys(TARGETS, BASE_ALPHA)
            alpha_by_target[target] = alpha
            configs.append(
                {
                    "name": candidate_name([target], alpha),
                    "kind": "single_target_alpha",
                    "changed_targets": [target],
                    "alpha": alpha,
                    "alpha_by_target": alpha_by_target,
                }
            )
    for group_name, group_targets in GROUPS.items():
        for alpha in ALPHA_GRID:
            if alpha == BASE_ALPHA:
                continue
            alpha_by_target = dict.fromkeys(TARGETS, BASE_ALPHA)
            for target in group_targets:
                alpha_by_target[target] = alpha
            configs.append(
                {
                    "name": f"id178_id221_alpha{alpha:.2f}_{group_name}_rest050",
                    "kind": "group_alpha",
                    "changed_targets": group_targets,
                    "alpha": alpha,
                    "alpha_by_target": alpha_by_target,
                }
            )
    return configs


def evaluate_candidate(
    config: dict[str, Any],
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    base_oof: np.ndarray,
    source_oof: np.ndarray,
    anchor_oof: np.ndarray,
    base_test: np.ndarray,
    source_test: np.ndarray,
    anchor_test: np.ndarray,
    anchor_losses: dict[str, float],
    anchor_split_losses: dict[str, float],
) -> dict[str, Any]:
    candidate_oof = blend_values(base_oof, source_oof, config["alpha_by_target"])
    candidate_test = blend_values(base_test, source_test, config["alpha_by_target"])
    losses = target_losses(train, candidate_oof)
    local_cv = average_loss(losses)
    split_deltas: dict[str, float] = {}
    split_losses: dict[str, float] = {}
    wins = 0
    for split_name in PRIMARY_SPLITS:
        split_loss = split_mean_loss(train, candidate_oof, splits[split_name])
        delta = split_loss - anchor_split_losses[split_name]
        split_losses[split_name] = split_loss
        split_deltas[split_name] = delta
        wins += int(delta < 0)
    test_delta_vs342 = candidate_test - anchor_test
    test_delta_vs178 = candidate_test - base_test
    target_loss_deltas = {target: float(losses[target] - anchor_losses[target]) for target in TARGETS}
    mean_delta = float(np.mean(list(split_deltas.values())))
    max_delta = float(np.max(list(split_deltas.values())))
    max_test_movement_vs342 = float(np.max(np.abs(test_delta_vs342)))
    return {
        "name": config["name"],
        "kind": config["kind"],
        "changed_targets": config["changed_targets"],
        "alpha": float(config["alpha"]),
        "alpha_by_target": {target: float(config["alpha_by_target"][target]) for target in TARGETS},
        "local_cv": local_cv,
        "local_delta_vs_id342": float(local_cv - average_loss(anchor_losses)),
        "target_losses": losses,
        "target_loss_deltas_vs_id342": target_loss_deltas,
        "primary_wins_vs_id342": int(wins),
        "mean_primary_delta_vs_id342": mean_delta,
        "max_primary_delta_vs_id342": max_delta,
        "min_primary_delta_vs_id342": float(np.min(list(split_deltas.values()))),
        "split_deltas_vs_id342": split_deltas,
        "split_losses": split_losses,
        "test_mean_abs_delta_vs_id342": float(np.mean(np.abs(test_delta_vs342))),
        "test_max_abs_delta_vs_id342": max_test_movement_vs342,
        "test_mean_abs_delta_vs_id178": float(np.mean(np.abs(test_delta_vs178))),
        "test_max_abs_delta_vs_id178": float(np.max(np.abs(test_delta_vs178))),
        "per_target_test_mean_abs_delta_vs_id342": {
            target: float(np.mean(np.abs(test_delta_vs342[:, idx])))
            for idx, target in enumerate(TARGETS)
        },
        "eligible_local_gate": bool(
            wins >= PRIMARY_WIN_FLOOR
            and mean_delta <= MEAN_DELTA_CEILING
            and max_delta <= MAX_DELTA_CEILING
            and max_test_movement_vs342 <= MAX_TEST_MOVEMENT_VS_ID342
        ),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    base_oof_df = align(load_artifact("oof", BASE_ID), train)
    source_oof_df = align(load_artifact("oof", SOURCE_ID), train)
    anchor_oof_df = align(load_artifact("oof", ANCHOR_ID), train)
    base_test_df = align(load_artifact("test", BASE_ID), test)
    source_test_df = align(load_artifact("test", SOURCE_ID), test)
    anchor_test_df = align(load_artifact("test", ANCHOR_ID), test)
    base_oof = base_oof_df[TARGETS].astype(float).to_numpy()
    source_oof = source_oof_df[TARGETS].astype(float).to_numpy()
    anchor_oof = anchor_oof_df[TARGETS].astype(float).to_numpy()
    base_test = base_test_df[TARGETS].astype(float).to_numpy()
    source_test = source_test_df[TARGETS].astype(float).to_numpy()
    anchor_test = anchor_test_df[TARGETS].astype(float).to_numpy()
    reconstructed_anchor = blend_values(base_oof, source_oof, dict.fromkeys(TARGETS, BASE_ALPHA))
    reconstruction_error = float(np.max(np.abs(reconstructed_anchor - anchor_oof)))
    splits = load_split_manifest()
    missing_splits = [split for split in PRIMARY_SPLITS if split not in splits]
    if missing_splits:
        raise ValueError(f"Missing required split names: {missing_splits}")
    anchor_losses = target_losses(train, anchor_oof)
    source_losses = target_losses(train, source_oof)
    base_losses = target_losses(train, base_oof)
    anchor_split_losses = {
        split_name: split_mean_loss(train, anchor_oof, splits[split_name])
        for split_name in PRIMARY_SPLITS
    }
    rows = [
        evaluate_candidate(
            config,
            train,
            splits,
            base_oof,
            source_oof,
            anchor_oof,
            base_test,
            source_test,
            anchor_test,
            anchor_losses,
            anchor_split_losses,
        )
        for config in build_candidate_configs()
    ]
    rows = sorted(rows, key=lambda row: (not row["eligible_local_gate"], row["mean_primary_delta_vs_id342"], row["local_delta_vs_id342"], row["test_max_abs_delta_vs_id342"]))
    eligible_rows = [row for row in rows if row["eligible_local_gate"]]
    best = rows[0] if rows else None
    best_proxy = ID342_PUBLIC + float(best["local_delta_vs_id342"] if best else 0.0)
    decision = {
        "anchor_reconstruction_error": reconstruction_error,
        "eligible_count": int(len(eligible_rows)),
        "best_config": best["name"] if best else None,
        "best_changed_targets": best["changed_targets"] if best else [],
        "best_alpha_by_target": best["alpha_by_target"] if best else {},
        "best_local_delta_vs_id342": float(best["local_delta_vs_id342"] if best else 0.0),
        "best_mean_primary_delta_vs_id342": float(best["mean_primary_delta_vs_id342"] if best else 0.0),
        "best_max_primary_delta_vs_id342": float(best["max_primary_delta_vs_id342"] if best else 0.0),
        "best_test_max_abs_delta_vs_id342": float(best["test_max_abs_delta_vs_id342"] if best else 0.0),
        "projected_public_proxy_from_id342": best_proxy,
        "gap_to_public_055_after_proxy": float(best_proxy - PUBLIC_055),
        "public_055_evidence": False,
        "create_submission_artifact": False,
        "submit_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "next_action": (
            "Keep this as report-only decomposition. If a row is locally eligible, it still needs a stricter "
            "artifact/joint-risk gate because it is an ID342-neighborhood post-public scan, not independent "
            "Public 0.55 evidence."
            if eligible_rows
            else "ID342 appears close to the safe local optimum for simple target-alpha ID221 decomposition. "
            "Do not create a candidate or Kaggle run from this scan; move to a genuinely new feature-similarity "
            "source or a clean fold-internal rebuild."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_feature_similarity_source_decomposition",
        "purpose": "Decompose the ID178/ID221 alpha source behind ID342 without creating artifacts.",
        "public_response_geometry_report": rel(PUBLIC_RESPONSE_PATH),
        "base_id": BASE_ID,
        "source_id": SOURCE_ID,
        "anchor_id": ANCHOR_ID,
        "base_alpha": BASE_ALPHA,
        "alpha_grid": ALPHA_GRID,
        "primary_splits": PRIMARY_SPLITS,
        "gates": {
            "primary_win_floor": PRIMARY_WIN_FLOOR,
            "mean_delta_ceiling": MEAN_DELTA_CEILING,
            "max_delta_ceiling": MAX_DELTA_CEILING,
            "max_test_movement_vs_id342": MAX_TEST_MOVEMENT_VS_ID342,
        },
        "reference_losses": {
            "id178": {"local_cv": average_loss(base_losses), "target_losses": base_losses},
            "id221": {"local_cv": average_loss(source_losses), "target_losses": source_losses},
            "id342": {"local_cv": average_loss(anchor_losses), "target_losses": anchor_losses},
        },
        "rows": rows,
        "top_rows": rows[:12],
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
    refs = report["reference_losses"]
    lines = [
        "# Post-ID342 feature-similarity source decomposition",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.",
        "",
        "## Reference Losses",
        "",
        f"- ID178 local: `{refs['id178']['local_cv']:.6f}`",
        f"- ID221 local: `{refs['id221']['local_cv']:.6f}`",
        f"- ID342 local: `{refs['id342']['local_cv']:.6f}`",
        f"- ID342 reconstruction max error from ID178/ID221 alpha 0.50: `{decision['anchor_reconstruction_error']:.12f}`",
        "",
        "## Top Rows",
        "",
        "| rank | config | targets | alpha | eligible | local Δ342 | wins | mean Δ | max Δ | max test Δ342 | proxy |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(report["top_rows"], start=1):
        proxy = ID342_PUBLIC + row["local_delta_vs_id342"]
        lines.append(
            "| {rank} | `{name}` | `{targets}` | {alpha:.2f} | `{eligible}` | {local_delta:+.6f} | {wins} | {mean_delta:+.6f} | {max_delta:+.6f} | {max_test:.6f} | {proxy:.6f} |".format(
                rank=rank,
                name=row["name"],
                targets=",".join(row["changed_targets"]),
                alpha=row["alpha"],
                eligible=row["eligible_local_gate"],
                local_delta=row["local_delta_vs_id342"],
                wins=row["primary_wins_vs_id342"],
                mean_delta=row["mean_primary_delta_vs_id342"],
                max_delta=row["max_primary_delta_vs_id342"],
                max_test=row["test_max_abs_delta_vs_id342"],
                proxy=proxy,
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Eligible rows: `{decision['eligible_count']}`",
            f"- Best config: `{decision['best_config']}`",
            f"- Best changed targets: `{decision['best_changed_targets']}`",
            f"- Best local delta vs ID342: `{decision['best_local_delta_vs_id342']:+.6f}`",
            f"- Best mean/max primary delta vs ID342: `{decision['best_mean_primary_delta_vs_id342']:+.6f}` / `{decision['best_max_primary_delta_vs_id342']:+.6f}`",
            f"- Projected public proxy from ID342: `{decision['projected_public_proxy_from_id342']:.9f}`",
            f"- Gap to Public 0.55 after proxy: `{decision['gap_to_public_055_after_proxy']:+.9f}`",
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
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    print(f"Wrote {rel(args.output)} and {rel(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
