#!/usr/bin/env python3
"""Post-ID342 public-response geometry diagnostic.

This report-only diagnostic explains the 2026-06-23 three-slot public feedback
using current local artifacts. It compares the scored slate IDs 338/339/342
against ID178/337/340 controls, then asks what geometry made ID342 transfer
best and what that implies for the next no-submit research route.

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
from scripts.run_strict_raw_feature_experiment import TARGETS, build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_public_response_geometry_20260624.json"
THREE_SLOT_RESULT_PATH = REPORTS_DIR / "dacon_three_slot_submission_result_20260623.json"
EPS = 1e-6

CONTROL_PUBLIC_SCORES = {
    178: 0.6014113009,
    337: 0.6006206286219425,
    340: 0.601379175253277,
}
ANALYSIS_IDS = [178, 337, 338, 339, 340, 342]
SCORED_SLATE_IDS = [338, 342, 339]
CURRENT_ANCHOR_ID = 342
PREVIOUS_ANCHOR_ID = 337
PUBLIC_055 = 0.55


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def target_loss(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(pred, EPS, 1.0 - EPS), labels=[0, 1]))


def local_scores(train: pd.DataFrame, pred: pd.DataFrame) -> tuple[float, dict[str, float]]:
    losses = {
        target: target_loss(train[target].astype(int).to_numpy(), pred[target].astype(float).to_numpy())
        for target in TARGETS
    }
    return float(np.mean(list(losses.values()))), losses


def load_public_scores() -> dict[int, dict[str, Any]]:
    result = json.loads(THREE_SLOT_RESULT_PATH.read_text(encoding="utf-8"))
    public_scores = {
        int(row["id"]): {
            "public_score": float(row["public_score"]),
            "dacon_submission_id": row.get("dacon_submission_id"),
            "leaderboard_c_time": row.get("leaderboard_c_time"),
            "memo": row.get("memo"),
            "source": "dacon_three_slot_submission_result",
        }
        for row in result.get("api_attempts", [])
        if row.get("public_score") is not None
    }
    for candidate_id, public_score in CONTROL_PUBLIC_SCORES.items():
        public_scores.setdefault(
            candidate_id,
            {
                "public_score": public_score,
                "dacon_submission_id": None,
                "leaderboard_c_time": None,
                "memo": "control_public_score",
                "source": "recorded_control",
            },
        )
    return public_scores


def prediction_stats(
    pred: pd.DataFrame,
    train_reference: pd.DataFrame,
    anchor_pred: pd.DataFrame | None = None,
) -> dict[str, Any]:
    arr = pred[TARGETS].astype(float).to_numpy()
    train_means = train_reference[TARGETS].astype(float).mean().to_dict()
    target_means = pred[TARGETS].astype(float).mean().to_dict()
    target_mean_deltas_to_train = {
        target: float(target_means[target] - train_means[target])
        for target in TARGETS
    }
    out: dict[str, Any] = {
        "prediction_mean": float(np.mean(arr)),
        "prediction_std": float(np.std(arr)),
        "prediction_entropy": float(np.mean(-(arr * np.log(arr + EPS) + (1.0 - arr) * np.log(1.0 - arr + EPS)))),
        "prediction_sharpness": float(np.mean(np.abs(arr - 0.5))),
        "extreme_rate_05_95": float(np.mean((arr < 0.05) | (arr > 0.95))),
        "target_means": {target: float(target_means[target]) for target in TARGETS},
        "target_mean_deltas_to_train": target_mean_deltas_to_train,
        "target_mean_l1_to_train": float(np.mean([abs(v) for v in target_mean_deltas_to_train.values()])),
        "target_mean_max_abs_to_train": float(max(abs(v) for v in target_mean_deltas_to_train.values())),
    }
    if anchor_pred is not None:
        anchor_arr = anchor_pred[TARGETS].astype(float).to_numpy()
        delta = arr - anchor_arr
        out["movement_vs_anchor"] = {
            "mean_abs_delta": float(np.mean(np.abs(delta))),
            "max_abs_delta": float(np.max(np.abs(delta))),
            "signed_mean_delta": float(np.mean(delta)),
            "per_target_mean_abs_delta": {
                target: float(np.mean(np.abs(delta[:, idx])))
                for idx, target in enumerate(TARGETS)
            },
            "per_target_signed_mean_delta": {
                target: float(np.mean(delta[:, idx]))
                for idx, target in enumerate(TARGETS)
            },
            "per_target_max_abs_delta": {
                target: float(np.max(np.abs(delta[:, idx])))
                for idx, target in enumerate(TARGETS)
            },
        }
    return out


def row_for_candidate(
    candidate_id: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    public_scores: dict[int, dict[str, Any]],
    anchor_test: pd.DataFrame,
) -> dict[str, Any]:
    oof = align(load_artifact("oof", candidate_id), train)
    test_pred = align(load_artifact("test", candidate_id), test)
    local_cv, target_losses = local_scores(train, oof)
    public_info = public_scores.get(candidate_id, {})
    public_score = public_info.get("public_score")
    return {
        "id": candidate_id,
        "local_cv": local_cv,
        "target_losses": target_losses,
        "public_score": float(public_score) if public_score is not None else None,
        "public_minus_local_gap": float(public_score - local_cv) if public_score is not None else None,
        "public_source": public_info.get("source"),
        "dacon_submission_id": public_info.get("dacon_submission_id"),
        "leaderboard_c_time": public_info.get("leaderboard_c_time"),
        "test_geometry_vs_id178": prediction_stats(test_pred, train, anchor_test),
    }


def pairwise_summary(
    ids: list[int],
    rows: dict[int, dict[str, Any]],
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> list[dict[str, Any]]:
    test_preds = {candidate_id: align(load_artifact("test", candidate_id), test) for candidate_id in ids}
    oof_preds = {candidate_id: align(load_artifact("oof", candidate_id), train) for candidate_id in ids}
    out: list[dict[str, Any]] = []
    for left_id in ids:
        for right_id in ids:
            if left_id >= right_id:
                continue
            left_public = rows[left_id]["public_score"]
            right_public = rows[right_id]["public_score"]
            left_local = rows[left_id]["local_cv"]
            right_local = rows[right_id]["local_cv"]
            test_delta = test_preds[right_id][TARGETS].astype(float).to_numpy() - test_preds[left_id][
                TARGETS
            ].astype(float).to_numpy()
            oof_delta = oof_preds[right_id][TARGETS].astype(float).to_numpy() - oof_preds[left_id][TARGETS].astype(
                float
            ).to_numpy()
            out.append(
                {
                    "left_id": left_id,
                    "right_id": right_id,
                    "public_delta_right_minus_left": (
                        float(right_public - left_public)
                        if left_public is not None and right_public is not None
                        else None
                    ),
                    "local_delta_right_minus_left": float(right_local - left_local),
                    "test_mean_abs_delta": float(np.mean(np.abs(test_delta))),
                    "test_max_abs_delta": float(np.max(np.abs(test_delta))),
                    "oof_mean_abs_delta": float(np.mean(np.abs(oof_delta))),
                    "per_target_test_mean_abs_delta": {
                        target: float(np.mean(np.abs(test_delta[:, idx])))
                        for idx, target in enumerate(TARGETS)
                    },
                    "per_target_oof_loss_delta_right_minus_left": {
                        target: float(rows[right_id]["target_losses"][target] - rows[left_id]["target_losses"][target])
                        for target in TARGETS
                    },
                }
            )
    return out


def derive_decision(rows: dict[int, dict[str, Any]], pairwise: list[dict[str, Any]]) -> dict[str, Any]:
    scored_rows = [rows[candidate_id] for candidate_id in SCORED_SLATE_IDS]
    scored_by_public = sorted(scored_rows, key=lambda row: row["public_score"])
    scored_by_local = sorted(scored_rows, key=lambda row: row["local_cv"])
    local_public_order_agrees = [row["id"] for row in scored_by_public] == [row["id"] for row in scored_by_local]
    id342 = rows[342]
    id339 = rows[339]
    id338 = rows[338]
    id342_public_gain_vs_337 = rows[342]["public_score"] - rows[337]["public_score"]
    id342_public_gain_vs_178 = rows[342]["public_score"] - rows[178]["public_score"]
    id342_gap = rows[342]["public_minus_local_gap"]
    slate_gaps = [row["public_minus_local_gap"] for row in scored_rows if row["public_minus_local_gap"] is not None]
    gap_compression_vs_id178 = id342_gap - rows[178]["public_minus_local_gap"]
    pair_342_339 = next(row for row in pairwise if row["left_id"] == 339 and row["right_id"] == 342)
    pair_338_342 = next(row for row in pairwise if row["left_id"] == 338 and row["right_id"] == 342)
    movement_342 = id342["test_geometry_vs_id178"]["movement_vs_anchor"]
    dominant_342_targets = sorted(
        movement_342["per_target_mean_abs_delta"],
        key=movement_342["per_target_mean_abs_delta"].get,
        reverse=True,
    )[:3]
    target_loss_delta_342_vs_339 = pair_342_339["per_target_oof_loss_delta_right_minus_left"]
    best_local_targets_342_vs_339 = sorted(target_loss_delta_342_vs_339, key=target_loss_delta_342_vs_339.get)[:3]
    kaggle_now = False
    return {
        "local_public_order_agrees_within_20260623_slate": bool(local_public_order_agrees),
        "slate_public_order": [row["id"] for row in scored_by_public],
        "slate_local_order": [row["id"] for row in scored_by_local],
        "id342_public_gain_vs_id337": float(id342_public_gain_vs_337),
        "id342_public_gain_vs_id178": float(id342_public_gain_vs_178),
        "id342_gap_to_public_055": float(id342["public_score"] - PUBLIC_055),
        "id342_public_minus_local_gap": float(id342_gap),
        "slate_gap_range": [float(min(slate_gaps)), float(max(slate_gaps))],
        "id342_gap_compression_vs_id178_gap": float(gap_compression_vs_id178),
        "id342_mean_abs_delta_vs_id178": movement_342["mean_abs_delta"],
        "id342_max_abs_delta_vs_id178": movement_342["max_abs_delta"],
        "id342_dominant_movement_targets_vs_id178": dominant_342_targets,
        "id342_best_local_loss_targets_vs_id339": best_local_targets_342_vs_339,
        "id342_vs_id339_public_delta": float(id342["public_score"] - id339["public_score"]),
        "id342_vs_id339_local_delta": float(id342["local_cv"] - id339["local_cv"]),
        "id342_vs_id339_test_mean_abs_delta": pair_342_339["test_mean_abs_delta"],
        "id342_vs_id338_public_delta": float(id342["public_score"] - id338["public_score"]),
        "id342_vs_id338_local_delta": float(id342["local_cv"] - id338["local_cv"]),
        "id342_vs_id338_test_mean_abs_delta": pair_338_342["test_mean_abs_delta"],
        "create_submission_artifact": False,
        "submit_now": False,
        "push_kaggle_now": kaggle_now,
        "run_kaggle_now": kaggle_now,
        "next_research_route": "id342_preserving_feature_similarity_source_decomposition",
        "next_action": (
            "Do not spend DACON or Kaggle on the failed Q3/S3 raw/sequence branch. "
            "Decompose the ID221 feature-similarity source that transferred through ID342, then test only low-movement "
            "ID342-preserving variants with the same public/local order and joint-risk constraints."
        ),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    public_scores = load_public_scores()
    id178_test = align(load_artifact("test", 178), test)
    rows = {
        candidate_id: row_for_candidate(candidate_id, train, test, public_scores, id178_test)
        for candidate_id in ANALYSIS_IDS
    }
    pairwise = pairwise_summary(ANALYSIS_IDS, rows, train, test)
    decision = derive_decision(rows, pairwise)
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_public_response_geometry",
        "purpose": "Explain the ID342 public improvement and choose the next no-submit research route.",
        "public_score_source": rel(THREE_SLOT_RESULT_PATH),
        "analysis_ids": ANALYSIS_IDS,
        "scored_slate_ids": SCORED_SLATE_IDS,
        "current_anchor_id": CURRENT_ANCHOR_ID,
        "previous_anchor_id": PREVIOUS_ANCHOR_ID,
        "candidate_rows": rows,
        "pairwise_rows": pairwise,
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
    rows = report["candidate_rows"]
    lines = [
        "# Post-ID342 public-response geometry",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.",
        "",
        "## Public Read",
        "",
        f"- Slate public order: `{decision['slate_public_order']}`",
        f"- Slate local order: `{decision['slate_local_order']}`",
        f"- Local/public order agrees in 2026-06-23 slate: `{decision['local_public_order_agrees_within_20260623_slate']}`",
        f"- ID342 public gain vs ID337: `{decision['id342_public_gain_vs_id337']:+.9f}`",
        f"- ID342 public gain vs ID178: `{decision['id342_public_gain_vs_id178']:+.9f}`",
        f"- ID342 gap to Public 0.55: `{decision['id342_gap_to_public_055']:+.9f}`",
        f"- ID342 public-minus-local gap: `{decision['id342_public_minus_local_gap']:+.9f}`",
        f"- ID342 gap compression vs ID178 gap: `{decision['id342_gap_compression_vs_id178_gap']:+.9f}`",
        "",
        "## Candidate Rows",
        "",
        "| id | local | public | gap | mean Δ178 | max Δ178 | target-mean L1 | entropy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate_id in report["analysis_ids"]:
        row = rows[str(candidate_id)] if str(candidate_id) in rows else rows[candidate_id]
        geom = row["test_geometry_vs_id178"]
        movement = geom["movement_vs_anchor"]
        public = row["public_score"]
        lines.append(
            "| {id} | {local:.6f} | {public} | {gap} | {mean_delta:.6f} | {max_delta:.6f} | {l1:.6f} | {entropy:.6f} |".format(
                id=candidate_id,
                local=row["local_cv"],
                public=f"{public:.9f}" if public is not None else "",
                gap=f"{row['public_minus_local_gap']:+.6f}" if row["public_minus_local_gap"] is not None else "",
                mean_delta=movement["mean_abs_delta"],
                max_delta=movement["max_abs_delta"],
                l1=geom["target_mean_l1_to_train"],
                entropy=geom["prediction_entropy"],
            )
        )
    lines.extend(
        [
            "",
            "## ID342 Movement",
            "",
            f"- Dominant movement targets vs ID178: `{decision['id342_dominant_movement_targets_vs_id178']}`",
            f"- Best local-loss targets vs ID339: `{decision['id342_best_local_loss_targets_vs_id339']}`",
            f"- ID342 vs ID339 public/local deltas: `{decision['id342_vs_id339_public_delta']:+.9f}` / `{decision['id342_vs_id339_local_delta']:+.9f}`",
            f"- ID342 vs ID338 public/local deltas: `{decision['id342_vs_id338_public_delta']:+.9f}` / `{decision['id342_vs_id338_local_delta']:+.9f}`",
            "",
            "## Decision",
            "",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Next research route: `{decision['next_research_route']}`",
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
