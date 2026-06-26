#!/usr/bin/env python3
"""ID342 Q3/S3 slice anatomy and stable-donor selector gate.

This report-only diagnostic explains why the public-scored target-swap lane
does not open Q3/S3. It compares ID342 with stable public-scored donors and
failed-frontier donors by target, subject, and frozen validation lens, then
tests a fold-internal stable-donor selector.

It creates no prediction CSV, base prediction artifact, notebook-history
artifact, submission-log row, Kaggle push/run, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
BASE_PREDICTION_DIR = ROOT / "artifacts" / "base_predictions"
SPLIT_MANIFEST_PATH = REPORTS_DIR / "id340_validation_split_manifest_20260618.json"
TARGET_SWAP_GATE_PATH = REPORTS_DIR / "post_id342_public_scored_target_swap_gate_20260624.json"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_q3s3_slice_selector_anatomy_20260624.json"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
FOCUS_TARGETS = ["Q3", "S3"]
ANCHOR_ID = 342
STABLE_DONOR_IDS = [178, 337, 338, 339, 340]
FAILED_FRONTIER_IDS = [287, 329]
ALL_IDS = [178, 287, 329, 337, 338, 339, 340, 342]
PRIMARY_SPLITS = [
    "subject_hole_5",
    "actual_test_band_holdout",
    "test_boundary_neighbor2_holdout",
    "test_boundary_neighbor5_holdout",
    "loso_10",
]
SELECTOR_WEIGHTS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
EPS = 1e-6
ID342_PUBLIC = 0.5960832519
PUBLIC_055 = 0.55
TARGET_WIN_FLOOR = 4
TARGET_MEAN_DELTA_CEILING = -0.00005
TARGET_MAX_DELTA_CEILING = 0.00075

PUBLIC_SCORES = {
    178: 0.6014113009,
    287: 0.6698787318,
    329: 0.6017908577,
    337: 0.6006206286219425,
    338: 0.5992145177,
    339: 0.5985187787,
    340: 0.601379175253277,
    342: ID342_PUBLIC,
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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
    return paths[0]


def load_reference_values(train: pd.DataFrame, candidate_id: int) -> np.ndarray:
    pred = pd.read_csv(find_oof(candidate_id))
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    merged = train[["subject_id", "date"]].merge(pred, on=["subject_id", "date"], how="left")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"Reference ID{candidate_id} does not align with sorted train rows.")
    return merged[TARGETS].astype(float).to_numpy()


def fold_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(y_pred, EPS, 1 - EPS), labels=[0, 1]))


def weighted_loss(y: np.ndarray, pred: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    weighted = 0.0
    total = 0
    for _fit_idx, valid_idx in folds:
        weighted += fold_loss(y[valid_idx], pred[valid_idx]) * len(valid_idx)
        total += len(valid_idx)
    return float(weighted / total)


def target_loss_table(train: pd.DataFrame, refs: dict[int, np.ndarray]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for candidate_id, values in refs.items():
        losses = {
            target: fold_loss(train[target].astype(int).to_numpy(), values[:, target_idx])
            for target_idx, target in enumerate(TARGETS)
        }
        table[str(candidate_id)] = {
            "public_score": PUBLIC_SCORES.get(candidate_id),
            "mean_loss": float(np.mean(list(losses.values()))),
            "focus_mean_loss": float(np.mean([losses[target] for target in FOCUS_TARGETS])),
            "target_losses": losses,
        }
    return table


def subject_slice_table(train: pd.DataFrame, refs: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    anchor = refs[ANCHOR_ID]
    for target in FOCUS_TARGETS:
        target_idx = TARGETS.index(target)
        y = train[target].astype(int).to_numpy()
        for subject_id, subject_idx in train.groupby("subject_id", sort=True).groups.items():
            idx = np.asarray(sorted(subject_idx), dtype=int)
            anchor_loss = fold_loss(y[idx], anchor[idx, target_idx])
            stable_losses = {
                str(candidate_id): fold_loss(y[idx], refs[candidate_id][idx, target_idx])
                for candidate_id in STABLE_DONOR_IDS
            }
            failed_losses = {
                str(candidate_id): fold_loss(y[idx], refs[candidate_id][idx, target_idx])
                for candidate_id in FAILED_FRONTIER_IDS
            }
            best_stable_id, best_stable_loss = min(stable_losses.items(), key=lambda item: item[1])
            best_failed_id, best_failed_loss = min(failed_losses.items(), key=lambda item: item[1])
            rows.append(
                {
                    "target": target,
                    "subject_id": subject_id,
                    "rows": int(len(idx)),
                    "positive_rate": float(y[idx].mean()),
                    "id342_loss": anchor_loss,
                    "best_stable_id": int(best_stable_id),
                    "best_stable_loss": float(best_stable_loss),
                    "best_stable_delta_vs_id342": float(best_stable_loss - anchor_loss),
                    "best_failed_frontier_id": int(best_failed_id),
                    "best_failed_frontier_loss": float(best_failed_loss),
                    "best_failed_frontier_delta_vs_id342": float(best_failed_loss - anchor_loss),
                }
            )
    return sorted(rows, key=lambda row: (row["target"], row["best_stable_delta_vs_id342"]))


def selector_config_predictions(refs: dict[int, np.ndarray], target_idx: int) -> list[tuple[str, np.ndarray]]:
    configs = [("base", refs[ANCHOR_ID][:, target_idx])]
    for donor_id in STABLE_DONOR_IDS:
        for weight in SELECTOR_WEIGHTS:
            pred = np.clip(
                (1.0 - weight) * refs[ANCHOR_ID][:, target_idx] + weight * refs[donor_id][:, target_idx],
                EPS,
                1.0 - EPS,
            )
            configs.append((f"donor{donor_id}_w{weight:g}", pred))
    return configs


def stable_selector_gate(
    train: pd.DataFrame,
    refs: dict[int, np.ndarray],
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
) -> dict[str, Any]:
    target_reports: dict[str, Any] = {}
    for target in FOCUS_TARGETS:
        target_idx = TARGETS.index(target)
        y = train[target].astype(int).to_numpy()
        split_rows: list[dict[str, Any]] = []
        for split_name in PRIMARY_SPLITS:
            pred = np.full(len(train), np.nan, dtype=float)
            selected: list[str] = []
            for fit_idx, valid_idx in splits[split_name]:
                config_losses = [
                    (fold_loss(y[fit_idx], config_pred[fit_idx]), config_name, config_pred)
                    for config_name, config_pred in selector_config_predictions(refs, target_idx)
                ]
                config_losses.sort(key=lambda item: item[0])
                _loss, config_name, config_pred = config_losses[0]
                pred[valid_idx] = config_pred[valid_idx]
                selected.append(config_name)
            anchor_loss = weighted_loss(y, refs[ANCHOR_ID][:, target_idx], splits[split_name])
            selector_loss = weighted_loss(y, pred, splits[split_name])
            split_rows.append(
                {
                    "split_name": split_name,
                    "anchor_loss": anchor_loss,
                    "selector_loss": selector_loss,
                    "delta_vs_id342": selector_loss - anchor_loss,
                    "selected_counts": dict(Counter(selected)),
                }
            )
        deltas = [row["delta_vs_id342"] for row in split_rows]
        wins = int(sum(delta < 0 for delta in deltas))
        mean_delta = float(np.mean(deltas))
        max_delta = float(np.max(deltas))
        target_reports[target] = {
            "wins": wins,
            "mean_delta": mean_delta,
            "max_delta": max_delta,
            "min_delta": float(np.min(deltas)),
            "eligible": bool(
                wins >= TARGET_WIN_FLOOR
                and mean_delta <= TARGET_MEAN_DELTA_CEILING
                and max_delta <= TARGET_MAX_DELTA_CEILING
            ),
            "split_rows": split_rows,
        }
    return target_reports


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    splits = load_split_manifest()
    refs = {candidate_id: load_reference_values(train, candidate_id) for candidate_id in ALL_IDS}
    target_scores = target_loss_table(train, refs)
    subject_rows = subject_slice_table(train, refs)
    selector_reports = stable_selector_gate(train, refs, splits)
    selected_targets = [target for target, report in selector_reports.items() if report["eligible"]]
    estimated_all_target_delta = float(sum(selector_reports[target]["mean_delta"] for target in selected_targets) / len(TARGETS))
    projected_public_proxy = ID342_PUBLIC + estimated_all_target_delta
    stable_subject_leads = [row for row in subject_rows if row["best_stable_delta_vs_id342"] < 0]
    failed_subject_leads = [row for row in subject_rows if row["best_failed_frontier_delta_vs_id342"] < 0]

    decision = {
        "stable_selector_selected_targets": selected_targets,
        "q3_selector_pass": "Q3" in selected_targets,
        "s3_selector_pass": "S3" in selected_targets,
        "stable_subject_lead_rows": int(len(stable_subject_leads)),
        "failed_frontier_subject_lead_rows": int(len(failed_subject_leads)),
        "estimated_all_target_delta_vs_id342": estimated_all_target_delta,
        "projected_public_proxy_from_id342": projected_public_proxy,
        "gap_to_public_055_after_proxy": projected_public_proxy - PUBLIC_055,
        "selector_gate_pass": bool(selected_targets),
        "public_055_evidence": False,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "submit_now": False,
        "next_action": (
            "Stable public-scored donors help isolated subject/target slices but fail the fold-internal selector. "
            "Do not promote a selector or target-swap artifact; use this only to guide future representation work."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_q3s3_slice_selector_anatomy",
        "purpose": "Explain Q3/S3 stable-donor behavior under ID342 and test a fold-internal selector.",
        "target_swap_gate_report": rel(TARGET_SWAP_GATE_PATH),
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "validation": {
            "primary_splits": PRIMARY_SPLITS,
            "stable_donor_ids": STABLE_DONOR_IDS,
            "failed_frontier_ids": FAILED_FRONTIER_IDS,
            "selector_weights": SELECTOR_WEIGHTS,
            "target_win_floor": TARGET_WIN_FLOOR,
            "target_mean_delta_ceiling": TARGET_MEAN_DELTA_CEILING,
            "target_max_delta_ceiling": TARGET_MAX_DELTA_CEILING,
        },
        "target_score_table": target_scores,
        "subject_slice_rows": subject_rows,
        "top_stable_subject_leads": stable_subject_leads[:12],
        "top_failed_frontier_subject_leads": sorted(
            failed_subject_leads, key=lambda row: (row["target"], row["best_failed_frontier_delta_vs_id342"])
        )[:12],
        "selector_reports": selector_reports,
        "decision": decision,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action": True,
        "no_dacon_submission": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 Q3/S3 slice-selector anatomy",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Selector Gate",
        "",
        "| target | wins | mean delta | max delta | eligible |",
        "|---|---:|---:|---:|---|",
    ]
    for target, target_report in report["selector_reports"].items():
        lines.append(
            f"| `{target}` | {target_report['wins']} | {target_report['mean_delta']:.9f} | "
            f"{target_report['max_delta']:.9f} | `{target_report['eligible']}` |"
        )
    lines.extend(
        [
            "",
            "## Stable Subject Leads",
            "",
            "| target | subject | rows | best stable | delta vs ID342 | ID342 loss |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["top_stable_subject_leads"][:10]:
        lines.append(
            f"| `{row['target']}` | `{row['subject_id']}` | {row['rows']} | {row['best_stable_id']} | "
            f"{row['best_stable_delta_vs_id342']:.9f} | {row['id342_loss']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Stable selector selected targets: `{decision['stable_selector_selected_targets']}`",
            f"- Stable subject lead rows: `{decision['stable_subject_lead_rows']}`",
            f"- Failed-frontier subject lead rows: `{decision['failed_frontier_subject_lead_rows']}`",
            f"- Projected Public proxy: `{decision['projected_public_proxy_from_id342']:.9f}`",
            f"- Gap to Public 0.55: `{decision['gap_to_public_055_after_proxy']:.9f}`",
            f"- Kaggle now: `{decision['run_kaggle_now']}`",
            f"- DACON submit now: `{decision['submit_now']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.output)
    decision = report["decision"]
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(
        "Selector selected targets: "
        f"{decision['stable_selector_selected_targets']} | stable subject leads="
        f"{decision['stable_subject_lead_rows']} | failed subject leads="
        f"{decision['failed_frontier_subject_lead_rows']}"
    )


if __name__ == "__main__":
    main()
