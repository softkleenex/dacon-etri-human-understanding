#!/usr/bin/env python3
"""Cross-target consistency gate after ID342.

This report-only diagnostic tests whether ID342 can be improved by a tiny
second-stage consistency model that uses only the other ID342 target
predictions as features. It does not use same-row true sibling labels at test
time. Config selection is fold-internal for the frozen primary split rows.

It creates no submission CSV, base prediction artifact, notebook-history
artifact, submission-log row, Kaggle push/run, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_clean_id221_feature_similarity_smoke import (  # noqa: E402
    FAILURE_ID,
    GAP_ID,
    HISTORICAL_ID221,
    REFERENCE_ID,
    TARGETS,
    average_loss,
    binary_loss,
    label_frame,
    load_base,
    load_train_test,
    projection,
    split_rows,
)
from scripts.diagnose_post_id341_clean_id221_inner_selection_probe import make_inner_folds  # noqa: E402
from scripts.diagnose_post_id342_feature_similarity_source_decomposition import (  # noqa: E402
    ANCHOR_ID,
    ID342_PUBLIC,
    PUBLIC_055,
)
from scripts.diagnose_post_id342_raw_temporal_q3s3_sequence_image_smoke import PRIMARY_SPLITS  # noqa: E402

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_cross_target_consistency_gate_20260624.json"

FOCUS_TARGETS = ["Q2", "Q3", "S3", "S4"]
C_VALUES = [0.2]
BLEND_WEIGHTS = [0.01, 0.03, 0.05]

MIN_PRIMARY_WINS_VS_ID342 = 4
MAX_PRIMARY_MEAN_DELTA_VS_ID342 = -0.00050
MAX_PRIMARY_WORST_DELTA_VS_ID342 = 0.0
MAX_PUBLIC_PROXY_FOR_ESCALATION = 0.5920
MAX_GAP_TO_PUBLIC_055_FOR_ESCALATION = 0.0400
MAX_TEST_MEAN_MOVEMENT_VS_ID342 = 0.0100
MAX_TEST_MAX_MOVEMENT_VS_ID342 = 0.0800


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def selected_split_rows() -> list[dict[str, Any]]:
    return [fold for fold in split_rows() if fold["split_name"] in PRIMARY_SPLITS]


def logit(values: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(arr / (1.0 - arr))


def build_meta_features(predictions: pd.DataFrame, target: str) -> pd.DataFrame:
    out = pd.DataFrame(index=predictions.index)
    other_targets = [candidate for candidate in TARGETS if candidate != target]
    for other in other_targets:
        values = predictions[other].astype(float).to_numpy()
        out[f"pred_{other}"] = values
        out[f"logit_{other}"] = logit(values)
    q_others = [f"pred_{target_name}" for target_name in ["Q1", "Q2", "Q3"] if target_name != target]
    s_others = [f"pred_{target_name}" for target_name in ["S1", "S2", "S3", "S4"] if target_name != target]
    if q_others:
        out["q_other_mean"] = out[q_others].mean(axis=1)
        out["q_other_std"] = out[q_others].std(axis=1).fillna(0.0)
    if s_others:
        out["s_other_mean"] = out[s_others].mean(axis=1)
        out["s_other_std"] = out[s_others].std(axis=1).fillna(0.0)
    if "pred_Q2" in out and "pred_Q3" in out:
        out["q2_q3_product"] = out["pred_Q2"] * out["pred_Q3"]
    if "pred_S3" in out and "pred_S4" in out:
        out["s3_s4_product"] = out["pred_S3"] * out["pred_S4"]
    return out


def candidate_grid() -> list[dict[str, Any]]:
    configs = [{"name": "base", "kind": "base"}]
    for c_value in C_VALUES:
        for blend_weight in BLEND_WEIGHTS:
            configs.append(
                {
                    "name": f"lr_c{c_value:g}_w{blend_weight:g}",
                    "c": c_value,
                    "blend_weight": blend_weight,
                }
            )
    return configs


def fit_predict_lr(train_x: pd.DataFrame, train_y: np.ndarray, query_x: pd.DataFrame, c_value: float) -> np.ndarray:
    if len(np.unique(train_y)) < 2:
        model = DummyClassifier(strategy="prior")
        model.fit(np.zeros((len(train_y), 1)), train_y)
        return np.full(len(query_x), float(np.mean(train_y)), dtype=float)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=c_value, max_iter=1000, random_state=342),
    )
    model.fit(train_x, train_y)
    return np.clip(model.predict_proba(query_x)[:, 1], 1e-6, 1.0 - 1e-6)


def blended_values(base_values: np.ndarray, raw_values: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    weight = float(config["blend_weight"])
    return np.clip((1.0 - weight) * base_values + weight * raw_values, 1e-6, 1.0 - 1e-6)


def select_config_for_outer_train(features: pd.DataFrame, base_oof, train, target: str, outer_train_idx: np.ndarray):
    inner_folds = make_inner_folds(train, outer_train_idx)
    y_outer = train.iloc[outer_train_idx][target].astype(int).to_numpy()
    base_outer = base_oof.iloc[outer_train_idx][target].astype(float).to_numpy()
    base_loss = binary_loss(y_outer, base_outer)
    index_to_pos = {int(row_idx): pos for pos, row_idx in enumerate(outer_train_idx)}
    candidate_rows: list[dict[str, Any]] = []
    for config in candidate_grid():
        if config.get("kind") == "base":
            pred = base_outer.copy()
        else:
            pred = np.zeros(len(outer_train_idx), dtype=float)
            for inner_train_idx, inner_valid_idx in inner_folds:
                raw = fit_predict_lr(
                    features.iloc[inner_train_idx],
                    train.iloc[inner_train_idx][target].astype(int).to_numpy(),
                    features.iloc[inner_valid_idx],
                    float(config["c"]),
                )
                base_values = base_oof.iloc[inner_valid_idx][target].astype(float).to_numpy()
                values = blended_values(base_values, raw, config)
                for row_idx, value in zip(inner_valid_idx, values, strict=True):
                    pred[index_to_pos[int(row_idx)]] = float(value)
        loss = binary_loss(y_outer, pred)
        candidate_rows.append({"name": config["name"], "loss": loss, "delta_vs_base": loss - base_loss, "config": config})
    candidate_rows = sorted(candidate_rows, key=lambda row: row["loss"])
    return candidate_rows[0]["config"], candidate_rows[:8]


def outer_predictions(train, base_oof):
    pred = base_oof[TARGETS].astype(float).to_numpy().copy()
    fold_details = []
    selected_counts = Counter()
    selected_by_target = defaultdict(Counter)
    feature_by_target = {target: build_meta_features(base_oof, target) for target in FOCUS_TARGETS}
    for fold in selected_split_rows():
        outer_train_idx = np.asarray(fold["train_idx"], dtype=int)
        outer_valid_idx = np.asarray(fold["valid_idx"], dtype=int)
        target_losses = {}
        target_rows = {}
        for target_idx, target in enumerate(TARGETS):
            if target not in FOCUS_TARGETS:
                values = base_oof.iloc[outer_valid_idx][target].astype(float).to_numpy()
                selected = {"name": "base", "kind": "base"}
                top_rows = []
            else:
                features = feature_by_target[target]
                selected, top_rows = select_config_for_outer_train(features, base_oof, train, target, outer_train_idx)
                if selected.get("kind") == "base":
                    values = base_oof.iloc[outer_valid_idx][target].astype(float).to_numpy()
                else:
                    raw = fit_predict_lr(
                        features.iloc[outer_train_idx],
                        train.iloc[outer_train_idx][target].astype(int).to_numpy(),
                        features.iloc[outer_valid_idx],
                        float(selected["c"]),
                    )
                    base_values = base_oof.iloc[outer_valid_idx][target].astype(float).to_numpy()
                    values = blended_values(base_values, raw, selected)
            pred[outer_valid_idx, target_idx] = values
            selected_name = str(selected["name"])
            selected_counts[selected_name] += 1
            selected_by_target[target][selected_name] += 1
            target_losses[target] = binary_loss(train.iloc[outer_valid_idx][target].to_numpy(), values)
            target_rows[target] = {"selected": selected, "inner_top": top_rows, "outer_valid_loss": target_losses[target]}
        fold_details.append(
            {
                "split_name": fold["split_name"],
                "fold_id": int(fold["fold_id"]),
                "valid_n": int(len(outer_valid_idx)),
                "loss": float(np.mean(list(target_losses.values()))),
                "target_rows": target_rows,
            }
        )
    return pred, fold_details, selected_counts, selected_by_target


def full_train_test_preview(train, base_oof, base_test):
    pred = base_test[TARGETS].astype(float).to_numpy().copy()
    selected_configs = {}
    fit_idx = np.arange(len(train))
    for target_idx, target in enumerate(TARGETS):
        if target not in FOCUS_TARGETS:
            selected_configs[target] = {"selected": {"name": "base", "kind": "base"}, "inner_top": []}
            continue
        train_features = build_meta_features(base_oof, target)
        test_features = build_meta_features(base_test, target)
        selected, top_rows = select_config_for_outer_train(train_features, base_oof, train, target, fit_idx)
        selected_configs[target] = {"selected": selected, "inner_top": top_rows}
        if selected.get("kind") != "base":
            raw = fit_predict_lr(
                train_features,
                train[target].astype(int).to_numpy(),
                test_features,
                float(selected["c"]),
            )
            base_values = base_test[target].astype(float).to_numpy()
            pred[:, target_idx] = blended_values(base_values, raw, selected)
    return pred, selected_configs


def split_summary(
    fold_details: list[dict[str, Any]], y_true: np.ndarray, candidate: np.ndarray, references: dict[str, np.ndarray]
) -> dict[str, Any]:
    by_split: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for fold, detail in zip(selected_split_rows(), fold_details, strict=True):
        valid_idx = np.asarray(fold["valid_idx"], dtype=int)
        by_split[detail["split_name"]]["loss"].append(float(detail["loss"]))
        for name, ref in references.items():
            delta = float(detail["loss"] - average_loss(y_true[valid_idx], ref[valid_idx]))
            by_split[detail["split_name"]][f"delta_vs_{name}"].append(delta)
    split_out = {
        split: {
            "mean_loss": float(np.mean(values["loss"])),
            **{key: float(np.mean(value)) for key, value in values.items() if key.startswith("delta_vs_")},
        }
        for split, values in by_split.items()
    }
    primary_deltas = [split_out[name]["delta_vs_id342"] for name in PRIMARY_SPLITS if name in split_out]
    return {
        "by_split": split_out,
        "primary_delta_vs_id342": {
            "wins": int(sum(delta < 0 for delta in primary_deltas)),
            "mean": float(np.mean(primary_deltas)) if primary_deltas else 0.0,
            "max": float(np.max(primary_deltas)) if primary_deltas else 0.0,
        },
    }


def target_loss_summary(y_true: np.ndarray, candidate: np.ndarray, references: dict[str, np.ndarray]) -> dict[str, Any]:
    rows = {}
    for target_idx, target in enumerate(TARGETS):
        loss = binary_loss(y_true[:, target_idx], candidate[:, target_idx])
        rows[target] = {
            "loss": loss,
            **{
                f"delta_vs_{name}": loss - binary_loss(y_true[:, target_idx], ref[:, target_idx])
                for name, ref in references.items()
            },
        }
    return rows


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    train, test = load_train_test()
    labels = label_frame(train)
    y_true = train[TARGETS].astype(int).to_numpy()
    base_oof = load_base("oof", ANCHOR_ID, labels)
    base_test = load_base("test", ANCHOR_ID)
    references_oof = {
        "id178": load_base("oof", REFERENCE_ID, labels)[TARGETS].astype(float).to_numpy(),
        "id221": load_base("oof", HISTORICAL_ID221, labels)[TARGETS].astype(float).to_numpy(),
        "id342": base_oof[TARGETS].astype(float).to_numpy(),
        "id287": load_base("oof", FAILURE_ID, labels)[TARGETS].astype(float).to_numpy(),
        "id329": load_base("oof", GAP_ID, labels)[TARGETS].astype(float).to_numpy(),
    }
    candidate_oof, fold_details, selected_counts, selected_by_target = outer_predictions(train, base_oof)
    candidate_test, full_test_selection = full_train_test_preview(train, base_oof, base_test)
    full_loss = average_loss(y_true, candidate_oof)
    id342_loss = average_loss(y_true, references_oof["id342"])
    split_eval = split_summary(fold_details, y_true, candidate_oof, references_oof)
    target_eval = target_loss_summary(y_true, candidate_oof, references_oof)
    id342_test = base_test[TARGETS].astype(float).to_numpy()
    movement = candidate_test - id342_test
    failed_direction = references_oof["id287"].reshape(-1) - references_oof["id178"].reshape(-1)
    gap_direction = references_oof["id329"].reshape(-1) - references_oof["id178"].reshape(-1)
    candidate_direction = candidate_oof.reshape(-1) - references_oof["id178"].reshape(-1)
    public_proxy = ID342_PUBLIC + float(full_loss - id342_loss)
    gate_checks = {
        "primary_wins_vs_id342": split_eval["primary_delta_vs_id342"]["wins"] >= MIN_PRIMARY_WINS_VS_ID342,
        "primary_mean_delta_material": (
            split_eval["primary_delta_vs_id342"]["mean"] <= MAX_PRIMARY_MEAN_DELTA_VS_ID342
        ),
        "primary_worst_delta_not_positive": (
            split_eval["primary_delta_vs_id342"]["max"] <= MAX_PRIMARY_WORST_DELTA_VS_ID342
        ),
        "public_proxy_low_enough": public_proxy <= MAX_PUBLIC_PROXY_FOR_ESCALATION,
        "public_055_gap_low_enough": (public_proxy - PUBLIC_055) <= MAX_GAP_TO_PUBLIC_055_FOR_ESCALATION,
        "test_mean_movement_vs_id342_within_ceiling": (
            float(np.mean(np.abs(movement))) <= MAX_TEST_MEAN_MOVEMENT_VS_ID342
        ),
        "test_max_movement_vs_id342_within_ceiling": (
            float(np.max(np.abs(movement))) <= MAX_TEST_MAX_MOVEMENT_VS_ID342
        ),
    }
    gate_pass = bool(all(gate_checks.values()))
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_cross_target_consistency_gate",
        "purpose": "Test tiny second-stage consistency models using only other ID342 target predictions.",
        "focus_targets": FOCUS_TARGETS,
        "grid": {
            "c_values": C_VALUES,
            "blend_weights": BLEND_WEIGHTS,
            "evaluated_split_names": PRIMARY_SPLITS,
        },
        "full_oof": {
            "candidate_loss": full_loss,
            "id342_loss": id342_loss,
            "delta_vs_id342": float(full_loss - id342_loss),
            "delta_vs_id178": float(full_loss - average_loss(y_true, references_oof["id178"])),
            "delta_vs_id221": float(full_loss - average_loss(y_true, references_oof["id221"])),
        },
        "split_summary": split_eval,
        "target_summary": target_eval,
        "selection_summary": {
            "selected_counts": dict(selected_counts.most_common()),
            "selected_by_target": {
                target: dict(counter.most_common()) for target, counter in selected_by_target.items()
            },
            "full_train_test_selection": full_test_selection,
        },
        "test_geometry": {
            "mean_abs_delta_vs_id342": float(np.mean(np.abs(movement))),
            "max_abs_delta_vs_id342": float(np.max(np.abs(movement))),
            "per_target_mean_abs_delta_vs_id342": {
                target: float(np.mean(np.abs(movement[:, idx]))) for idx, target in enumerate(TARGETS)
            },
            "projection_onto_id287_failed_direction": projection(candidate_direction, failed_direction),
            "projection_onto_id329_gap_direction": projection(candidate_direction, gap_direction),
        },
        "public_proxy": {
            "id342_public": ID342_PUBLIC,
            "projected_public_proxy": public_proxy,
            "gap_to_public_055": float(public_proxy - PUBLIC_055),
        },
        "decision": {
            "cross_target_consistency_gate_pass": gate_pass,
            "checks": gate_checks,
            "artifact_review_allowed": False,
            "create_submission_artifact": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "submit_now": False,
            "reason": (
                "Mechanical gate passed, but this script is report-only; require a separate decision note before "
                "any artifact/Kaggle action."
                if gate_pass
                else "Cross-target consistency does not clear the strict ID342/Public-0.55 gate."
            ),
        },
        "no_prediction_file_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    full = report["full_oof"]
    primary = report["split_summary"]["primary_delta_vs_id342"]
    proxy = report["public_proxy"]
    decision = report["decision"]
    lines = [
        "# Post-ID342 cross-target consistency gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.",
        "",
        "## OOF Summary",
        "",
        f"- Candidate loss: `{full['candidate_loss']:.6f}`",
        f"- ID342 loss: `{full['id342_loss']:.6f}`",
        f"- Delta vs ID342: `{full['delta_vs_id342']:+.6f}`",
        f"- Delta vs ID178: `{full['delta_vs_id178']:+.6f}`",
        f"- Delta vs ID221: `{full['delta_vs_id221']:+.6f}`",
        f"- Primary wins vs ID342: `{primary['wins']}`",
        f"- Primary mean/max delta vs ID342: `{primary['mean']:+.6f}` / `{primary['max']:+.6f}`",
        f"- Projected public proxy: `{proxy['projected_public_proxy']:.9f}`",
        f"- Gap to Public 0.55: `{proxy['gap_to_public_055']:+.9f}`",
        "",
        "## Split Gate",
        "",
        "| split | loss | delta vs ID342 | delta vs ID178 | delta vs ID221 |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, row in report["split_summary"]["by_split"].items():
        lines.append(
            "| {split} | {loss:.6f} | {d342:+.6f} | {d178:+.6f} | {d221:+.6f} |".format(
                split=split,
                loss=row["mean_loss"],
                d342=row["delta_vs_id342"],
                d178=row["delta_vs_id178"],
                d221=row["delta_vs_id221"],
            )
        )
    lines.extend(
        [
            "",
            "## Selection Summary",
            "",
            "| target | top selections | full-train/test selected |",
            "|---|---|---|",
        ]
    )
    for target in TARGETS:
        selected = report["selection_summary"]["selected_by_target"].get(target, {})
        selected_str = ", ".join(f"{name}:{count}" for name, count in list(selected.items())[:5])
        full_selected = report["selection_summary"]["full_train_test_selection"][target]["selected"]["name"]
        lines.append(f"| `{target}` | `{selected_str}` | `{full_selected}` |")
    failed = [name for name, ok in decision["checks"].items() if not ok]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Gate pass: `{decision['cross_target_consistency_gate_pass']}`",
            f"- Failed checks: `{','.join(failed)}`",
            f"- Artifact review allowed: `{decision['artifact_review_allowed']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Submit now: `{decision['submit_now']}`",
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
