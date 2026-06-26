#!/usr/bin/env python3
"""Paper-aligned feature-similarity rebuild gate after ID342.

This report-only diagnostic tests whether the post-ID342 feature-similarity
route has a domain-aligned next step beyond ID221 alpha tweaking. It restricts
same-subject feature-similarity anchors by target according to the ETRI/DACON
documentation and the ETRI Lifelog 2024 paper:

- Q1/S3: movement/step and calendar context.
- Q2/Q3: mobile app/screen usage and calendar context.
- S4: HR/light/activity/calendar context.

All config selection is fold-internal: every outer fold selects from the
allowed target-specific grid using only its outer-training rows, then scores
the held-out outer-valid rows. It creates no prediction CSV, no base artifact,
no notebook-history artifact, no submission-log row, no Kaggle run, and no
DACON submission.
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
    feature_families,
    label_frame,
    load_base,
    load_train_test,
    projection,
    split_rows,
)
from scripts.diagnose_post_id341_clean_id221_inner_selection_probe import (  # noqa: E402
    make_inner_folds,
    predict_config,
)
from scripts.diagnose_post_id342_feature_similarity_source_decomposition import (  # noqa: E402
    ANCHOR_ID,
    ID342_PUBLIC,
    PUBLIC_055,
)
from scripts.diagnose_post_id342_raw_temporal_q3s3_sequence_image_smoke import PRIMARY_SPLITS  # noqa: E402

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_paper_aligned_feature_similarity_rebuild_gate_20260624.json"

FOCUS_TARGETS = ["Q1", "Q2", "Q3", "S3", "S4"]
TARGET_FAMILY_MAP: dict[str, list[str]] = {
    "Q1": ["move_gps", "date_time"],
    "Q2": ["screen_app", "date_time"],
    "Q3": ["screen_app", "date_time"],
    "S1": [],
    "S2": [],
    "S3": ["move_gps", "date_time", "light_hr"],
    "S4": ["light_hr", "move_gps", "date_time"],
}

K_VALUES = [1]
DATE_WEIGHTS = [0.0, 0.03]
BLEND_WEIGHTS = [0.01, 0.03]

MIN_PRIMARY_WINS_VS_ID342 = 4
MAX_PRIMARY_MEAN_DELTA_VS_ID342 = -0.00050
MAX_PRIMARY_WORST_DELTA_VS_ID342 = 0.0
MAX_PUBLIC_PROXY_FOR_ESCALATION = 0.5920
MAX_GAP_TO_PUBLIC_055_FOR_ESCALATION = 0.0400
MAX_TEST_MEAN_MOVEMENT_VS_ID342 = 0.0100
MAX_TEST_MAX_MOVEMENT_VS_ID342 = 0.0800

RESEARCH_PRIORS = [
    {
        "source": "DACON 4th ETRI Human Understanding AI Paper Challenge data page",
        "url": "https://dacon.io/en/competitions/official/236468/data",
        "read": (
            "The competition data items include smartphone charge/activity/ambience/BLE/GPS/light/screen/app/Wi-Fi "
            "plus smartwatch HR/light/steps; Q3 is pre-sleep stress and S3 is SOL guideline adherence."
        ),
    },
    {
        "source": "ETRI Lifelog Dataset 2024 paper",
        "url": "https://arxiv.org/abs/2508.03698",
        "read": (
            "The paper reports that Q1 and S3 baseline performance was strongest with 3-hour step-count features, "
            "while Q2 and Q3 were strongest with daily mobile app usage; demographics/weekday helped all targets."
        ),
    },
]


def selected_split_rows() -> list[dict[str, Any]]:
    return [fold for fold in split_rows() if fold["split_name"] in PRIMARY_SPLITS]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def candidate_grid(target: str) -> list[dict[str, Any]]:
    configs = [{"name": "base", "kind": "base"}]
    for family in TARGET_FAMILY_MAP[target]:
        for k in K_VALUES:
            for date_weight in DATE_WEIGHTS:
                for blend_weight in BLEND_WEIGHTS:
                    configs.append(
                        {
                            "name": f"{family}_k{k}_dw{date_weight:g}_w{blend_weight:g}",
                            "family": family,
                            "k": k,
                            "date_weight": date_weight,
                            "blend_weight": blend_weight,
                            "prior": "paper_aligned",
                        }
                    )
    return configs


def select_config_for_outer_train(
    train,
    test,
    families: dict[str, list[str]],
    base_oof,
    target: str,
    outer_train_idx: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inner_folds = make_inner_folds(train, outer_train_idx)
    y_outer = train.iloc[outer_train_idx][target].astype(int).to_numpy()
    index_to_pos = {int(row_idx): pos for pos, row_idx in enumerate(outer_train_idx)}
    candidate_rows: list[dict[str, Any]] = []
    for config in candidate_grid(target):
        if config.get("kind") == "base":
            pred = base_oof.iloc[outer_train_idx][target].astype(float).to_numpy()
        else:
            pred = np.zeros(len(outer_train_idx), dtype=float)
            for inner_train_idx, inner_valid_idx in inner_folds:
                inner_pred = predict_config(
                    train=train,
                    test=test,
                    families=families,
                    base218=base_oof,
                    target=target,
                    known_idx=inner_train_idx,
                    query_idx=inner_valid_idx,
                    config=config,
                    query_is_test=False,
                )
                for row_idx, value in zip(inner_valid_idx, inner_pred, strict=True):
                    pred[index_to_pos[int(row_idx)]] = float(value)
        loss = binary_loss(y_outer, pred)
        candidate_rows.append(
            {
                "name": config["name"],
                "loss": loss,
                "delta_vs_base": loss
                - binary_loss(y_outer, base_oof.iloc[outer_train_idx][target].astype(float).to_numpy()),
                "config": config,
            }
        )
    candidate_rows = sorted(candidate_rows, key=lambda row: row["loss"])
    return candidate_rows[0]["config"], candidate_rows[:8]


def outer_predictions(train, test, base_oof, families: dict[str, list[str]]):
    pred = np.zeros((len(train), len(TARGETS)), dtype=float)
    fold_details = []
    selected_counts = Counter()
    selected_by_target = defaultdict(Counter)
    for fold in selected_split_rows():
        outer_train_idx = np.asarray(fold["train_idx"], dtype=int)
        outer_valid_idx = np.asarray(fold["valid_idx"], dtype=int)
        target_rows = {}
        for target_idx, target in enumerate(TARGETS):
            selected, top_rows = select_config_for_outer_train(
                train=train,
                test=test,
                families=families,
                base_oof=base_oof,
                target=target,
                outer_train_idx=outer_train_idx,
            )
            if selected.get("kind") == "base":
                values = base_oof.iloc[outer_valid_idx][target].astype(float).to_numpy()
            else:
                values = predict_config(
                    train=train,
                    test=test,
                    families=families,
                    base218=base_oof,
                    target=target,
                    known_idx=outer_train_idx,
                    query_idx=outer_valid_idx,
                    config=selected,
                    query_is_test=False,
                )
            pred[outer_valid_idx, target_idx] = values
            selected_name = str(selected["name"])
            selected_counts[selected_name] += 1
            selected_by_target[target][selected_name] += 1
            target_rows[target] = {
                "selected": selected,
                "outer_valid_loss": binary_loss(train.iloc[outer_valid_idx][target].to_numpy(), values),
                "inner_top": top_rows,
            }
        fold_details.append(
            {
                "split_name": fold["split_name"],
                "fold_id": int(fold["fold_id"]),
                "valid_n": int(len(outer_valid_idx)),
                "loss": float(np.mean([row["outer_valid_loss"] for row in target_rows.values()])),
                "target_rows": target_rows,
            }
        )
    return pred, fold_details, selected_counts, selected_by_target


def full_train_test_preview(train, test, base_oof, base_test, families: dict[str, list[str]]):
    pred = np.zeros((len(test), len(TARGETS)), dtype=float)
    selected_configs = {}
    fit_idx = np.arange(len(train))
    for target_idx, target in enumerate(TARGETS):
        selected, top_rows = select_config_for_outer_train(
            train=train,
            test=test,
            families=families,
            base_oof=base_oof,
            target=target,
            outer_train_idx=fit_idx,
        )
        selected_configs[target] = {"selected": selected, "inner_top": top_rows}
        if selected.get("kind") == "base":
            pred[:, target_idx] = base_test[target].astype(float).to_numpy()
        else:
            pred[:, target_idx] = predict_config(
                train=train,
                test=test,
                families=families,
                base218=base_test,
                target=target,
                known_idx=fit_idx,
                query_idx=None,
                config=selected,
                query_is_test=True,
            )
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
    primary_rows = [split_out[name] for name in PRIMARY_SPLITS if name in split_out]
    primary_deltas = [row["delta_vs_id342"] for row in primary_rows]
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
    families = feature_families(train, test)
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
    candidate_oof, fold_details, selected_counts, selected_by_target = outer_predictions(
        train=train,
        test=test,
        base_oof=base_oof,
        families=families,
    )
    candidate_test, full_test_selection = full_train_test_preview(
        train=train,
        test=test,
        base_oof=base_oof,
        base_test=base_test,
        families=families,
    )

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
        "diagnostic": "post_id342_paper_aligned_feature_similarity_rebuild_gate",
        "purpose": (
            "Test a target-wise paper-aligned feature-similarity rebuild on top of ID342 before any artifact/Kaggle "
            "consideration."
        ),
        "research_priors": RESEARCH_PRIORS,
        "target_family_map": TARGET_FAMILY_MAP,
        "grid": {
            "k_values": K_VALUES,
            "date_weights": DATE_WEIGHTS,
            "blend_weights": BLEND_WEIGHTS,
            "evaluated_split_names": PRIMARY_SPLITS,
            "note": (
                "Fast gate: S1/S2 remain ID342 base; evaluation uses the frozen primary split names after the "
                "full grid was too slow locally."
            ),
        },
        "references": {
            "base_anchor_id": ANCHOR_ID,
            "historical_feature_similarity_id": HISTORICAL_ID221,
            "previous_public_anchor_id": REFERENCE_ID,
            "failed_frontier_id": FAILURE_ID,
            "gap_probe_id": GAP_ID,
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
            "paper_aligned_feature_similarity_gate_pass": gate_pass,
            "checks": gate_checks,
            "artifact_review_allowed": False,
            "create_submission_artifact": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "submit_now": False,
            "reason": (
                "Mechanical paper-aligned gate passed, but this script is still report-only; require a separate "
                "decision note before any artifact/Kaggle action."
                if gate_pass
                else "The paper-aligned feature-similarity rebuild does not clear the strict ID342/Public-0.55 gate."
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
        "# Post-ID342 paper-aligned feature-similarity rebuild gate",
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
    )
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
            f"- Gate pass: `{decision['paper_aligned_feature_similarity_gate_pass']}`",
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
