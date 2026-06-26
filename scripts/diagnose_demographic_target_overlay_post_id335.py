#!/usr/bin/env python3
"""Report-only target-specific demographic overlay scan after ID335.

The previous gate allowed demographic/static research only for Q2/S2/S4. This
diagnostic tests whether those demographic models add a tiny orthogonal signal
on top of ID335 without creating submission artifacts.
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
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_strict_demographic_feature_audit import (  # noqa: E402
    Candidate,
    build_candidates,
    make_loso_folds,
    predict_candidate,
    prepare_frames,
)
from scripts.diagnose_target_specific_demographic_gate_post_id335 import target_demo_delta  # noqa: E402
from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_strict_raw_feature_experiment import REPORTS_DIR, TARGETS  # noqa: E402
from src.utils.validation import make_subject_hole_folds  # noqa: E402

KST = timezone(timedelta(hours=9))
EPS = 1e-6
DEFAULT_ALPHA_GRID = [0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]


def target_loss(y: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y.astype(int), np.clip(pred, EPS, 1.0 - EPS), labels=[0, 1]))


def blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip((1.0 - alpha) * base + alpha * overlay, EPS, 1.0 - EPS)


def fit_full_test_prediction(train: pd.DataFrame, test: pd.DataFrame, target: str, candidate: Candidate) -> np.ndarray:
    y = train[target].astype(int).reset_index(drop=True)
    if y.nunique() < 2:
        model = DummyClassifier(strategy="prior")
    else:
        model = clone(candidate.model)
    model.fit(train[candidate.feature_cols], y)
    return np.clip(model.predict_proba(test[candidate.feature_cols])[:, 1], EPS, 1.0 - EPS)


def candidate_lookup(candidates: list[Candidate]) -> dict[str, Candidate]:
    return {candidate.name: candidate for candidate in candidates}


def demo_candidate_names(audit: dict[str, Any], target: str) -> list[str]:
    subject_hole = audit["splits"]["subject_hole_temporal"]
    loso = audit["splits"]["leave_one_subject_out_stress"]
    names = [
        target_demo_delta(subject_hole, target)["best_demo_candidate"],
        target_demo_delta(loso, target)["best_demo_candidate"],
    ]
    return list(dict.fromkeys(str(name) for name in names))


def geometry(
    test_pred: pd.DataFrame, id178_test: pd.DataFrame, id287_test: pd.DataFrame, id329_test: pd.DataFrame
) -> dict[str, float]:
    arr = test_pred[TARGETS].astype(float).to_numpy()
    a178 = id178_test[TARGETS].astype(float).to_numpy()
    a287 = id287_test[TARGETS].astype(float).to_numpy()
    a329 = id329_test[TARGETS].astype(float).to_numpy()
    failed_vec = (a287 - a178).reshape(-1)
    vec = (arr - a178).reshape(-1)
    denom = float(np.dot(failed_vec, failed_vec))
    projection_failed = float(np.dot(vec, failed_vec) / denom) if denom > 0 else 0.0
    return {
        "mean_abs_delta_vs_id178": float(np.mean(np.abs(arr - a178))),
        "max_abs_delta_vs_id178": float(np.max(np.abs(arr - a178))),
        "rmse_vs_id178": float(np.sqrt(np.mean((arr - a178) ** 2))),
        "mean_abs_delta_vs_id329": float(np.mean(np.abs(arr - a329))),
        "max_abs_delta_vs_id329": float(np.max(np.abs(arr - a329))),
        "projection_to_id287_failed_direction": projection_failed,
        "prediction_mean": float(np.mean(arr)),
        "prediction_std": float(np.std(arr)),
    }


def scan_target_candidate(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    candidate: Candidate,
    subject_hole_folds: list[tuple[np.ndarray, np.ndarray]],
    loso_folds: list[tuple[np.ndarray, np.ndarray]],
    id335_oof: pd.DataFrame,
    id335_test: pd.DataFrame,
    alpha_grid: list[float],
) -> dict[str, Any]:
    y = train[target].astype(int).to_numpy()
    base_oof = np.clip(id335_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
    base_test = np.clip(id335_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
    sh_oof, sh_details = predict_candidate(train, target, candidate, subject_hole_folds)
    loso_oof, loso_details = predict_candidate(train, target, candidate, loso_folds)
    full_test = fit_full_test_prediction(train, test, target, candidate)
    baseline_loss = target_loss(y, base_oof)
    rows = []
    for alpha in alpha_grid:
        sh_pred = blend(base_oof, sh_oof, alpha)
        loso_pred = blend(base_oof, loso_oof, alpha)
        test_pred = blend(base_test, full_test, alpha)
        rows.append(
            {
                "alpha": float(alpha),
                "subject_hole_blend_loss": target_loss(y, sh_pred),
                "loso_blend_loss": target_loss(y, loso_pred),
                "mean_stress_loss": float(np.mean([target_loss(y, sh_pred), target_loss(y, loso_pred)])),
                "test_max_abs_delta_vs_id335": float(np.max(np.abs(test_pred - base_test))),
                "test_mean_abs_delta_vs_id335": float(np.mean(np.abs(test_pred - base_test))),
                "test_prediction_mean": float(np.mean(test_pred)),
            }
        )
    guarded = [
        row
        for row in rows
        if row["subject_hole_blend_loss"] < baseline_loss
        and row["loso_blend_loss"] <= baseline_loss + 0.00025
        and row["test_max_abs_delta_vs_id335"] <= 0.035
    ]
    best_guarded = (
        min(
            guarded,
            key=lambda row: (
                row["subject_hole_blend_loss"],
                row["loso_blend_loss"],
                row["test_max_abs_delta_vs_id335"],
            ),
        )
        if guarded
        else None
    )
    best_mean_stress = min(
        rows,
        key=lambda row: (
            row["mean_stress_loss"],
            row["test_max_abs_delta_vs_id335"],
        ),
    )
    return {
        "target": target,
        "candidate": candidate.name,
        "model": candidate.model_name,
        "n_features": int(len(candidate.feature_cols)),
        "baseline_id335_loss": baseline_loss,
        "demo_subject_hole_loss": target_loss(y, sh_oof),
        "demo_loso_loss": target_loss(y, loso_oof),
        "rows": rows,
        "best_guarded": best_guarded,
        "best_mean_stress": best_mean_stress,
        "fold_details": {
            "subject_hole": sh_details,
            "loso": loso_details,
        },
    }


def select_target_rows(scans: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, list[dict[str, Any]]] = {}
    for scan in scans:
        by_target.setdefault(scan["target"], []).append(scan)
    selected: dict[str, Any] = {}
    for target, rows in by_target.items():
        guarded = [row for row in rows if row["best_guarded"] is not None]
        if guarded:
            best = min(
                guarded,
                key=lambda row: (
                    row["best_guarded"]["subject_hole_blend_loss"],
                    row["best_guarded"]["loso_blend_loss"],
                    row["best_guarded"]["test_max_abs_delta_vs_id335"],
                ),
            )
            selected[target] = {
                "decision": "allow_tiny_overlay_hypothesis",
                "candidate": best["candidate"],
                "alpha": float(best["best_guarded"]["alpha"]),
                "baseline_id335_loss": float(best["baseline_id335_loss"]),
                "subject_hole_blend_loss": float(best["best_guarded"]["subject_hole_blend_loss"]),
                "loso_blend_loss": float(best["best_guarded"]["loso_blend_loss"]),
                "test_max_abs_delta_vs_id335": float(best["best_guarded"]["test_max_abs_delta_vs_id335"]),
                "test_mean_abs_delta_vs_id335": float(best["best_guarded"]["test_mean_abs_delta_vs_id335"]),
            }
        else:
            best = min(
                rows,
                key=lambda row: (
                    row["best_mean_stress"]["mean_stress_loss"],
                    row["best_mean_stress"]["test_max_abs_delta_vs_id335"],
                ),
            )
            selected[target] = {
                "decision": "reject_no_guarded_alpha",
                "candidate": best["candidate"],
                "alpha": float(best["best_mean_stress"]["alpha"]),
                "baseline_id335_loss": float(best["baseline_id335_loss"]),
                "subject_hole_blend_loss": float(best["best_mean_stress"]["subject_hole_blend_loss"]),
                "loso_blend_loss": float(best["best_mean_stress"]["loso_blend_loss"]),
                "test_max_abs_delta_vs_id335": float(best["best_mean_stress"]["test_max_abs_delta_vs_id335"]),
                "test_mean_abs_delta_vs_id335": float(best["best_mean_stress"]["test_mean_abs_delta_vs_id335"]),
            }
    return selected


def build_combined_predictions(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected: dict[str, Any],
    candidate_by_name: dict[str, Candidate],
    subject_hole_folds: list[tuple[np.ndarray, np.ndarray]],
    id335_oof: pd.DataFrame,
    id335_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    oof = id335_oof[["subject_id", "date", *TARGETS]].copy()
    test_pred = id335_test[["subject_id", "date", *TARGETS]].copy()
    target_losses: dict[str, float] = {}
    for target in TARGETS:
        if target in selected and selected[target]["decision"] == "allow_tiny_overlay_hypothesis":
            candidate = candidate_by_name[selected[target]["candidate"]]
            alpha = float(selected[target]["alpha"])
            sh_oof, _details = predict_candidate(train, target, candidate, subject_hole_folds)
            full_test = fit_full_test_prediction(train, test, target, candidate)
            oof[target] = blend(id335_oof[target].astype(float).to_numpy(), sh_oof, alpha)
            test_pred[target] = blend(id335_test[target].astype(float).to_numpy(), full_test, alpha)
        target_losses[target] = target_loss(train[target].astype(int).to_numpy(), oof[target].astype(float).to_numpy())
    return oof, test_pred, target_losses


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Demographic target overlay scan — post ID335",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. No submission CSV, notebook, base artifact, submission-log row, or DACON submission was created.",
        "",
        "## Selected target decisions",
        "",
        "| target | decision | candidate | alpha | ID335 loss | SH blend | LOSO blend | max Δtest335 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for target, row in report["selected_target_rows"].items():
        lines.append(
            f"| {target} | `{row['decision']}` | `{row['candidate']}` | {row['alpha']:.3f} | "
            f"{row['baseline_id335_loss']:.6f} | {row['subject_hole_blend_loss']:.6f} | "
            f"{row['loso_blend_loss']:.6f} | {row['test_max_abs_delta_vs_id335']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Combined guarded overlay",
            "",
            f"- Combined local OOF CV: `{report['combined']['local_cv']:.6f}`",
            f"- ID335 local OOF CV: `{report['id335_local_cv']:.6f}`",
            f"- Geometry max Δ178: `{report['combined']['geometry']['max_abs_delta_vs_id178']:.6f}`",
            f"- Failed-vector projection: `{report['combined']['geometry']['projection_to_id287_failed_direction']:+.4f}`",
            "",
            "## Verdict",
            "",
            report["decision"]["reason"],
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    output_json: Path,
    gate_json: Path,
    audit_json: Path,
    alpha_grid: list[float],
    n_estimators: int,
) -> dict[str, Any]:
    train, test = prepare_frames()
    gate = json.loads(gate_json.read_text())
    audit = json.loads(audit_json.read_text())
    allowed_targets = [target for target in gate["allow_research_targets"] if target in TARGETS]
    id335_oof, id335_test, id335_source = load_artifact_pair(train, test, 335)
    _id178_oof, id178_test, id178_source = load_artifact_pair(train, test, 178)
    _id287_oof, id287_test, id287_source = load_artifact_pair(train, test, 287)
    _id329_oof, id329_test, id329_source = load_artifact_pair(train, test, 329)

    candidates = build_candidates(train, test, seed=42, n_estimators=n_estimators)
    candidates_by_name = candidate_lookup(candidates)
    subject_hole_folds = make_subject_hole_folds(train, n_folds=5)
    loso_folds = make_loso_folds(train)
    scans = []
    for target in allowed_targets:
        for candidate_name in demo_candidate_names(audit, target):
            scans.append(
                scan_target_candidate(
                    train=train,
                    test=test,
                    target=target,
                    candidate=candidates_by_name[candidate_name],
                    subject_hole_folds=subject_hole_folds,
                    loso_folds=loso_folds,
                    id335_oof=id335_oof,
                    id335_test=id335_test,
                    alpha_grid=alpha_grid,
                )
            )

    selected = select_target_rows(scans)
    combined_oof, combined_test, target_losses = build_combined_predictions(
        train=train,
        test=test,
        selected=selected,
        candidate_by_name=candidates_by_name,
        subject_hole_folds=subject_hole_folds,
        id335_oof=id335_oof,
        id335_test=id335_test,
    )
    id335_target_losses = {
        target: target_loss(train[target].astype(int).to_numpy(), id335_oof[target].astype(float).to_numpy())
        for target in TARGETS
    }
    local_cv = float(np.mean(list(target_losses.values())))
    id335_local_cv = float(np.mean(list(id335_target_losses.values())))
    promoted_targets = [
        target for target, row in selected.items() if row["decision"] == "allow_tiny_overlay_hypothesis"
    ]
    test_geom = geometry(combined_test, id178_test, id287_test, id329_test)
    promote = bool(
        promoted_targets
        and local_cv < id335_local_cv - 0.0002
        and test_geom["max_abs_delta_vs_id178"] <= 0.125
        and test_geom["projection_to_id287_failed_direction"] <= 0.006
    )
    reason = (
        "A tiny target-specific demographic overlay is reviewable as a generated_unsubmitted artifact, but it is still not Public 0.58-plausible and needs full artifact promotion before any submission review."
        if promote
        else "No demographic overlay is promoted. The branch remains useful as target-level research, but the guarded alpha evidence is too weak or too risky for a new artifact."
    )
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": "demographic_target_overlay_post_id335",
        "sources": {
            "gate": str(gate_json.relative_to(ROOT)),
            "audit": str(audit_json.relative_to(ROOT)),
            "id335": id335_source,
            "id178": id178_source,
            "id287": id287_source,
            "id329": id329_source,
        },
        "allowed_targets": allowed_targets,
        "alpha_grid": alpha_grid,
        "n_estimators": n_estimators,
        "scans": scans,
        "selected_target_rows": selected,
        "id335_target_losses": id335_target_losses,
        "id335_local_cv": id335_local_cv,
        "combined": {
            "promoted_targets": promoted_targets,
            "local_cv": local_cv,
            "target_losses": target_losses,
            "geometry": test_geom,
        },
        "decision": {
            "promote_generated_unsubmitted_artifact": promote,
            "public_058_plausible": False,
            "reason": reason,
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_json.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_DIR / "demographic_target_overlay_post_id335_20260617.json",
    )
    parser.add_argument(
        "--gate-json",
        type=Path,
        default=REPORTS_DIR / "target_specific_demographic_gate_post_id335_20260617.json",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=REPORTS_DIR / "strict_demographic_feature_audit_kaggle_20260617.json",
    )
    parser.add_argument("--alpha-grid", default="0,0.01,0.025,0.05,0.075,0.10,0.15,0.20")
    parser.add_argument("--n-estimators", type=int, default=160)
    args = parser.parse_args()
    report = run(
        output_json=args.output_json,
        gate_json=args.gate_json,
        audit_json=args.audit_json,
        alpha_grid=[float(value) for value in args.alpha_grid.split(",")],
        n_estimators=args.n_estimators,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(json.dumps(report["decision"] | report["combined"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
