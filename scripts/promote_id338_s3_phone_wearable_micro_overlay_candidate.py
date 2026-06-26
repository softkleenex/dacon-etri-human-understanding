#!/usr/bin/env python3
"""Promote the ID338 S3 phone/wearable micro-overlay candidate.

This creates a generated-unsubmitted full-contract artifact only. It never
submits to DACON.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_id179_vs_id335_sequence_probe_review import (  # noqa: E402
    EPS,
    geometry,
    robust_score,
    slice_summary,
)
from scripts.diagnose_post_id338_material_q3s3_geometry_gate import (  # noqa: E402
    fit_predict_test_source,
    load_source_train_test,
)
from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402
    ModelSpec,
    load_split_manifest,
    predict_target_for_split,
)
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    TARGETS,
    append_submission_log,
    build_feature_frames,
    map_predictions_to_original_sample,
    sha256_file,
    validate_submission,
)
from src.utils.validation import clip_probabilities  # noqa: E402

KST = timezone(timedelta(hours=9))
DEFAULT_EXPERIMENT_ID = 343
DEFAULT_BASE_ID = 338
DEFAULT_TARGET = "S3"
DEFAULT_SOURCE_GROUP = "phone_wearable_discordance"
DEFAULT_MODEL_NAME = "lr_c0_20"
DEFAULT_MODEL_C = 0.20
DEFAULT_WEIGHT = 0.005
PROBE_REPORT = REPORTS_DIR / "post_id338_material_q3s3_probe_20260623.json"
GEOMETRY_GATE_REPORT = REPORTS_DIR / "post_id338_material_q3s3_geometry_gate_20260623.json"
WEIGHT_GEOMETRY_SCAN_REPORT = REPORTS_DIR / "post_id338_s3_phone_wearable_weight_geometry_scan_20260624.json"
ID338_PUBLIC_SCORE = 0.5960832519


def find_base(kind: str, candidate_id: int) -> Path:
    paths = sorted(BASE_PREDICTION_DIR.glob(f"base_{kind}_id_{candidate_id}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing base_{kind} ID{candidate_id}")
    return paths[0]


def align_base_oof(path: Path, train: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(path).copy()
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    key = train[["subject_id", "date"]].copy()
    key["date"] = pd.to_datetime(key["date"]).dt.date
    merged = key.merge(pred, on=["subject_id", "date"], how="left", validate="one_to_one")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"{path} does not align with train rows.")
    return merged[["subject_id", "date", *TARGETS]]


def align_base_test(path: Path, test: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(path).copy()
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    key = test[["subject_id", "date"]].copy()
    key["date"] = pd.to_datetime(key["date"]).dt.date
    merged = key.merge(pred, on=["subject_id", "date"], how="left", validate="one_to_one")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"{path} does not align with test rows.")
    return merged[["subject_id", "date", *TARGETS]]


def target_loglosses(train: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    return {
        target: float(
            log_loss(
                train[target].astype(int),
                np.clip(pred[target].astype(float), EPS, 1.0 - EPS),
                labels=[0, 1],
            )
        )
        for target in TARGETS
    }


def max_delta_by_target(candidate: pd.DataFrame, base: pd.DataFrame) -> dict[str, float]:
    return {
        target: float(
            np.max(np.abs(candidate[target].astype(float).to_numpy() - base[target].astype(float).to_numpy()))
        )
        for target in TARGETS
    }


def mean_delta_by_target(candidate: pd.DataFrame, base: pd.DataFrame) -> dict[str, float]:
    return {
        target: float(
            np.mean(np.abs(candidate[target].astype(float).to_numpy() - base[target].astype(float).to_numpy()))
        )
        for target in TARGETS
    }


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_config_from_reports(target: str, weight: float) -> dict[str, Any]:
    probe = load_report(PROBE_REPORT)
    gate = load_report(GEOMETRY_GATE_REPORT)
    weight_scan = load_report(WEIGHT_GEOMETRY_SCAN_REPORT) if WEIGHT_GEOMETRY_SCAN_REPORT.exists() else {}
    selected = probe.get("selected_target_configs", {}).get(target)
    if float(weight) != DEFAULT_WEIGHT:
        selected = weight_scan.get("selected_config")
    if not selected:
        raise ValueError(f"No selected config for {target} at weight={weight:g}")
    if gate.get("decision", {}).get("geometry_gate_pass") is not True:
        raise ValueError("Geometry gate has not passed; refusing to promote.")
    if gate.get("decision", {}).get("create_submission_artifact") is not False:
        raise ValueError("Geometry gate must remain no-artifact before this explicit promotion script.")
    if float(weight) != DEFAULT_WEIGHT:
        decision = weight_scan.get("decision", {})
        if decision.get("selected_weight") != float(weight):
            raise ValueError("Weight geometry scan did not select the requested weight.")
        if decision.get("create_submission_artifact") is not False:
            raise ValueError("Weight geometry scan must remain no-artifact before this explicit promotion script.")
    return selected


def build_candidate(
    *,
    base_id: int,
    target: str,
    source_group: str,
    model_c: float,
    weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float, dict[str, float], dict[str, Any], pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train["date"] = pd.to_datetime(train["date"]).dt.date
    test["date"] = pd.to_datetime(test["date"]).dt.date
    base_oof = align_base_oof(find_base("oof", base_id), train)
    base_test = align_base_test(find_base("test", base_id), test)

    selected = selected_config_from_reports(target, weight)
    if selected["source_group"] != source_group:
        raise ValueError("Selected source group does not match requested source group.")
    if float(selected["weight"]) != float(weight):
        raise ValueError("Selected weight does not match requested weight.")

    feature_frame, feature_metadata = load_source_train_test(source_group, train, test)
    feature_cols = [col for col in feature_frame.columns if col.startswith(f"{source_group}__")]
    splits = load_split_manifest()
    spec = ModelSpec(name=DEFAULT_MODEL_NAME, kind="lr", c_value=model_c)
    y = train[target].astype(int).to_numpy()
    oof_source, covered = predict_target_for_split(
        feature_frame.iloc[: len(train)].reset_index(drop=True),
        feature_cols,
        y,
        splits["subject_hole_5"],
        spec,
    )
    test_source = fit_predict_test_source(
        features=feature_frame,
        feature_cols=feature_cols,
        y=y,
        model_spec={"name": DEFAULT_MODEL_NAME, "kind": "lr", "c_value": model_c},
        n_train=len(train),
    )

    target_idx = TARGETS.index(target)
    oof = base_oof.copy()
    test_pred = base_test.copy()
    oof[target] = np.clip(
        (1.0 - weight) * base_oof[target].astype(float).to_numpy() + weight * oof_source,
        EPS,
        1.0 - EPS,
    )
    test_pred[target] = np.clip(
        (1.0 - weight) * base_test[target].astype(float).to_numpy() + weight * test_source,
        EPS,
        1.0 - EPS,
    )
    target_scores = target_loglosses(train, oof)
    base_target_scores = target_loglosses(train, base_oof)
    local_cv = float(np.mean(list(target_scores.values())))
    slices = slice_summary(train, oof)
    anchors = {
        candidate_id: align_base_test(find_base("test", candidate_id), test)
        for candidate_id in [178, 287, 329, 335, 337, 338, 340]
    }
    geom = geometry(test_pred, anchors)
    test_delta_vs_base = np.abs(test_pred[TARGETS].astype(float).to_numpy() - base_test[TARGETS].astype(float).to_numpy())
    selector_info = {
        "base_id": base_id,
        "target": target,
        "target_index": target_idx,
        "source_group": source_group,
        "model_spec": {"name": DEFAULT_MODEL_NAME, "kind": "lr", "c_value": model_c},
        "weight": weight,
        "canonical_oof_split": "subject_hole_5",
        "covered_oof_rows": int(covered),
        "feature_metadata": feature_metadata,
        "diagnostic_reports": [str(PROBE_REPORT.relative_to(ROOT)), str(GEOMETRY_GATE_REPORT.relative_to(ROOT))],
        "weight_geometry_scan_report": str(WEIGHT_GEOMETRY_SCAN_REPORT.relative_to(ROOT))
        if WEIGHT_GEOMETRY_SCAN_REPORT.exists()
        else None,
        "selected_probe_config": selected,
        "candidate_geometry": geom,
        "candidate_slice_summary": {
            "worst_subject": slices["worst_subject"],
            "worst_subject_loss": slices["worst_subject_loss"],
            "worst_date_bin": slices["worst_date_bin"],
            "worst_date_loss": slices["worst_date_loss"],
            "worst_target": slices["worst_target"],
            "worst_target_loss": slices["worst_target_loss"],
            "late_minus_early": slices["late_minus_early"],
        },
        "candidate_robust_score": robust_score(local_cv, slices),
        "max_delta_vs_base_by_target": max_delta_by_target(test_pred, base_test),
        "mean_delta_vs_base_by_target": mean_delta_by_target(test_pred, base_test),
        "mean_delta_vs_base_all_targets": float(test_delta_vs_base.mean()),
        "max_delta_vs_base_all_targets": float(test_delta_vs_base.max()),
        "oof_target_source_summary": {
            "source_mean": float(np.nanmean(oof_source)),
            "source_min": float(np.nanmin(oof_source)),
            "source_max": float(np.nanmax(oof_source)),
            "base_target_loss": float(base_target_scores[target]),
            "candidate_target_loss": float(target_scores[target]),
        },
        "test_target_source_summary": {
            "source_mean": float(np.mean(test_source)),
            "source_min": float(np.min(test_source)),
            "source_max": float(np.max(test_source)),
            "base_target_mean": float(base_test[target].mean()),
            "candidate_target_mean": float(test_pred[target].mean()),
        },
        "public_context": {
            "id338_public_score": ID338_PUBLIC_SCORE,
            "interpretation": (
                "This is a tiny S3-only independent-source reserve candidate. The geometry gate is safe, but the "
                "fold-internal gain is small and not Public 0.58 evidence."
            ),
        },
        "selection_contract": (
            f"ID{base_id:03d} remains unchanged for all targets except {target}. {target} is blended with a "
            f"phone/wearable discordance logistic source at weight {weight:g}, using subject_hole_5 OOF for train "
            "and a full-train fit for test. This creates a generated-unsubmitted reserve artifact only."
        ),
    }
    selector_info["oof_target_source_summary"]["candidate_target_delta_vs_base"] = float(
        target_scores[target] - base_target_scores[target]
    )
    selector_info["local_cv_delta_vs_base"] = float(local_cv - np.mean(list(base_target_scores.values())))
    return oof, test_pred, local_cv, target_scores, selector_info, sample_original


def write_repro_notebook(
    *,
    notebook_path: Path,
    submission_path: Path,
    metadata_path: Path,
    experiment_id: int,
    slug: str,
    local_cv: float,
    weight: float,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: ID338 S3 phone/wearable micro overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Slug: `{slug}`\n"
            f"- Local OOF CV: `{local_cv:.6f}`\n\n"
            "This is a generated-unsubmitted reserve artifact. It is not Public 0.58 evidence and was not submitted to DACON."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/promote_id338_s3_phone_wearable_micro_overlay_candidate.py "
            f"--experiment-id {experiment_id} --slug {slug} --weight {weight:g}"
        ),
        nbf.v4.new_markdown_cell(
            "### Domain rationale\n\n"
            "S3 is a sleep-continuity/onset-related target. The source features measure disagreement between "
            "phone rest state and wearable activation/light across the day, which is a plausible independent "
            "behavioral signal for sleep-state regularity. The overlay is deliberately tiny to preserve ID338."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--slug", default="id338_s3_phone_wearable_micro_w0005")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--source-group", default=DEFAULT_SOURCE_GROUP)
    parser.add_argument("--weight", type=float, default=DEFAULT_WEIGHT)
    parser.add_argument("--model-c", type=float, default=DEFAULT_MODEL_C)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict artifact ids must be >= 94.")

    oof, test_pred, local_cv, target_scores, selector_info, sample_original = build_candidate(
        base_id=args.base_id,
        target=args.target,
        source_group=args.source_group,
        model_c=args.model_c,
        weight=args.weight,
    )

    status = "unsubmitted"
    filename = f"submission_id_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.csv"
    notebook_name = f"submission_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.ipynb"
    metadata_name = f"experiment_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.json"
    decision_name = f"decision_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.md"
    base_oof_name = f"base_oof_id_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}.csv"
    base_test_name = f"base_test_id_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}.csv"

    submission_path = SUBMISSIONS_DIR / filename
    notebook_path = NOTEBOOK_DIR / notebook_name
    metadata_path = REPORTS_DIR / metadata_name
    decision_path = REPORTS_DIR / decision_name
    base_oof_path = BASE_PREDICTION_DIR / base_oof_name
    base_test_path = BASE_PREDICTION_DIR / base_test_name

    for path in [submission_path, notebook_path, metadata_path, decision_path, base_oof_path, base_test_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path.relative_to(ROOT)}")

    out = clip_probabilities(map_predictions_to_original_sample(sample_original, test_pred), TARGETS, eps=EPS)
    if not validate_submission(out, sample_original):
        raise ValueError("Generated submission failed sample-format validation.")

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BASE_PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    out.to_csv(submission_path, index=False)
    oof.to_csv(base_oof_path, index=False)
    test_pred.to_csv(base_test_path, index=False)

    created_at = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z")
    metadata = {
        "created_at_kst": created_at,
        "experiment_id": args.experiment_id,
        "slug": args.slug,
        "method": "ID338 S3-only phone/wearable discordance micro overlay",
        "submission_file": submission_path.name,
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "id338_s3_subject_hole5_phone_wearable_micro_overlay",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "selector_info": selector_info,
        "risk_decision": {
            "public_058_plausible": False,
            "controlled_public_probe_hypothesis": True,
            "reserve_candidate_only": True,
            "requires_explicit_user_approval_before_dacon_submit": True,
            "reason": (
                "The selected S3 overlay passed the no-artifact geometry gate and moves ID338 by only a tiny "
                "amount, but the fold-internal gain is too small for automatic promotion to DACON submission."
            ),
        },
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "notebook_file": str(notebook_path.relative_to(ROOT)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    geom = selector_info["candidate_geometry"]
    slices = selector_info["candidate_slice_summary"]
    decision_md = (
        f"# Decision note - submission {args.experiment_id:03d} {args.slug}\n\n"
        f"Created: `{created_at}`\n\n"
        "## Decision\n\n"
        "- DACON submission: **no**\n"
        "- Public 0.58 plausible: **no**\n"
        "- Reserve controlled S3 public probe hypothesis: **yes, still unsubmitted**\n\n"
        "## Evidence\n\n"
        f"- Local OOF CV: `{local_cv:.6f}`\n"
        f"- Local CV delta vs ID338 base: `{selector_info['local_cv_delta_vs_base']:+.8f}`\n"
        f"- S3 delta vs ID338 base: `{selector_info['oof_target_source_summary']['candidate_target_delta_vs_base']:+.8f}`\n"
        f"- Robust score: `{selector_info['candidate_robust_score']:.6f}`\n"
        f"- Mean delta vs ID338: `{selector_info['mean_delta_vs_base_all_targets']:.8f}`\n"
        f"- Max delta vs ID338: `{selector_info['max_delta_vs_base_all_targets']:.8f}`\n"
        f"- Failed-direction projection: `{geom['projection_failed']:+.8f}`\n"
        f"- Worst target: `{slices['worst_target']}` `{slices['worst_target_loss']:.6f}`\n\n"
        "## Rationale\n\n"
        "The phone/wearable discordance source is domain-relevant for S3 because it contrasts phone rest "
        f"state with wearable activation/light. The selected weight is `{args.weight:g}`, so the artifact preserves "
        "ID338 while adding one independent S3 signal. This is a reserve candidate, not a submission "
        "recommendation by itself.\n\n"
        "## Artifact contract\n\n"
        f"- Submission CSV: `{submission_path.relative_to(ROOT)}`\n"
        f"- Notebook: `{notebook_path.relative_to(ROOT)}`\n"
        f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
        f"- Base OOF: `{base_oof_path.relative_to(ROOT)}`\n"
        f"- Base test: `{base_test_path.relative_to(ROOT)}`\n"
    )
    decision_path.write_text(decision_md, encoding="utf-8")
    write_repro_notebook(
        notebook_path=notebook_path,
        submission_path=submission_path,
        metadata_path=metadata_path,
        experiment_id=args.experiment_id,
        slug=args.slug,
        local_cv=local_cv,
        weight=args.weight,
    )

    notes = (
        "Generated ID338 S3-only phone/wearable discordance micro overlay. "
        f"Q1/Q2/Q3/S1/S2/S4 stay at ID338; S3 blends an independent phone/wearable source at weight {args.weight:g}. "
        "Geometry gate passed but signal is tiny; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description="ID338 S3 phone/wearable micro overlay",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
