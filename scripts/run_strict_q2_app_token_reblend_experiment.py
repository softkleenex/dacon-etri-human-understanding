#!/usr/bin/env python3
"""Diagnose/promote a narrow Q2 app-token reblend on top of a strict base artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import scripts.run_strict_q2_app_token_compact_experiment as app_base  # noqa: E402
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    TARGETS,
    map_predictions_to_original_sample,
    validate_submission,
)
from src.utils.validation import clip_probabilities  # noqa: E402

TARGET = "Q2"
DEFAULT_BASE_ID = 118
REBLEND_WEIGHTS = [0.02, 0.03, 0.05, 0.07, 0.10]


def apply_reblend_patch() -> None:
    """Retain the app-token model but restrict the second-stage blend neighborhood."""

    app_base.BLEND_WEIGHTS = REBLEND_WEIGHTS


def base_candidate_name(base_id: int) -> str:
    return f"id{base_id:03d}_base"


def write_diagnostic_report(base_id: int, local_cv: float, target_scores: dict[str, float], selector_info: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"strict_q2_app_token_reblend_diagnostic_id{base_id:03d}.json"
    if path.exists():
        suffix = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"strict_q2_app_token_reblend_diagnostic_id{base_id:03d}_{suffix}.json"
    report = {
        "base_id": base_id,
        "target": TARGET,
        "local_cv": local_cv,
        "target_scores": target_scores,
        "selector_info": selector_info,
        "reblend_weights": REBLEND_WEIGHTS,
        "confirm_note": (
            "Narrow second-stage Q2 app-token reblend after ID115/ID118: reuse the same top-app LGBM signal "
            "but cap additional raw-app blend weights at 0.10 so the already-successful ID118 Q2 prediction "
            "remains the anchor. Promote only when nested Q2 improves and full-test selection is not the base."
        ),
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_repro_notebook(
    notebook_path: Path,
    submission_path: Path,
    metadata_path: Path,
    experiment_id: int,
    slug: str,
    local_cv: float,
    base_id: int,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: Q2 app-token reblend overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Local Subject-Hole CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for an immutable artifact. "
            f"The generating script keeps ID{base_id:03d} predictions for every non-Q2 target and replaces Q2 only."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_strict_q2_app_token_reblend_experiment.py "
            f"--promote --experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "Do not submit it before reviewing the metadata, audit reports, and validation gap."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def append_submission_log(
    experiment_id: int,
    submission_path: Path,
    notebook_path: Path,
    local_cv: float,
    notes: str,
    base_id: int,
) -> None:
    log_path = SUBMISSIONS_DIR / "submission_log.csv"
    log_df = pd.read_csv(log_path, dtype=str).fillna("")
    if (log_df["id"].astype(str) == str(experiment_id)).any():
        raise ValueError(f"submission_log.csv already has id={experiment_id}")

    row = dict.fromkeys(log_df.columns, "")
    row.update(
        {
            "id": str(experiment_id),
            "date": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"),
            "submission_file": submission_path.name,
            "model_description": f"Strict Q2 app-token reblend overlay on ID{base_id:03d}",
            "local_cv_score": f"{local_cv:.6f}",
            "public_lb_score": "",
            "is_submitted": "False",
            "notes": notes,
            "is_submitted_bool": "False",
            "local_cv_score_num": f"{local_cv:.6f}",
            "public_lb_score_num": "",
            "cv_lb_gap": "",
            "artifact_exists": "True",
            "status_clean": "generated_unsubmitted",
            "risk_reason": "Generated locally but not submitted/scored.",
            "safe_for_default_ensemble": "False",
            "artifact_role": "submission",
            "duplicate_file_count": "1",
            "notebook_file": str(notebook_path.relative_to(ROOT)),
        }
    )
    pd.concat([log_df, pd.DataFrame([row])], ignore_index=True).to_csv(log_path, index=False)


def promote_artifacts(
    experiment_id: int,
    slug: str,
    base_id: int,
    oof: pd.DataFrame,
    test_pred: pd.DataFrame,
    target_scores: dict[str, float],
    selector_info: dict,
    sample_original: pd.DataFrame,
    diagnostic_path: Path,
) -> dict:
    if experiment_id < 94:
        raise ValueError("Strict experiments must use id >= 94.")

    base_name = base_candidate_name(base_id)
    full_choice = selector_info["full_train_test_choices"][TARGET]["candidate"]
    if full_choice == base_name:
        raise ValueError("Refusing to promote: full-training test candidate is the unchanged base.")

    local_cv = float(np.mean(list(target_scores.values())))
    status = "unsubmitted"
    filename = f"submission_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.csv"
    notebook_name = f"submission_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.ipynb"
    metadata_name = f"experiment_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.json"
    base_oof_name = f"base_oof_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}.csv"
    base_test_name = f"base_test_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}.csv"

    submission_path = SUBMISSIONS_DIR / filename
    notebook_path = NOTEBOOK_DIR / notebook_name
    metadata_path = REPORTS_DIR / metadata_name
    base_oof_path = BASE_PREDICTION_DIR / base_oof_name
    base_test_path = BASE_PREDICTION_DIR / base_test_name

    for path in [submission_path, notebook_path, metadata_path, base_oof_path, base_test_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path.relative_to(ROOT)}")

    out = clip_probabilities(map_predictions_to_original_sample(sample_original, test_pred), TARGETS)
    if not validate_submission(out, sample_original):
        raise ValueError("Generated submission failed sample-format validation.")

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BASE_PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    out.to_csv(submission_path, index=False)
    oof.to_csv(base_oof_path, index=False)
    test_pred.to_csv(base_test_path, index=False)

    metadata = {
        "experiment_id": experiment_id,
        "slug": slug,
        "method": "Q2 narrow second-stage app-token reblend overlay",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": app_base.sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "diagnostic_report": str(diagnostic_path.relative_to(ROOT)),
        "validation": "Subject-Hole CV with Q2-only narrow reblend over the existing top-app-token LGBM signal.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    write_repro_notebook(
        notebook_path=notebook_path,
        submission_path=submission_path,
        metadata_path=metadata_path,
        experiment_id=experiment_id,
        slug=slug,
        local_cv=local_cv,
        base_id=base_id,
    )

    notes = (
        f"Strict Q2-only second-stage app-token reblend on ID{base_id:03d}. Reuses the top-app LGBM signal "
        "with capped additional blend weights; promoted only after nested Q2 improved and full-test selection "
        "changed Q2 predictions. No DACON submission was made."
    )
    append_submission_log(
        experiment_id=experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        base_id=base_id,
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--slug", default="q2_app_token_reblend_overlay")
    args = parser.parse_args()

    apply_reblend_patch()
    oof, test_pred, target_scores, selector_info, sample_original = app_base.predict_q2_app_token_compact_overlay(
        base_id=args.base_id
    )
    local_cv = float(np.mean(list(target_scores.values())))
    selector_info["reblend_weights"] = REBLEND_WEIGHTS
    selector_info["reblend_base_id"] = args.base_id
    selector_info["reblend_contract"] = (
        "All non-Q2 targets remain fixed from the base artifact. Q2 candidates are the base, the same raw "
        "top-app LGBM signal, and small additional blends on top of the base."
    )
    diagnostic_path = write_diagnostic_report(args.base_id, local_cv, target_scores, selector_info)

    payload = {
        "mode": "diagnostic",
        "diagnostic_report": str(diagnostic_path.relative_to(ROOT)),
        "base_id": args.base_id,
        "target": TARGET,
        "local_cv": local_cv,
        "target_scores": target_scores,
        "candidate_scores": selector_info["q2_candidate_scores"],
        "full_train_test_choices": selector_info["full_train_test_choices"],
        "nested_choices": selector_info["nested_choices"],
    }
    if args.promote:
        if args.experiment_id is None:
            raise ValueError("--experiment-id is required with --promote")
        metadata = promote_artifacts(
            experiment_id=args.experiment_id,
            slug=args.slug,
            base_id=args.base_id,
            oof=oof,
            test_pred=test_pred,
            target_scores=target_scores,
            selector_info=selector_info,
            sample_original=sample_original,
            diagnostic_path=diagnostic_path,
        )
        payload = {"mode": "promote", **metadata}

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
