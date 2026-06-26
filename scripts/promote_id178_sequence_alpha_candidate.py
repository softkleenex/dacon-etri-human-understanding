#!/usr/bin/env python3
"""Promote a target-wise ID178 sequence alpha refinement as no-submit artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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

from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    TARGETS,
    append_submission_log,
    build_feature_frames,
    map_predictions_to_original_sample,
    validate_submission,
)
from scripts.run_strict_shutdown_feature_set_selector_experiment import align_candidate  # noqa: E402
from src.utils.validation import clip_probabilities  # noqa: E402

KST = timezone(timedelta(hours=9))
EPS = 1e-6
DEFAULT_ALPHA_BY_TARGET = {
    "Q1": 1.0,
    "Q2": 1.0,
    "Q3": 1.0,
    "S1": 0.0,
    "S2": 1.0,
    "S3": 0.0,
    "S4": 0.5,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_base_file(kind: str, candidate_id: int) -> Path:
    paths = sorted(BASE_PREDICTION_DIR.glob(f"base_{kind}_id_{candidate_id:03d}_*.csv"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected exactly one base_{kind} file for id {candidate_id}, found {paths}")
    return paths[0]


def candidate_slug(path: Path) -> str:
    match = re.search(r"id_\d+_(.+)\.csv$", path.name)
    return match.group(1) if match else path.stem


def load_aligned_pair(
    train: pd.DataFrame, test: pd.DataFrame, candidate_id: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    oof_path = find_base_file("oof", candidate_id)
    test_path = find_base_file("test", candidate_id)
    return (
        align_candidate(pd.read_csv(oof_path), train),
        align_candidate(pd.read_csv(test_path), test),
        {
            "oof": str(oof_path.relative_to(ROOT)),
            "test": str(test_path.relative_to(ROOT)),
            "slug": candidate_slug(oof_path),
        },
    )


def blend_predictions(base: pd.DataFrame, overlay: pd.DataFrame, alpha_by_target: dict[str, float]) -> pd.DataFrame:
    out = base[["subject_id", "date", *TARGETS]].copy()
    for target in TARGETS:
        alpha = float(alpha_by_target[target])
        base_values = np.clip(base[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        overlay_values = np.clip(overlay[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        out[target] = np.clip(base_values + alpha * (overlay_values - base_values), EPS, 1.0 - EPS)
    return clip_probabilities(out, TARGETS, eps=EPS)


def mean_logloss(train: pd.DataFrame, pred: pd.DataFrame) -> tuple[float, dict[str, float]]:
    per_target = {
        target: float(
            log_loss(
                train[target].astype(int),
                np.clip(pred[target].astype(float), EPS, 1.0 - EPS),
                labels=[0, 1],
            )
        )
        for target in TARGETS
    }
    return float(np.mean(list(per_target.values()))), per_target


def load_refine_row(path: Path, alpha_by_target: dict[str, float]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    top_rows = report.get("selected_rows", {}).get("top_12_geometry_guarded", [])
    for row in top_rows:
        if all(
            abs(float(row["alpha_by_target"][target]) - float(alpha_by_target[target])) < 1e-12 for target in TARGETS
        ):
            return row
    return report.get("selected_rows", {}).get("best_local_geometry_guarded")


def write_repro_notebook(
    *,
    notebook_path: Path,
    submission_path: Path,
    metadata_path: Path,
    experiment_id: int,
    slug: str,
    local_cv: float,
    alpha_by_target: dict[str, float],
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: ID178 sequence alpha refinement\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Slug: `{slug}`\n"
            f"- Local OOF CV: `{local_cv:.6f}`\n"
            f"- Alpha by target: `{alpha_by_target}`\n\n"
            "This notebook is a reproduction pointer for an immutable generated_unsubmitted artifact. "
            "No DACON submission was made."
        ),
        nbf.v4.new_code_cell(
            f"!python scripts/promote_id178_sequence_alpha_candidate.py --experiment-id {experiment_id} --slug {slug}"
        ),
        nbf.v4.new_markdown_cell(
            "### Decision note\n\n"
            "This artifact is a same-family ID178/ID334 refinement. It is not Public 0.58-plausible and must not be submitted without explicit approval."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, default=335)
    parser.add_argument("--slug", default="id178_alpha_s4half_stressguard")
    parser.add_argument(
        "--refine-report",
        type=Path,
        default=REPORTS_DIR / "id178_sequence_alpha_refine_post_id334_20260617.json",
    )
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict artifact ids must be >= 94.")

    train, test, _metrics_sorted, sample_original = build_feature_frames()
    id173_oof, id173_test, id173_files = load_aligned_pair(train, test, 173)
    id178_oof, id178_test, id178_files = load_aligned_pair(train, test, 178)

    alpha_by_target = DEFAULT_ALPHA_BY_TARGET.copy()
    oof = blend_predictions(id173_oof, id178_oof, alpha_by_target)
    test_pred = blend_predictions(id173_test, id178_test, alpha_by_target)
    local_cv, target_scores = mean_logloss(train, oof)

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

    refine_row = load_refine_row(args.refine_report, alpha_by_target)
    metadata = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "experiment_id": args.experiment_id,
        "slug": args.slug,
        "method": "Target-wise attenuation between ID173 base and ID178 sequence overlay; S4 uses alpha 0.5.",
        "submission_file": submission_path.name,
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "alpha_by_target": alpha_by_target,
        "source_files": {
            "id173": id173_files,
            "id178": id178_files,
        },
        "refine_diagnostic_report": str(args.refine_report.relative_to(ROOT)),
        "refine_diagnostic_row": refine_row,
        "feature_contract": (
            "No new model training. Deterministically blends existing ID173 and ID178 base OOF/test artifacts "
            "target-wise with alpha Q1/Q2/Q3/S2=1.0, S1/S3=0.0, S4=0.5. No DACON submission was made."
        ),
        "risk_decision": {
            "public_058_plausible": False,
            "public_060_probe_hypothesis": True,
            "requires_explicit_user_approval_before_dacon_submit": True,
            "reason": (
                "Targeted alpha refinement improves ID334 local OOF by a small amount while keeping stress roughly flat. "
                "It remains same-family with ID178/ID334 and is only a weak controlled-probe hypothesis."
            ),
        },
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "notebook_file": str(notebook_path.relative_to(ROOT)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    decision_md = (
        f"# Decision note — submission {args.experiment_id:03d} {args.slug}\n\n"
        f"Created: `{metadata['created_at_kst']}`\n\n"
        "## Decision\n\n"
        "- DACON submission: **no**\n"
        "- Public 0.58 plausible: **no**\n"
        "- Public <0.60 controlled probe hypothesis: **yes, weak**\n\n"
        "## Rationale\n\n"
        "This is a tiny refinement of ID334: keep Q1/Q2/Q3/S2 at the ID178 overlay, keep S1/S3 at ID173, "
        "and restore only half of the S4 overlay. The targeted alpha diagnostic found a small local improvement "
        "with stress remaining near ID334, but the family is still anchored to ID178 and must not be submitted without explicit approval.\n\n"
        "## Artifact contract\n\n"
        f"- Submission CSV: `{submission_path.relative_to(ROOT)}`\n"
        f"- Notebook: `{notebook_path.relative_to(ROOT)}`\n"
        f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
        f"- Base OOF: `{base_oof_path.relative_to(ROOT)}`\n"
        f"- Base test: `{base_test_path.relative_to(ROOT)}`\n"
        f"- Local OOF CV: `{local_cv:.6f}`\n"
    )
    decision_path.write_text(decision_md, encoding="utf-8")

    write_repro_notebook(
        notebook_path=notebook_path,
        submission_path=submission_path,
        metadata_path=metadata_path,
        experiment_id=args.experiment_id,
        slug=args.slug,
        local_cv=local_cv,
        alpha_by_target=alpha_by_target,
    )

    notes = (
        "Generated ID178/ID334 target-wise alpha refinement artifact. Q1/Q2/Q3/S2 use ID178, "
        "S1/S3 use ID173, and S4 uses 0.5 * ID178 overlay over ID173. Not Public 0.58-plausible; "
        "no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description="ID178 sequence alpha S4-half stressguard",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
