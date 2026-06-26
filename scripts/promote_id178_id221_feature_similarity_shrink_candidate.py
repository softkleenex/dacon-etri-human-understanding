#!/usr/bin/env python3
"""Promote an ID178/ID221 feature-similarity shrink candidate.

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_id179_vs_id335_sequence_probe_review import (  # noqa: E402
    EPS,
    TARGETS,
    align,
    geometry,
    load_artifact,
    local_scores,
    robust_score,
    slice_summary,
)
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    append_submission_log,
    build_feature_frames,
    map_predictions_to_original_sample,
    sha256_file,
    validate_submission,
)
from src.utils.validation import clip_probabilities  # noqa: E402

KST = timezone(timedelta(hours=9))
DEFAULT_EXPERIMENT_ID = 342
DEFAULT_BASE_ID = 178
DEFAULT_SOURCE_ID = 221
DEFAULT_ALPHA = 0.50
ID178_PUBLIC_SCORE = 0.6014113009
ID178_LOCAL_REFERENCE = 0.5386382667903545
ID178_PUBLIC_GAP = ID178_PUBLIC_SCORE - ID178_LOCAL_REFERENCE
DIAGNOSTIC_REPORTS = [
    "reports/post_id341_transfer_gate_v2_joint_risk_20260623.json",
    "reports/joint_risk_candidate_rerank_post_id341_20260623.json",
    "reports/public_joint_distribution_risk_post_id341_20260623.json",
]


def max_delta_by_target(candidate: pd.DataFrame, base: pd.DataFrame) -> dict[str, float]:
    return {
        target: float(
            np.max(np.abs(candidate[target].astype(float).to_numpy() - base[target].astype(float).to_numpy()))
        )
        for target in TARGETS
    }


def blend_predictions(base: pd.DataFrame, source: pd.DataFrame, alpha: float) -> pd.DataFrame:
    out = base.copy()
    base_values = base[TARGETS].astype(float).to_numpy()
    source_values = source[TARGETS].astype(float).to_numpy()
    out[TARGETS] = np.clip((1.0 - alpha) * base_values + alpha * source_values, EPS, 1.0 - EPS)
    return out


def build_candidate(*, base_id: int, source_id: int, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1].")

    train, test, _metrics_sorted, sample_original = build_feature_frames()
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    base_oof = align(load_artifact("oof", base_id), train)
    base_test = align(load_artifact("test", base_id), test)
    source_oof = align(load_artifact("oof", source_id), train)
    source_test = align(load_artifact("test", source_id), test)
    anchors = {candidate_id: align(load_artifact("test", candidate_id), test) for candidate_id in [178, 287, 329, 335]}

    oof = blend_predictions(base_oof, source_oof, alpha)
    test_pred = blend_predictions(base_test, source_test, alpha)
    local_cv, target_scores = local_scores(train, oof)
    slices = slice_summary(train, oof)
    geom = geometry(test_pred, anchors)
    selector_info = {
        "base_id": base_id,
        "source_id": source_id,
        "alpha": alpha,
        "modified_targets": TARGETS,
        "unchanged_targets": [],
        "diagnostic_reports": DIAGNOSTIC_REPORTS,
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
        "selection_contract": (
            f"ID{base_id:03d} is kept as the public anchor and blended with ID{source_id:03d} "
            f"feature-similarity predictions at alpha={alpha:.2f} for all targets. ID{source_id:03d} "
            "is a strict-feature/feature-similarity family probe that was not consumed because DACON "
            "returned day_max_count. The shrink keeps the independent source signal while reducing "
            "public-geometry movement. No DACON submission was made."
        ),
        "public_gap_context": {
            "id178_public_score": ID178_PUBLIC_SCORE,
            "id178_local_reference": ID178_LOCAL_REFERENCE,
            "id178_public_minus_local_gap": ID178_PUBLIC_GAP,
            "naive_gap_projected_public": float(local_cv + ID178_PUBLIC_GAP),
            "interpretation": (
                "This projection is context only. Recent public probes stayed near 0.60, so this artifact is "
                "a controlled independent-source probe, not Public 0.58 evidence."
            ),
        },
    }
    return (
        oof,
        test_pred,
        {
            "train": train,
            "test": test,
            "sample_original": sample_original,
            "local_cv": local_cv,
            "target_scores": target_scores,
            "selector_info": selector_info,
        },
    )


def write_repro_notebook(
    *,
    notebook_path: Path,
    submission_path: Path,
    metadata_path: Path,
    experiment_id: int,
    slug: str,
    base_id: int,
    source_id: int,
    alpha: float,
    local_cv: float,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: ID{base_id:03d}/ID{source_id:03d} feature-similarity shrink\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Slug: `{slug}`\n"
            f"- Alpha: `{alpha:.2f}`\n"
            f"- Local OOF CV: `{local_cv:.6f}`\n\n"
            "This is a generated-unsubmitted artifact. It is not Public 0.58 evidence and was not submitted to DACON."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/promote_id178_id221_feature_similarity_shrink_candidate.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id} "
            f"--source-id {source_id} --alpha {alpha:g}"
        ),
        nbf.v4.new_markdown_cell(
            "### Domain rationale\n\n"
            "The source family uses feature similarity over processed behavioral and sensor summaries. "
            "Shrinking toward the ID178 temporal anchor keeps subject/day similarity information while reducing "
            "distribution movement in the hidden-test prediction matrix."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--slug", default="id178_id221_featuresim_shrink050")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--source-id", type=int, default=DEFAULT_SOURCE_ID)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict artifact ids must be >= 94.")

    oof, test_pred, context = build_candidate(base_id=args.base_id, source_id=args.source_id, alpha=args.alpha)
    local_cv = float(context["local_cv"])
    sample_original = context["sample_original"]
    selector_info = context["selector_info"]

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
        "method": f"ID{args.base_id:03d}/ID{args.source_id:03d} feature-similarity shrink alpha {args.alpha:.2f}",
        "submission_file": submission_path.name,
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_feature_similarity_shrink_probe",
        "strict_subject_hole_comparable": False,
        "target_scores": context["target_scores"],
        "targets": TARGETS,
        "selector_info": selector_info,
        "risk_decision": {
            "public_058_plausible": False,
            "controlled_public_probe_hypothesis": True,
            "requires_explicit_user_approval_before_dacon_submit": True,
            "reason": (
                "ID221 is an independent strict-feature/feature-similarity source but has high geometry movement. "
                "Alpha 0.50 keeps local improvement while capping max movement below 0.10 versus ID178."
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
        f"# Decision note — submission {args.experiment_id:03d} {args.slug}\n\n"
        f"Created: `{created_at}`\n\n"
        "## Decision\n\n"
        "- DACON submission: **no**\n"
        "- Public 0.58 plausible: **no**\n"
        "- Controlled independent-source public probe hypothesis: **yes, still unsubmitted**\n\n"
        "## Evidence\n\n"
        f"- Local OOF CV: `{local_cv:.6f}`\n"
        f"- Robust score: `{selector_info['candidate_robust_score']:.6f}`\n"
        f"- Mean delta vs ID178: `{geom['mean_delta_vs_178']:.6f}`\n"
        f"- Max delta vs ID178: `{geom['max_delta_vs_178']:.6f}`\n"
        f"- Failed-direction projection: `{geom['projection_failed']:+.6f}`\n"
        f"- Worst target: `{slices['worst_target']}` `{slices['worst_target_loss']:.6f}`\n\n"
        "## Rationale\n\n"
        f"ID{args.source_id:03d} brings feature-similarity signal from the strict-feature family, but the original "
        "unshrunk test geometry was too large for a direct probe. This candidate blends it with ID178 at "
        f"alpha `{args.alpha:.2f}` across all targets. The intent is to test whether non-sequence behavioral "
        "similarity transfers publicly while keeping anchor movement restrained.\n\n"
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
        base_id=args.base_id,
        source_id=args.source_id,
        alpha=args.alpha,
        local_cv=local_cv,
    )

    notes = (
        f"Generated ID{args.base_id:03d}/ID{args.source_id:03d} feature-similarity shrink alpha {args.alpha:.2f}. "
        "All targets are convex blends against ID178 to keep independent strict-feature signal with lower movement. "
        "Not Public 0.58-plausible; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"ID{args.base_id:03d}/ID{args.source_id:03d} feature-similarity shrink a{args.alpha:.2f}",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
