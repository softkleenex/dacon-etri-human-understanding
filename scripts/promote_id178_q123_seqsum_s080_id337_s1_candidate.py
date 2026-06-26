#!/usr/bin/env python3
"""Promote the post-A1 ID178 Q123 sequence-summary + ID337 S1 candidate.

This creates a generated-unsubmitted artifact only. It never submits to DACON.
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
from scripts.run_public_sequence_summary_embedding_overlay_experiment import (  # noqa: E402
    predict_sequence_summary_embedding_overlay,
)
from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
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
DEFAULT_EXPERIMENT_ID = 339
DEFAULT_BASE_ID = 178
DEFAULT_S1_SOURCE_ID = 337
DEFAULT_SCALE = 0.80
ID178_PUBLIC_SCORE = 0.6014113009
ID178_LOCAL_REFERENCE = 0.5386382667903545
ID178_PUBLIC_GAP = ID178_PUBLIC_SCORE - ID178_LOCAL_REFERENCE
DIAGNOSTIC_REPORTS = [
    "reports/id178_sequence_summary_embedding_post_a1_20260618.json",
    "reports/sequence_summary_geometry_gate_post_a1_20260618.json",
    "reports/sequence_summary_cap_sensitivity_post_a1_20260618.json",
]
BASE_SEQUENCE_CONFIGS: dict[str, dict[str, Any]] = {
    "Q1": {"kind": "blend", "model": "lr_c0_3", "weight": 0.10},
    "Q2": {"kind": "blend", "model": "lr_c0_3", "weight": 0.15},
    "Q3": {"kind": "blend", "model": "lr_c0_1", "weight": 0.10},
    "S1": {"kind": "base"},
    "S2": {"kind": "base"},
    "S3": {"kind": "base"},
    "S4": {"kind": "base"},
}


def scaled_sequence_configs(scale: float) -> dict[str, dict[str, Any]]:
    if not 0.0 < scale <= 1.0:
        raise ValueError("scale must be in (0, 1].")
    configs: dict[str, dict[str, Any]] = {}
    for target, config in BASE_SEQUENCE_CONFIGS.items():
        configs[target] = dict(config)
        if config["kind"] == "blend":
            configs[target]["weight"] = float(config["weight"]) * scale
    return configs


def target_loglosses(train: Any, pred: Any) -> dict[str, float]:
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


def max_delta_by_target(candidate: Any, base: Any) -> dict[str, float]:
    return {
        target: float(
            np.max(np.abs(candidate[target].astype(float).to_numpy() - base[target].astype(float).to_numpy()))
        )
        for target in TARGETS
    }


def build_candidate(
    *,
    base_id: int,
    s1_source_id: int,
    scale: float,
) -> tuple[Any, Any, float, dict[str, float], dict[str, Any], Any]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    configs = scaled_sequence_configs(scale)
    oof, test_pred, _scores, selector_info, _sample_original = predict_sequence_summary_embedding_overlay(
        base_id,
        selected_configs=configs,
        diagnostic_reports=DIAGNOSTIC_REPORTS,
    )
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    s1_oof, s1_test, s1_source = load_artifact_pair(train, test, s1_source_id)
    anchor_tests = {
        candidate_id: load_artifact_pair(train, test, candidate_id)[1] for candidate_id in [178, 287, 329, 335]
    }

    oof = oof.copy()
    test_pred = test_pred.copy()
    oof["S1"] = s1_oof["S1"].to_numpy()
    test_pred["S1"] = s1_test["S1"].to_numpy()

    target_scores = target_loglosses(train, oof)
    local_cv = float(np.mean(list(target_scores.values())))
    slices = slice_summary(train, oof)
    geom = geometry(test_pred, anchor_tests)
    selector_info.update(
        {
            "base_id": base_id,
            "base_source": base_source,
            "s1_source_id": s1_source_id,
            "s1_source": s1_source,
            "scale": scale,
            "modified_targets": ["Q1", "Q2", "Q3", "S1"],
            "unchanged_targets": ["S2", "S3", "S4"],
            "scaled_sequence_configs": configs,
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
            "target_details": {
                **selector_info["target_details"],
                "S1": {
                    "config": {"kind": "artifact_replace", "source_id": s1_source_id},
                    "base_loss": float(
                        log_loss(
                            train["S1"].astype(int),
                            np.clip(base_oof["S1"].astype(float), EPS, 1.0 - EPS),
                            labels=[0, 1],
                        )
                    ),
                    "selected_loss": target_scores["S1"],
                    "source_loss": float(
                        log_loss(
                            train["S1"].astype(int),
                            np.clip(s1_oof["S1"].astype(float), EPS, 1.0 - EPS),
                            labels=[0, 1],
                        )
                    ),
                    "test_prediction_max_abs_delta_vs_base": max_delta_by_target(test_pred, base_test)["S1"],
                },
            },
            "selection_contract": (
                f"ID{base_id:03d} is kept as the public anchor. Q1/Q2/Q3 use the post-A1 "
                f"sequence-summary overlay with all selected weights scaled by {scale:.2f}; S1 is replaced by "
                f"the already-generated ID{s1_source_id:03d} S1 safety source. S2/S3/S4 remain ID{base_id:03d}. "
                "No DACON submission was made."
            ),
            "public_gap_context": {
                "id178_public_score": ID178_PUBLIC_SCORE,
                "id178_local_reference": ID178_LOCAL_REFERENCE,
                "id178_public_minus_local_gap": ID178_PUBLIC_GAP,
                "naive_gap_projected_public": float(local_cv + ID178_PUBLIC_GAP),
                "interpretation": (
                    "This projection is only context. ID287 and ID329 showed that local-to-public transfer is "
                    "fragile, so this artifact is a controlled candidate-construction lead, not a Public 0.58 claim."
                ),
            },
        }
    )
    return oof, test_pred, local_cv, target_scores, selector_info, sample_original


def write_repro_notebook(
    *,
    notebook_path: Path,
    submission_path: Path,
    metadata_path: Path,
    experiment_id: int,
    slug: str,
    local_cv: float,
    base_id: int,
    s1_source_id: int,
    scale: float,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: ID{base_id:03d} Q123 seq-summary s{scale:.2f} + "
            f"ID{s1_source_id:03d} S1\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Slug: `{slug}`\n"
            f"- Local OOF CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for an immutable generated_unsubmitted artifact. "
            "No DACON submission was made."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/promote_id178_q123_seqsum_s080_id337_s1_candidate.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id} "
            f"--s1-source-id {s1_source_id} --scale {scale:g}"
        ),
        nbf.v4.new_markdown_cell(
            "### Decision note\n\n"
            "This artifact is below the internal max-movement cap used in the report-only scan, but it still "
            "requires explicit user approval before any DACON submission."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--slug", default="id178_q123_seqsum_s080_id337_s1_safety")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--s1-source-id", type=int, default=DEFAULT_S1_SOURCE_ID)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict artifact ids must be >= 94.")

    oof, test_pred, local_cv, target_scores, selector_info, sample_original = build_candidate(
        base_id=args.base_id,
        s1_source_id=args.s1_source_id,
        scale=args.scale,
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
        "method": (
            f"ID{args.base_id:03d} Q1/Q2/Q3 sequence-summary scale {args.scale:.2f} "
            f"+ ID{args.s1_source_id:03d} S1 safety"
        ),
        "submission_file": submission_path.name,
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_interleaved_within_subject_blocked_cv_sequence_summary_scaled_s1_safety",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "selector_info": selector_info,
        "risk_decision": {
            "public_058_plausible": False,
            "public_060_controlled_probe_hypothesis": True,
            "requires_explicit_user_approval_before_dacon_submit": True,
            "reason": (
                "The candidate preserves the post-A1 local sequence-summary gain while keeping max movement below "
                "0.10 versus ID178. It is still selected from public-structure diagnostics and is not robust "
                "Public 0.58 evidence."
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
        "- Controlled public-anchor probe hypothesis: **yes, still unsubmitted**\n\n"
        "## Evidence\n\n"
        f"- Local OOF CV: `{local_cv:.6f}`\n"
        f"- Robust score: `{selector_info['candidate_robust_score']:.6f}`\n"
        f"- Max Δ vs ID178: `{geom['max_delta_vs_178']:.6f}`\n"
        f"- Failed-direction projection: `{geom['projection_failed']:+.6f}`\n"
        f"- Worst target: `{slices['worst_target']}` `{slices['worst_target_loss']:.6f}`\n\n"
        "## Rationale\n\n"
        f"Q1/Q2/Q3 use the ID178 sequence-summary overlay with weights scaled by `{args.scale:.2f}`. "
        f"S1 uses the existing ID{args.s1_source_id:03d} safety source. S2/S3/S4 remain ID{args.base_id:03d}. "
        "This is the first full-contract promotion of the post-A1 sequence-summary cap-sensitivity lead, but it "
        "must not be treated as a leaderboard improvement until the user explicitly approves and DACON scores it.\n\n"
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
        base_id=args.base_id,
        s1_source_id=args.s1_source_id,
        scale=args.scale,
    )

    notes = (
        f"Generated ID{args.base_id:03d} Q1/Q2/Q3 sequence-summary scale{args.scale:.2f} plus "
        f"ID{args.s1_source_id:03d} S1 safety artifact. S2/S3/S4 stay at ID{args.base_id:03d}. "
        "Max movement is below 0.10 versus ID178 in the post-A1 cap-sensitivity gate. "
        "Not Public 0.58-plausible; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"ID{args.base_id:03d} Q123 seq-summary scale{args.scale:.2f} + ID{args.s1_source_id:03d} S1",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
