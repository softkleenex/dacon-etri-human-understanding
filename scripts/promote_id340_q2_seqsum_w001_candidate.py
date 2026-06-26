#!/usr/bin/env python3
"""Promote ID340 Q2-only sequence-summary safety candidate.

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

from scripts.diagnose_id179_vs_id335_sequence_probe_review import geometry, robust_score, slice_summary  # noqa: E402
from scripts.diagnose_id340_exact_sequence_full_geometry_gate import (  # noqa: E402
    MODEL_SPEC,
    train_full_sequence_source,
)
from scripts.diagnose_id340_sequence_subset_geometry_scan import (  # noqa: E402
    SPLIT_MANIFEST_PATH,
    load_artifact_pair,
    load_split_manifest,
    split_source_predictions,
)
from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    RAW_DIR,
    align_table,
    build_daily_sequence_table,
    table_to_tensor,
)
from scripts.diagnose_public_sequence_summary_embedding import sequence_summary_features  # noqa: E402
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
DEFAULT_EXPERIMENT_ID = 340
DEFAULT_BASE_ID = 178
DEFAULT_TARGET = "Q2"
DEFAULT_WEIGHT = 0.01
DIAGNOSTIC_REPORTS = [
    "reports/id340_sequence_summary_split_probe_20260618.json",
    "reports/id340_exact_sequence_full_geometry_gate_20260618.json",
    "reports/id340_sequence_subset_geometry_scan_20260618.json",
    "reports/id340_sequence_summary_candidate_decision_20260618.json",
    "reports/post_id340_micro_sequence_target_scan_20260618.json",
]


def target_loglosses(train: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    return {
        target: float(
            log_loss(
                train[target].astype(int),
                np.clip(pred[target].astype(float), 1e-6, 1.0 - 1e-6),
                labels=[0, 1],
            )
        )
        for target in TARGETS
    }


def build_candidate(*, base_id: int, target: str, weight: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    references = {candidate_id: load_artifact_pair(train, test, candidate_id) for candidate_id in [178, 287, 329, 335]}
    splits = load_split_manifest(SPLIT_MANIFEST_PATH)
    subject_hole_folds = splits["subject_hole_5"]

    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    x_train, feature_names = sequence_summary_features(train_tensor, channel_names)
    x_test, _ = sequence_summary_features(test_tensor, channel_names)

    split_source_values, split_covered = split_source_predictions(x_train, train, subject_hole_folds)
    _source_train_full, source_test_full, source_details = train_full_sequence_source(x_train, x_test, train)

    target_idx = TARGETS.index(target)
    oof = base_oof.copy()
    test_pred = base_test.copy()
    oof[target] = np.where(
        split_covered[:, target_idx],
        np.clip(
            (1.0 - weight) * base_oof[target].astype(float).to_numpy() + weight * split_source_values[:, target_idx],
            1e-6,
            1.0 - 1e-6,
        ),
        base_oof[target].astype(float).to_numpy(),
    )
    test_pred[target] = np.clip(
        (1.0 - weight) * base_test[target].astype(float).to_numpy() + weight * source_test_full[:, target_idx],
        1e-6,
        1.0 - 1e-6,
    )

    target_scores = target_loglosses(train, oof)
    local_cv = float(np.mean(list(target_scores.values())))
    slices = slice_summary(train, oof)
    anchors = {candidate_id: references[candidate_id][1] for candidate_id in [178, 287, 329, 335]}
    geom = geometry(test_pred, anchors)

    selector_info = {
        "base_id": base_id,
        "base_source": base_source,
        "source_model": {
            "name": MODEL_SPEC.name,
            "kind": MODEL_SPEC.kind,
            "c_value": MODEL_SPEC.c_value,
            "max_depth": MODEL_SPEC.max_depth,
            "max_leaf_nodes": MODEL_SPEC.max_leaf_nodes,
        },
        "target": target,
        "weight": weight,
        "modified_targets": [target],
        "unchanged_targets": [col for col in TARGETS if col != target],
        "split_manifest": str(SPLIT_MANIFEST_PATH.relative_to(ROOT)),
        "diagnostic_reports": DIAGNOSTIC_REPORTS,
        "sequence_feature_summary": {
            "n_sequence_rows": int(len(seq_table)),
            "n_total_channels": int(len(channel_names)),
            "n_summary_features": int(x_train.shape[1]),
            "feature_family_count": int(len(feature_names) // max(len(channel_names), 1)),
        },
        "source_prediction_details": source_details[target],
        "geometry": geom,
        "slice_summary": {
            "worst_subject": slices["worst_subject"],
            "worst_subject_loss": slices["worst_subject_loss"],
            "worst_date_bin": slices["worst_date_bin"],
            "worst_date_loss": slices["worst_date_loss"],
            "worst_target": slices["worst_target"],
            "worst_target_loss": slices["worst_target_loss"],
            "late_minus_early": slices["late_minus_early"],
        },
        "robust_score": robust_score(local_cv, slices),
        "target_scores": target_scores,
        "selection_contract": (
            f"ID{base_id:03d} is kept as the public anchor. Only {target} is blended with the full-train "
            f"sequence-summary {MODEL_SPEC.name} source at weight {weight:g}. OOF {target} uses the frozen "
            "subject_hole_5 split source; test predictions use the full-train source. No DACON submission was made."
        ),
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
    target: str,
    weight: float,
    local_cv: float,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: ID{base_id:03d} {target} seq-summary w{weight:g}\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Slug: `{slug}`\n"
            f"- Local Subject-Hole CV: `{local_cv:.6f}`\n\n"
            "This is a generated-unsubmitted artifact. It is not Public 0.58 evidence and was not submitted to DACON."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/promote_id340_q2_seqsum_w001_candidate.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id} "
            f"--target {target} --weight {weight:g}"
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--slug", default="id178_q2_seqsum_lr_c03_w001_safety")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--target", default=DEFAULT_TARGET, choices=TARGETS)
    parser.add_argument("--weight", type=float, default=DEFAULT_WEIGHT)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict artifact ids must be >= 94.")
    if not 0.0 < args.weight <= 0.2:
        raise ValueError("weight must be in (0, 0.2].")

    oof, test_pred, context = build_candidate(base_id=args.base_id, target=args.target, weight=args.weight)
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

    created_at = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z")
    metadata = {
        "created_at_kst": created_at,
        "experiment_id": args.experiment_id,
        "slug": args.slug,
        "method": f"ID{args.base_id:03d} {args.target}-only sequence-summary {MODEL_SPEC.name} weight {args.weight:g}",
        "submission_file": submission_path.name,
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "subject_hole_5_q2_only_sequence_summary_split_source",
        "strict_subject_hole_comparable": True,
        "target_scores": context["target_scores"],
        "targets": TARGETS,
        "selector_info": selector_info,
        "risk_decision": {
            "public_058_plausible": False,
            "controlled_public_probe_hypothesis": True,
            "requires_explicit_user_approval_before_dacon_submit": True,
            "reason": (
                f"{args.target}-only weight {args.weight:g} is a low-movement sequence-summary variant that "
                "passed the frozen split/geometry review, but the conservative proxy remains above Public 0.58."
            ),
        },
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "notebook_file": str(notebook_path.relative_to(ROOT)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    geom = selector_info["geometry"]
    slices = selector_info["slice_summary"]
    decision_md = (
        f"# Decision note — submission {args.experiment_id:03d} {args.slug}\n\n"
        f"Created: `{created_at}`\n\n"
        "## Decision\n\n"
        "- DACON submission: **no**\n"
        "- Public 0.58 plausible: **no**\n"
        "- Controlled low-movement public probe hypothesis: **yes, still unsubmitted**\n\n"
        "## Evidence\n\n"
        f"- Local Subject-Hole CV: `{local_cv:.6f}`\n"
        f"- Robust score: `{selector_info['robust_score']:.6f}`\n"
        f"- Mean Δ vs ID178: `{geom['mean_delta_vs_178']:.6f}`\n"
        f"- Max Δ vs ID178: `{geom['max_delta_vs_178']:.6f}`\n"
        f"- Failed-direction projection: `{geom['projection_failed']:+.6f}`\n"
        f"- Worst target: `{slices['worst_target']}` `{slices['worst_target_loss']:.6f}`\n\n"
        "## Rationale\n\n"
        f"Only `{args.target}` is blended with sequence-summary `{MODEL_SPEC.name}` at weight `{args.weight:g}`. "
        f"All other targets remain ID{args.base_id:03d}. This is the safest target-subset ID340 construction lead "
        "from the report-only geometry scan, but it is not a Public 0.58 claim and needs explicit approval before "
        "any DACON submission.\n\n"
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
        target=args.target,
        weight=args.weight,
        local_cv=local_cv,
    )

    notes = (
        f"Generated ID{args.base_id:03d} {args.target}-only sequence-summary {MODEL_SPEC.name} weight {args.weight:g} "
        f"safety artifact. All non-{args.target} targets stay at ID{args.base_id:03d}. This was selected by the "
        "post-ID340 sequence-summary subset/micro geometry scans; "
        "not Public 0.58-plausible; no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"ID{args.base_id:03d} {args.target}-only seq-summary {MODEL_SPEC.name} w{args.weight:g}",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
