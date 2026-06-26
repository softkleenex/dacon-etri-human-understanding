#!/usr/bin/env python3
"""Promote the ID178 rollback_s1_s4 hypothesis as a full no-submit artifact.

This script creates a generated_unsubmitted submission artifact with matching
CSV, notebook, metadata JSON, base OOF/test predictions, and submission-log row.
It never submits to DACON.
"""

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
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_id178_sequence_target_rollback import VARIANTS, mean_logloss  # noqa: E402
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
from src.utils.validation import clip_probabilities  # noqa: E402

KST = timezone(timedelta(hours=9))


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


def align_predictions(pred: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    keys = frame[["subject_id", "date"]].copy()
    keys["date"] = pd.to_datetime(keys["date"]).dt.strftime("%Y-%m-%d")
    out = keys.merge(pred, on=["subject_id", "date"], how="left")
    if out[TARGETS].isna().any().any():
        missing = out[TARGETS].isna().sum().to_dict()
        raise ValueError(f"Prediction alignment failed; missing={missing}")
    return out[["subject_id", "date", *TARGETS]]


def build_variant_frame(base173: pd.DataFrame, id178: pd.DataFrame, keep_sequence_targets: list[str]) -> pd.DataFrame:
    out = base173[["subject_id", "date", *TARGETS]].copy()
    for target in keep_sequence_targets:
        out[target] = id178[target].astype(float).to_numpy()
    return clip_probabilities(out, TARGETS, eps=1e-6)


def load_rollback_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing rollback diagnostic report: {path}")
    return json.loads(path.read_text())


def write_repro_notebook(
    *,
    notebook_path: Path,
    submission_path: Path,
    metadata_path: Path,
    experiment_id: int,
    slug: str,
    local_cv: float,
    variant: str,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: ID178 sequence rollback candidate\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Variant: `{variant}`\n"
            f"- Slug: `{slug}`\n"
            f"- Local OOF CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for an immutable generated_unsubmitted artifact. "
            "No DACON submission was made."
        ),
        nbf.v4.new_code_cell(
            f"!python scripts/promote_id178_sequence_rollback_candidate.py "
            f"--experiment-id {experiment_id} --slug {slug} --variant {variant}"
        ),
        nbf.v4.new_markdown_cell(
            "### Decision note\n\n"
            "This artifact is a controlled Public <0.60 probe hypothesis, not a Public 0.58-proven candidate. "
            "Review the metadata and decision report before any DACON submission."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, default=334)
    parser.add_argument("--slug", default="id178_rollback_s1_s4_sequence_stressguard")
    parser.add_argument("--variant", default="rollback_s1_s4", choices=sorted(VARIANTS))
    parser.add_argument(
        "--rollback-report",
        type=Path,
        default=REPORTS_DIR / "id178_sequence_target_rollback_post_id329_20260617.json",
    )
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict artifact ids must be >= 94.")

    train, test, _metrics_sorted, sample_original = build_feature_frames()
    rollback_report = load_rollback_report(args.rollback_report)
    variant_spec = VARIANTS[args.variant]

    id173_oof_path = find_base_file("oof", 173)
    id173_test_path = find_base_file("test", 173)
    id178_oof_path = find_base_file("oof", 178)
    id178_test_path = find_base_file("test", 178)

    id173_oof = align_predictions(pd.read_csv(id173_oof_path), train)
    id173_test = align_predictions(pd.read_csv(id173_test_path), test)
    id178_oof = align_predictions(pd.read_csv(id178_oof_path), train)
    id178_test = align_predictions(pd.read_csv(id178_test_path), test)

    keep_targets = list(variant_spec["keep_sequence_targets"])
    oof = build_variant_frame(id173_oof, id178_oof, keep_targets)
    test_pred = build_variant_frame(id173_test, id178_test, keep_targets)
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

    out = clip_probabilities(map_predictions_to_original_sample(sample_original, test_pred), TARGETS, eps=1e-6)
    if not validate_submission(out, sample_original):
        raise ValueError("Generated submission failed sample-format validation.")

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BASE_PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    out.to_csv(submission_path, index=False)
    oof.to_csv(base_oof_path, index=False)
    test_pred.to_csv(base_test_path, index=False)

    ranked_variant = next(
        (row for row in rollback_report.get("ranked_variants", []) if row.get("variant") == args.variant),
        None,
    )
    metadata = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "experiment_id": args.experiment_id,
        "slug": args.slug,
        "variant": args.variant,
        "method": "ID178 target rollback: keep sequence targets Q1/Q2/Q3/S2, revert S1/S4 to ID173",
        "submission_file": submission_path.name,
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "keep_sequence_targets": keep_targets,
        "reverted_to_id173_targets": [target for target in TARGETS if target not in keep_targets],
        "source_files": {
            "id173_oof": str(id173_oof_path.relative_to(ROOT)),
            "id173_test": str(id173_test_path.relative_to(ROOT)),
            "id178_oof": str(id178_oof_path.relative_to(ROOT)),
            "id178_test": str(id178_test_path.relative_to(ROOT)),
            "id173_slug": candidate_slug(id173_oof_path),
            "id178_slug": candidate_slug(id178_oof_path),
        },
        "rollback_diagnostic_report": str(args.rollback_report.relative_to(ROOT)),
        "rollback_diagnostic_variant": ranked_variant,
        "feature_contract": (
            "No new model training. Builds a deterministic target-wise rollback from existing ID173 and ID178 "
            "base OOF/test artifacts. Q1/Q2/Q3/S2 use ID178 sequence overlay predictions; S1/S3/S4 use ID173 "
            "base predictions. No DACON submission was made."
        ),
        "risk_decision": {
            "public_058_plausible": False,
            "public_060_probe_hypothesis": True,
            "requires_explicit_user_approval_before_dacon_submit": True,
            "reason": "Stress diagnostics favor rollback_s1_s4 versus original ID178, but it is still same-family and only a controlled probe hypothesis.",
        },
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "notebook_file": str(notebook_path.relative_to(ROOT)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    decision_md = (
        f"# Decision note — submission {args.experiment_id:03d} {args.slug}\n\n"
        f"Created: `{metadata['created_at_kst']}`\n\n"
        "## Decision\n\n"
        "- DACON submission: **no**\n"
        "- Public 0.58 plausible: **no**\n"
        "- Public <0.60 controlled probe hypothesis: **yes, weak/moderate**\n\n"
        "## Rationale\n\n"
        "`rollback_s1_s4` keeps ID178 sequence-overlay targets `Q1`, `Q2`, `Q3`, and `S2`, "
        "while reverting fragile `S1` and `S4` to ID173. The rollback diagnostic ranked it first by "
        "stress consistency, but it remains same-family with ID178 and must not be submitted without explicit approval.\n\n"
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
        variant=args.variant,
    )

    notes = (
        "Generated ID178 rollback_s1_s4 sequence stressguard artifact. Keeps ID178 sequence overlay for "
        "Q1/Q2/Q3/S2 and reverts S1/S3/S4 to ID173. Stress diagnostics rank this rollback first, but it is "
        "not Public 0.58-plausible and no DACON submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description="ID178 sequence rollback S1/S4 stressguard",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
