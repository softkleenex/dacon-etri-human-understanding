#!/usr/bin/env python3
"""Audit the deadline paper-submission package.

This is report-only. It creates no predictions, feature matrices, submission
artifacts, Kaggle actions, submission-log rows, or DACON submissions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "paper_submission_package_audit_20260625.json"
KST = timezone(timedelta(hours=9))
SUBMISSION_LOG = ROOT / "submissions" / "submission_log.csv"
SOURCE_ARCHIVE_REPORT = REPORTS_DIR / "paper_submission_source_archive_20260625.json"
SOURCE_ARCHIVE_ZIP = ROOT / "build" / "paper_submission_20260626" / "paper_submission_20260626_source.zip"

PAPER_DIR = ROOT / "docs" / "paper_submission_20260626"
CURRENT_PDF = PAPER_DIR / "paper.pdf"
SUBMITTED_PDF = PAPER_DIR / "review_manuscript_submitted_20260717.pdf"
COMPILED_PDF = ROOT / "build" / "paper_submission_20260626" / "compile" / "paper.pdf"
EDAS_RECEIPT = PAPER_DIR / "edas_receipt.json"
LINEAGE_ATTRIBUTION_EVIDENCE = PAPER_DIR / "lineage_attribution_evidence.json"
METRICS_TRAIN = ROOT / "data" / "raw" / "ch2026_metrics_train.csv"
FEATURE_SIMILARITY_REPORT = REPORTS_DIR / "id218_public_feature_similarity_narrow_grid_20260616.json"
OOF_FILES = {
    "178": ROOT
    / "artifacts"
    / "base_predictions"
    / "base_oof_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638.csv",
    "218": ROOT / "artifacts" / "base_predictions" / "base_oof_id_218_id215_multi_signal_clean_overlay_cv0.534103.csv",
    "221": ROOT
    / "artifacts"
    / "base_predictions"
    / "base_oof_id_221_id218_public_feature_similarity_overlay_cv0.532990.csv",
    "342": ROOT / "artifacts" / "base_predictions" / "base_oof_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv",
}
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
REQUIRED_FILES = [
    ROOT / "docs" / "PAPER_SUBMISSION_DRAFT_20260626.md",
    ROOT / "docs" / "PAPER_SUBMISSION_CHECKLIST_20260626.md",
    PAPER_DIR / "README.md",
    PAPER_DIR / "paper.tex",
    CURRENT_PDF,
    SUBMITTED_PDF,
    PAPER_DIR / "references.bib",
    PAPER_DIR / "model_description_draft.md",
    PAPER_DIR / "id342_lineage_manifest.md",
    LINEAGE_ATTRIBUTION_EVIDENCE,
    PAPER_DIR / "reviewer_response.md",
    PAPER_DIR / "reproducibility_runbook.md",
    PAPER_DIR / "edas_submission_handoff.md",
    EDAS_RECEIPT,
    PAPER_DIR / "edas_metadata.json",
    PAPER_DIR / "edas_metadata.md",
    PAPER_DIR / "paper_length_budget.json",
    PAPER_DIR / "paper_length_budget.md",
    PAPER_DIR / "overleaf_compile_note.md",
    PAPER_DIR / "paper_submission_final_handoff.md",
]

REQUIRED_TEX_PATTERNS = {
    "ieee_conference_class": r"\\documentclass\[conference\]\{IEEEtran\}",
    "current_title": r"Anchor-Constrained Feature-Similarity Ensembling",
    "current_anchor_score": r"0\.5960832519",
    "id342_anchor": r"ID342",
    "id218_predecessor": r"ID218.*pre-existing multi-signal ensemble",
    "similarity_attribution_boundary": r"public contribution is unidentified",
    "posthoc_validation_boundary": r"retrospectively.*not pre-submission promotion evidence",
    "processed_feature_count": r"processed representation has 232 numeric features",
    "overlapping_similarity_families": r"potentially overlapping feature families.*Counts are not additive",
    "failed_frontier_controls": r"ID287.*ID329|ID329.*ID287",
    "leaderboard_adaptation_boundary": r"leaderboard-adapted competition findings",
    "alpha_sensitivity": r"0\.538638, 0\.535813, 0\.534000, 0\.533074, and 0\.532990",
    "similarity_search_size": r"216 settings per target.*1,512 target/configuration evaluations",
    "similarity_fold_dependence": r"21 of 35 target-fold losses",
    "transductive_deployment_check": r"0\.532990 to 0\.533015",
    "retrospective_rank_boundary": r"neither independently validates the promotion rule nor constitutes prospective validation",
    "bibliography": r"\\bibliography\{references\}",
}
EXPECTED_ATTRIBUTION_METRICS = {
    "id178_mean_log_loss": 0.5386377373903545,
    "id218_mean_log_loss": 0.53410316496098,
    "id221_mean_log_loss": 0.5329901873426955,
    "id342_mean_log_loss": 0.5340000057700848,
    "midpoint_id178_id218_mean_log_loss": 0.5345081742255229,
    "midpoint_id178_id221_mean_log_loss": 0.5340000057700848,
    "similarity_increment": 0.0005081684554380,
    "total_id178_to_id342_gain": 0.0046377316202697,
    "similarity_share": 0.1095726309856,
    "similarity_target_fold_comparisons": 35,
    "similarity_target_fold_wins_vs_id218": 21,
    "train_only_preprocessing_delta": 0.0000251989537261,
    "train_only_preprocessing_id221_mean_log_loss": 0.5330153862964216,
}
EXPECTED_ALPHA_SENSITIVITY = {
    "mean_log_loss": {
        "0.00": 0.5386377373903545,
        "0.25": 0.5358126881150828,
        "0.50": 0.5340000057700848,
        "0.75": 0.5330741072081471,
        "1.00": 0.5329901873426955,
    },
    "fold_wins_vs_id178": {"0.00": 0, "0.25": 5, "0.50": 5, "0.75": 4, "1.00": 4},
}
EXPECTED_ATTRIBUTION_METHOD = {
    "id221_predecessor_id": 218,
    "id342_formula": "0.5 * ID178 + 0.5 * ID221",
    "midpoint_ablation_formula": "0.5 * ID178 + 0.5 * ID218",
    "row_count": 450,
    "similarity_candidates_per_target": 216,
    "similarity_target_configuration_evaluations": 1512,
    "target_count": 7,
}
EXPECTED_ATTRIBUTION_SOURCES = {
    "metrics_train": {
        "path": "data/raw/ch2026_metrics_train.csv",
        "sha256": "650af4f15ad7b884febd331a2157a946bf9aa6e23347a6cd381916ff570e9a88",
    },
    "feature_similarity_report": {
        "path": "reports/id218_public_feature_similarity_narrow_grid_20260616.json",
        "sha256": "73a7b0857663a8cd8d58329ff1b0dd18fa3d5014a20f5ad790470b21610a7889",
    },
    "id178_oof": {
        "path": "artifacts/base_predictions/base_oof_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638.csv",
        "sha256": "43f2a245c64401fe576e46c99191221b6f1c3cef7e95b2f29997a57d063e2d49",
    },
    "id218_oof": {
        "path": "artifacts/base_predictions/base_oof_id_218_id215_multi_signal_clean_overlay_cv0.534103.csv",
        "sha256": "cb78ba8eacb2cf9f8125ba23d0415fdb746eadcf14fa720fef329cf5099fb5cb",
    },
    "id221_oof": {
        "path": "artifacts/base_predictions/base_oof_id_221_id218_public_feature_similarity_overlay_cv0.532990.csv",
        "sha256": "8f3a295a87485c070dd94c6fd20d813d2a21b838741b6ebd253d72459ee5b32e",
    },
    "id342_oof": {
        "path": "artifacts/base_predictions/base_oof_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv",
        "sha256": "a5129ac12a909978a6424d98c7c38fd14b66a262eb9cb294ec937d0ee47a26cf",
    },
}

FORBIDDEN_PATTERNS = {
    "secret_key_fragment": r"serviceKey\s*[:=]|(?:api|secret|access)[_-]?key\s*[:=]",
    "false_public055_claim": r"(achieved|reached|beat|below)\s+Public\s+0\.55",
    "false_prize_claim": r"prize[- ]?range\s+(result|score|solution)|in\s+the\s+prize[- ]?range",
}

LINEAGE_EXPECTED = {
    "id": "342",
    "submission_file": "submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv",
    "notebook_file": "notebooks/history/submission_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.ipynb",
    "public_lb_score": "0.5960832519",
    "local_cv_score": "0.534000",
    "is_submitted_bool": "True",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_status() -> dict[str, Any]:
    receipt: dict[str, Any] = {}
    receipt_parse_ok = False
    if EDAS_RECEIPT.exists():
        try:
            receipt = json.loads(EDAS_RECEIPT.read_text(encoding="utf-8"))
            receipt_parse_ok = True
        except json.JSONDecodeError:
            receipt = {}
    current_sha = sha256_file(CURRENT_PDF)
    compiled_sha = sha256_file(COMPILED_PDF)
    submitted_sha = sha256_file(SUBMITTED_PDF)
    receipt_sha = receipt.get("submitted_pdf_sha256")
    receipt_pdf = receipt.get("submitted_pdf")
    return {
        "current_pdf": rel(CURRENT_PDF),
        "compiled_pdf": rel(COMPILED_PDF),
        "submitted_pdf": rel(SUBMITTED_PDF),
        "current_sha256": current_sha,
        "compiled_sha256": compiled_sha,
        "submitted_sha256": submitted_sha,
        "receipt_parse_ok": receipt_parse_ok,
        "receipt_submitted_pdf": receipt_pdf,
        "receipt_submitted_pdf_sha256": receipt_sha,
        "current_matches_compiled": current_sha is not None and current_sha == compiled_sha,
        "submitted_matches_receipt": (
            submitted_sha is not None and submitted_sha == receipt_sha and receipt_pdf == rel(SUBMITTED_PDF)
        ),
        "matches_expected": (
            current_sha is not None
            and current_sha == compiled_sha
            and submitted_sha is not None
            and submitted_sha == receipt_sha
            and receipt_pdf == rel(SUBMITTED_PDF)
        ),
    }


def read_target_rows(path: Path, date_column: str) -> dict[tuple[str, str], dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["subject_id"], row[date_column]): {target: float(row[target]) for target in TARGETS} for row in rows}


def mean_log_loss(
    labels: dict[tuple[str, str], dict[str, float]],
    predictions: dict[tuple[str, str], dict[str, float]],
) -> float:
    losses: list[float] = []
    for target in TARGETS:
        target_loss = 0.0
        for key, row in labels.items():
            probability = min(max(predictions[key][target], 1e-15), 1.0 - 1e-15)
            label = row[target]
            target_loss -= label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability)
        losses.append(target_loss / len(labels))
    return sum(losses) / len(losses)


def midpoint_predictions(
    left: dict[tuple[str, str], dict[str, float]],
    right: dict[tuple[str, str], dict[str, float]],
) -> dict[tuple[str, str], dict[str, float]]:
    return {key: {target: 0.5 * left[key][target] + 0.5 * right[key][target] for target in TARGETS} for key in left}


def lineage_attribution_status(paper_text: str) -> dict[str, Any]:
    if not LINEAGE_ATTRIBUTION_EVIDENCE.exists():
        return {"evidence_exists": False, "matches_expected": False}
    try:
        evidence = json.loads(LINEAGE_ATTRIBUTION_EVIDENCE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {"evidence_exists": True, "parse_error": str(exc), "matches_expected": False}

    method = evidence.get("method", {})
    metrics = evidence.get("metrics", {})
    source_artifacts = evidence.get("source_artifacts", {})
    if not isinstance(method, dict) or not isinstance(metrics, dict) or not isinstance(source_artifacts, dict):
        return {
            "evidence_exists": True,
            "evidence_path": rel(LINEAGE_ATTRIBUTION_EVIDENCE),
            "schema_valid": False,
            "matches_expected": False,
        }
    expected_sources = {
        "metrics_train": METRICS_TRAIN,
        "feature_similarity_report": FEATURE_SIMILARITY_REPORT,
        **{f"id{name}_oof": path for name, path in OOF_FILES.items()},
    }
    source_records = {name: value if isinstance(value, dict) else {} for name, value in source_artifacts.items()}
    metric_checks = {
        name: isinstance(metrics.get(name), int | float) and abs(metrics[name] - expected) < 1e-12
        for name, expected in EXPECTED_ATTRIBUTION_METRICS.items()
    }
    alpha_losses = metrics.get("alpha_sensitivity_mean_log_loss", {})
    alpha_fold_wins = metrics.get("alpha_sensitivity_fold_wins_vs_id178", {})
    alpha_sensitivity_checks = {
        "mean_log_loss": isinstance(alpha_losses, dict)
        and all(
            isinstance(alpha_losses.get(alpha), int | float) and abs(alpha_losses[alpha] - expected) < 1e-12
            for alpha, expected in EXPECTED_ALPHA_SENSITIVITY["mean_log_loss"].items()
        ),
        "fold_wins_vs_id178": alpha_fold_wins == EXPECTED_ALPHA_SENSITIVITY["fold_wins_vs_id178"],
    }
    method_checks = {name: method.get(name) == expected for name, expected in EXPECTED_ATTRIBUTION_METHOD.items()}
    if all(metric_checks.values()):
        metric_consistency_checks = {
            "similarity_increment": abs(
                metrics["midpoint_id178_id218_mean_log_loss"]
                - metrics["id342_mean_log_loss"]
                - metrics["similarity_increment"]
            )
            < 1e-12,
            "total_gain": abs(
                metrics["id178_mean_log_loss"] - metrics["id342_mean_log_loss"] - metrics["total_id178_to_id342_gain"]
            )
            < 1e-12,
            "similarity_share": abs(
                metrics["similarity_increment"] / metrics["total_id178_to_id342_gain"] - metrics["similarity_share"]
            )
            < 1e-12,
        }
    else:
        metric_consistency_checks = dict.fromkeys(("similarity_increment", "total_gain", "similarity_share"), False)
    source_records_valid = {
        name: source_records.get(name) == EXPECTED_ATTRIBUTION_SOURCES[name] for name in expected_sources
    }
    local_hash_checks = {
        name: (sha256_file(path) == source_records.get(name, {}).get("sha256") if path.exists() else None)
        for name, path in expected_sources.items()
    }
    local_paths = [METRICS_TRAIN, FEATURE_SIMILARITY_REPORT, *OOF_FILES.values()]
    local_recomputation = {
        "available": all(path.exists() for path in local_paths),
        "missing_files": [rel(path) for path in local_paths if not path.exists()],
        "matches_evidence": None,
    }
    paper_claims = {
        "midpoint_score": bool(re.search(r"scores 0\.534508", paper_text)),
        "similarity_increment": bool(re.search(r"only 0\.000508", paper_text)),
        "total_gain": bool(re.search(r"0\.004638 gain", paper_text)),
        "similarity_share": bool(re.search(r"11\.0\\%", paper_text)),
        "public_attribution_boundary": bool(re.search(r"public contribution is unidentified", paper_text)),
    }
    portable_matches = (
        all(method_checks.values())
        and all(metric_checks.values())
        and all(alpha_sensitivity_checks.values())
        and all(metric_consistency_checks.values())
        and all(source_records_valid.values())
        and all(value is not False for value in local_hash_checks.values())
        and all(paper_claims.values())
    )
    if not local_recomputation["available"]:
        return {
            "evidence_exists": True,
            "evidence_path": rel(LINEAGE_ATTRIBUTION_EVIDENCE),
            "metric_checks": metric_checks,
            "alpha_sensitivity_checks": alpha_sensitivity_checks,
            "method_checks": method_checks,
            "metric_consistency_checks": metric_consistency_checks,
            "source_records_valid": source_records_valid,
            "local_hash_checks": local_hash_checks,
            "paper_claims": paper_claims,
            "local_recomputation": local_recomputation,
            "matches_expected": portable_matches,
        }

    try:
        report = json.loads(FEATURE_SIMILARITY_REPORT.read_text(encoding="utf-8"))
        labels = read_target_rows(METRICS_TRAIN, "lifelog_date")
        predictions = {name: read_target_rows(path, "date") for name, path in OOF_FILES.items()}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        local_recomputation["parse_error"] = str(exc)
        return {
            "evidence_exists": True,
            "evidence_path": rel(LINEAGE_ATTRIBUTION_EVIDENCE),
            "metric_checks": metric_checks,
            "alpha_sensitivity_checks": alpha_sensitivity_checks,
            "method_checks": method_checks,
            "metric_consistency_checks": metric_consistency_checks,
            "source_records_valid": source_records_valid,
            "local_hash_checks": local_hash_checks,
            "paper_claims": paper_claims,
            "local_recomputation": local_recomputation,
            "matches_expected": False,
        }
    aligned = all(set(rows) == set(labels) for rows in predictions.values())
    if not aligned:
        local_recomputation["row_keys_aligned"] = False
        return {
            "evidence_exists": True,
            "evidence_path": rel(LINEAGE_ATTRIBUTION_EVIDENCE),
            "metric_checks": metric_checks,
            "alpha_sensitivity_checks": alpha_sensitivity_checks,
            "method_checks": method_checks,
            "metric_consistency_checks": metric_consistency_checks,
            "source_records_valid": source_records_valid,
            "local_hash_checks": local_hash_checks,
            "paper_claims": paper_claims,
            "local_recomputation": local_recomputation,
            "matches_expected": False,
        }
    midpoint_178_218 = midpoint_predictions(predictions["178"], predictions["218"])
    midpoint_178_221 = midpoint_predictions(predictions["178"], predictions["221"])
    scores = {name: mean_log_loss(labels, rows) for name, rows in predictions.items()}
    midpoint_score = mean_log_loss(labels, midpoint_178_218)
    reconstructed_score = mean_log_loss(labels, midpoint_178_221)
    total_gain = scores["178"] - scores["342"]
    similarity_increment = midpoint_score - scores["342"]
    similarity_share = similarity_increment / total_gain
    max_reconstruction_error = max(
        abs(predictions["342"][key][target] - midpoint_178_221[key][target]) for key in labels for target in TARGETS
    )
    recomputed = {
        "id178_mean_log_loss": scores["178"],
        "id218_mean_log_loss": scores["218"],
        "id221_mean_log_loss": scores["221"],
        "id342_mean_log_loss": scores["342"],
        "midpoint_id178_id218_mean_log_loss": midpoint_score,
        "midpoint_id178_id221_mean_log_loss": reconstructed_score,
        "similarity_increment": similarity_increment,
        "total_id178_to_id342_gain": total_gain,
        "similarity_share": similarity_share,
    }
    recomputation_checks = {name: abs(value - metrics[name]) < 1e-12 for name, value in recomputed.items()}
    recomputation_checks["id342_reconstruction"] = max_reconstruction_error < 1e-12
    local_recomputation.update(
        {
            "row_keys_aligned": True,
            "id221_predecessor_id": report.get("base_id"),
            "metrics": recomputed,
            "max_id342_reconstruction_error": max_reconstruction_error,
            "checks": recomputation_checks,
            "matches_evidence": report.get("base_id") == 218 and all(recomputation_checks.values()),
        }
    )
    return {
        "evidence_exists": True,
        "evidence_path": rel(LINEAGE_ATTRIBUTION_EVIDENCE),
        "metric_checks": metric_checks,
        "alpha_sensitivity_checks": alpha_sensitivity_checks,
        "method_checks": method_checks,
        "metric_consistency_checks": metric_consistency_checks,
        "source_records_valid": source_records_valid,
        "local_hash_checks": local_hash_checks,
        "paper_claims": paper_claims,
        "local_recomputation": local_recomputation,
        "matches_expected": portable_matches and local_recomputation["matches_evidence"] is True,
    }


def pattern_hits(text: str, patterns: dict[str, str]) -> dict[str, bool]:
    return {name: bool(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)) for name, pattern in patterns.items()}


def forbidden_hits(text_by_file: dict[Path, str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for path, text in text_by_file.items():
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                hits.append({"file": rel(path), "pattern": name})
    return hits


def submission_lineage_status() -> dict[str, Any]:
    if not SUBMISSION_LOG.exists():
        return {"exists": False, "path": rel(SUBMISSION_LOG), "row_found": False, "matches_expected": False}
    with SUBMISSION_LOG.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matching = [row for row in rows if row.get("id") == LINEAGE_EXPECTED["id"]]
    row = matching[0] if matching else {}
    field_matches = {field: row.get(field) == expected for field, expected in LINEAGE_EXPECTED.items() if field != "id"}
    artifact_paths = [
        ROOT / "submissions" / LINEAGE_EXPECTED["submission_file"],
        ROOT / LINEAGE_EXPECTED["notebook_file"],
        ROOT / "reports" / "decision_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.md",
        ROOT / "reports" / "dacon_three_slot_submission_result_20260623.md",
    ]
    artifact_status = {rel(path): path.exists() for path in artifact_paths}
    return {
        "exists": True,
        "path": rel(SUBMISSION_LOG),
        "row_found": bool(matching),
        "matching_row_count": len(matching),
        "field_matches": field_matches,
        "artifact_status": artifact_status,
        "matches_expected": bool(matching) and all(field_matches.values()) and all(artifact_status.values()),
    }


def source_archive_status() -> dict[str, Any]:
    report: dict[str, Any] = {}
    parse_ok = False
    if SOURCE_ARCHIVE_REPORT.exists():
        try:
            report = json.loads(SOURCE_ARCHIVE_REPORT.read_text(encoding="utf-8"))
            parse_ok = True
        except json.JSONDecodeError:
            report = {}
    archive_names: list[str] = []
    zip_read_ok = False
    if SOURCE_ARCHIVE_ZIP.exists():
        try:
            with zipfile.ZipFile(SOURCE_ARCHIVE_ZIP) as zip_handle:
                archive_names = sorted(zip_handle.namelist())
            zip_read_ok = True
        except zipfile.BadZipFile:
            zip_read_ok = False
    required_names = {
        "paper.tex",
        "references.bib",
        "README.md",
        "model_description_draft.md",
        "id342_lineage_manifest.md",
        "lineage_attribution_evidence.json",
        "reviewer_response.md",
        "reproducibility_runbook.md",
        "edas_submission_handoff.md",
        "edas_receipt.json",
        "edas_metadata.json",
        "edas_metadata.md",
        "paper_length_budget.json",
        "paper_length_budget.md",
        "overleaf_compile_note.md",
        "paper_submission_final_handoff.md",
        "support/PAPER_SUBMISSION_CHECKLIST_20260626.md",
        "manifest.json",
    }
    archive_name_set = set(archive_names)
    return {
        "report_path": rel(SOURCE_ARCHIVE_REPORT),
        "report_exists": SOURCE_ARCHIVE_REPORT.exists(),
        "report_parse_ok": parse_ok,
        "report_archive_ready": report.get("archive_ready") if parse_ok else None,
        "zip_path": rel(SOURCE_ARCHIVE_ZIP),
        "zip_exists": SOURCE_ARCHIVE_ZIP.exists(),
        "zip_read_ok": zip_read_ok,
        "required_names_present": {name: name in archive_name_set for name in sorted(required_names)},
        "matches_expected": (
            parse_ok
            and report.get("archive_ready") is True
            and SOURCE_ARCHIVE_ZIP.exists()
            and zip_read_ok
            and required_names.issubset(archive_name_set)
        ),
    }


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    now = datetime.now(KST)
    text_by_file = {
        path: read_text(path) for path in REQUIRED_FILES if path.suffix.lower() in {".bib", ".json", ".md", ".tex"}
    }
    paper_text = text_by_file[PAPER_DIR / "paper.tex"]
    checklist_text = text_by_file[ROOT / "docs" / "PAPER_SUBMISSION_CHECKLIST_20260626.md"]
    lineage_text = text_by_file[PAPER_DIR / "id342_lineage_manifest.md"]
    edas_text = text_by_file[PAPER_DIR / "edas_submission_handoff.md"]
    edas_metadata_text = text_by_file[PAPER_DIR / "edas_metadata.json"]
    edas_receipt_text = text_by_file[EDAS_RECEIPT]
    length_budget_text = text_by_file[PAPER_DIR / "paper_length_budget.json"]
    compile_note_text = text_by_file[PAPER_DIR / "overleaf_compile_note.md"]
    final_handoff_text = text_by_file[PAPER_DIR / "paper_submission_final_handoff.md"]
    required_file_status = [{"path": rel(path), "exists": path.exists()} for path in REQUIRED_FILES]
    tex_hits = pattern_hits(paper_text, REQUIRED_TEX_PATTERNS)
    checklist_hits = pattern_hits(
        checklist_text,
        {
            "four_page_rule": r"4`?-page full paper|4`?\s*pages?",
            "human_fillins": r"author|affiliation|EDAS",
            "no_dacon_spend_for_paper": r"Do not spend DACON submissions solely to improve the paper story",
        },
    )
    lineage_hits = pattern_hits(
        lineage_text,
        {
            "notebook_csv_log_public_chain": (
                r"notebooks/history/.*-> submissions/.*-> submissions/submission_log\.csv row id=342"
                r".*-> DACON submission id 1467937"
            ),
            "public_score": r"0\.5960832519",
            "public_055_boundary": r"not a prize-range or Public `0\.55` result",
        },
    )
    edas_hits = pattern_hits(
        edas_text,
        {
            "ictc_submission_page": r"https://ictc\.org/authors_paperSubmission",
            "edas_submission_link": r"https://edas\.info/newPaper\.php\?c=35149",
            "dacon_rules_page": r"https://dacon\.io/en/competitions/official/236690/overview/rules",
            "review_upload_completed": r"Review manuscript uploaded:\s*`True`",
            "no_credentials": r"Do not expose API keys.*EDAS\s+credentials",
        },
    )
    metadata_hits = pattern_hits(
        edas_metadata_text,
        {
            "title": r"Anchor-Constrained Feature-Similarity Ensembling",
            "abstract_score_boundary": r"0\.5960832519",
            "keyword_count": r'"keyword_count":\s*6',
            "topic": r'"topic":\s*"Artificial Intelligence and Machine Learning"',
            "affiliation": r'"affiliation":\s*"Kyungpook National University"',
            "author_count": r'"author_count":\s*1',
            "edas_paper_id": r'"edas_paper_id":\s*"1571325954"',
            "active_with_manuscript": r'"edas_status":\s*"Active \(has manuscript\)"',
            "review_manuscript_uploaded": r'"review_manuscript_uploaded":\s*true',
            "edas_link": r"https://edas\.info/newPaper\.php\?c=35149",
            "receipt_source": r'"edas_receipt":\s*"docs/paper_submission_20260626/edas_receipt\.json"',
            "registered_title_matches_source": r'"registered_title_matches_source":\s*true',
        },
    )
    receipt_hits = pattern_hits(
        edas_receipt_text,
        {
            "paper_id": r'"paper_id":\s*"1571325954"',
            "active_with_manuscript": r'"status":\s*"Active \(has manuscript\)"',
            "review_manuscript_uploaded": r'"review_manuscript_uploaded":\s*true',
            "submitted_pdf": r'"submitted_pdf":\s*"docs/paper_submission_20260626/review_manuscript_submitted_20260717\.pdf"',
            "observation_source": r'"source":\s*"Author-confirmed EDAS paper record"',
        },
    )
    length_budget_hits = pattern_hits(
        length_budget_text,
        {
            "length_pass": r'"length_audit_pass":\s*true',
            "actual_pdf_pages": r'"actual_pdf_pages":\s*4',
            "final_pdf_verified": r'"pdf_compile_required_for_final_page_count":\s*false',
            "max_pages": r'"max_pages":\s*4\.0',
        },
    )
    compile_note_hits = pattern_hits(
        compile_note_text,
        {
            "source_zip": r"build/paper_submission_20260626/paper_submission_20260626_source\.zip",
            "main_file": r"paper\.tex",
            "four_page_check": r"four pages|4 pages",
            "local_pdf_verified": r"tectonic.*four pages|four pages.*tectonic",
        },
    )
    final_handoff_hits = pattern_hits(
        final_handoff_text,
        {
            "edas_link": r"https://edas\.info/newPaper\.php\?c=35149",
            "source_zip": r"build/paper_submission_20260626/paper_submission_20260626_source\.zip",
            "id342_score": r"ID342.*0\.5960832519|0\.5960832519.*ID342",
            "public055_boundary": r"Public `?0\.55`?.*not achieved|not achieved.*Public `?0\.55`?",
            "no_dacon_now": r"DACON submit now: `False`",
        },
    )
    submission_lineage = submission_lineage_status()
    source_archive = source_archive_status()
    pdfs = pdf_status()
    attribution = lineage_attribution_status(paper_text)
    forbidden = forbidden_hits(text_by_file)
    placeholders = [
        {"file": rel(path), "token": token}
        for path, text in text_by_file.items()
        for token in ("To Be Filled", "to be filled", "Author Name")
        if token in text
    ]
    decision = {
        "package_audit_pass": (
            all(item["exists"] for item in required_file_status)
            and all(tex_hits.values())
            and all(checklist_hits.values())
            and all(lineage_hits.values())
            and all(edas_hits.values())
            and all(metadata_hits.values())
            and all(receipt_hits.values())
            and all(length_budget_hits.values())
            and all(compile_note_hits.values())
            and all(final_handoff_hits.values())
            and submission_lineage["matches_expected"]
            and source_archive["matches_expected"]
            and pdfs["matches_expected"]
            and attribution["matches_expected"]
            and not forbidden
        ),
        "pdflatex_available": shutil.which("pdflatex") is not None,
        "tectonic_available": shutil.which("tectonic") is not None,
        "chktex_available": shutil.which("chktex") is not None,
        "human_fillins_remaining": bool(placeholders),
        "create_prediction_now": False,
        "create_submission_artifact_now": False,
        "modify_submission_log_now": False,
        "run_kaggle_now": False,
        "submit_dacon_now": False,
        "next_action": (
            "Review submission is complete. Monitor the EDAS email/status and do not use Final manuscript "
            "unless acceptance or organizer instructions open the camera-ready stage."
        ),
    }
    return {
        "created_at_kst": now.strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "paper_submission_package_audit",
        "purpose": "Check the deadline paper package for required files, claim boundaries, and secret hygiene.",
        "required_file_status": required_file_status,
        "tex_required_patterns": tex_hits,
        "checklist_required_patterns": checklist_hits,
        "lineage_required_patterns": lineage_hits,
        "edas_required_patterns": edas_hits,
        "edas_metadata_patterns": metadata_hits,
        "edas_receipt_patterns": receipt_hits,
        "length_budget_patterns": length_budget_hits,
        "overleaf_compile_note_patterns": compile_note_hits,
        "final_handoff_patterns": final_handoff_hits,
        "submission_lineage": submission_lineage,
        "source_archive": source_archive,
        "pdf_status": pdfs,
        "lineage_attribution": attribution,
        "forbidden_hits": forbidden,
        "human_fillins": placeholders,
        "decision": decision,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Paper submission package audit",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Package audit pass: `{decision['package_audit_pass']}`",
        f"- pdflatex available: `{decision['pdflatex_available']}`",
        f"- tectonic available: `{decision['tectonic_available']}`",
        f"- chktex available: `{decision['chktex_available']}`",
        f"- Human fill-ins remaining: `{decision['human_fillins_remaining']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['submit_dacon_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Required Files",
        "",
    ]
    lines.extend(f"- `{item['path']}`: `{item['exists']}`" for item in report["required_file_status"])
    lines.extend(
        [
            "",
            "## Required Claim Patterns",
            "",
            "```json",
            json.dumps(report["tex_required_patterns"], indent=2, sort_keys=True),
            "```",
            "",
            "## Checklist Patterns",
            "",
            "```json",
            json.dumps(report["checklist_required_patterns"], indent=2, sort_keys=True),
            "```",
            "",
            "## Lineage Patterns",
            "",
            "```json",
            json.dumps(report["lineage_required_patterns"], indent=2, sort_keys=True),
            "```",
            "",
            "## Submission Lineage",
            "",
            "```json",
            json.dumps(report["submission_lineage"], indent=2, sort_keys=True),
            "```",
            "",
            "## EDAS Handoff Patterns",
            "",
            "```json",
            json.dumps(report["edas_required_patterns"], indent=2, sort_keys=True),
            "```",
            "",
            "## EDAS Metadata Patterns",
            "",
            "```json",
            json.dumps(report["edas_metadata_patterns"], indent=2, sort_keys=True),
            "```",
            "",
            "## EDAS Receipt Patterns",
            "",
            "```json",
            json.dumps(report["edas_receipt_patterns"], indent=2, sort_keys=True),
            "```",
            "",
            "## Length Budget Patterns",
            "",
            "```json",
            json.dumps(report["length_budget_patterns"], indent=2, sort_keys=True),
            "```",
            "",
            "## Overleaf Compile Note Patterns",
            "",
            "```json",
            json.dumps(report["overleaf_compile_note_patterns"], indent=2, sort_keys=True),
            "```",
            "",
            "## Final Handoff Patterns",
            "",
            "```json",
            json.dumps(report["final_handoff_patterns"], indent=2, sort_keys=True),
            "```",
            "",
            "## Source Archive",
            "",
            "```json",
            json.dumps(report["source_archive"], indent=2, sort_keys=True),
            "```",
            "",
            "## PDF Status",
            "",
            "```json",
            json.dumps(report["pdf_status"], indent=2, sort_keys=True),
            "```",
            "",
            "## Lineage Attribution",
            "",
            "```json",
            json.dumps(report["lineage_attribution"], indent=2, sort_keys=True),
            "```",
            "",
            "## Forbidden Hits",
            "",
            "```json",
            json.dumps(report["forbidden_hits"], indent=2, sort_keys=True),
            "```",
            "",
            "## Human Fill-Ins",
            "",
            "```json",
            json.dumps(report["human_fillins"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    decision = report["decision"]
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(f"Package audit pass: {decision['package_audit_pass']} | human_fillins={decision['human_fillins_remaining']}")


if __name__ == "__main__":
    main()
