#!/usr/bin/env python3
"""Audit the deadline paper-submission package.

This is report-only. It creates no predictions, feature matrices, submission
artifacts, Kaggle actions, submission-log rows, or DACON submissions.
"""

from __future__ import annotations

import argparse
import csv
import json
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
REQUIRED_FILES = [
    ROOT / "docs" / "PAPER_SUBMISSION_DRAFT_20260626.md",
    ROOT / "docs" / "PAPER_SUBMISSION_CHECKLIST_20260626.md",
    PAPER_DIR / "README.md",
    PAPER_DIR / "paper.tex",
    PAPER_DIR / "references.bib",
    PAPER_DIR / "model_description_draft.md",
    PAPER_DIR / "id342_lineage_manifest.md",
    PAPER_DIR / "reproducibility_runbook.md",
    PAPER_DIR / "edas_submission_handoff.md",
    PAPER_DIR / "edas_metadata.json",
    PAPER_DIR / "edas_metadata.md",
    PAPER_DIR / "paper_length_budget.json",
    PAPER_DIR / "paper_length_budget.md",
    PAPER_DIR / "overleaf_compile_note.md",
    PAPER_DIR / "paper_submission_final_handoff.md",
]

REQUIRED_TEX_PATTERNS = {
    "ieee_conference_class": r"\\documentclass\[conference\]\{IEEEtran\}",
    "four_page_rule_context": r"four-page full paper",
    "current_anchor_score": r"0\.5960832519",
    "id342_anchor": r"ID342",
    "public_055_not_achieved": r"did not reach Public 0\.55",
    "failed_frontier_controls": r"ID287.*ID329|ID329.*ID287",
    "bibliography": r"\\bibliography\{references\}",
}

FORBIDDEN_PATTERNS = {
    "secret_key_fragment": r"serviceKey\s*=|[0-9a-f]{64}",
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
    field_matches = {
        field: row.get(field) == expected
        for field, expected in LINEAGE_EXPECTED.items()
        if field != "id"
    }
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
        "reproducibility_runbook.md",
        "edas_submission_handoff.md",
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
    text_by_file = {path: read_text(path) for path in REQUIRED_FILES}
    paper_text = text_by_file[PAPER_DIR / "paper.tex"]
    checklist_text = text_by_file[ROOT / "docs" / "PAPER_SUBMISSION_CHECKLIST_20260626.md"]
    lineage_text = text_by_file[PAPER_DIR / "id342_lineage_manifest.md"]
    edas_text = text_by_file[PAPER_DIR / "edas_submission_handoff.md"]
    edas_metadata_text = text_by_file[PAPER_DIR / "edas_metadata.json"]
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
            "human_upload_steps": r"Log in to EDAS.*Upload the compiled IEEE-format four-page PDF",
            "no_credentials": r"Do not expose API keys.*EDAS\s+credentials",
        },
    )
    metadata_hits = pattern_hits(
        edas_metadata_text,
        {
            "title": r"Conservative Multimodal Lifelog Modeling",
            "abstract_score_boundary": r"0\.5960832519",
            "keyword_count": r'"keyword_count":\s*6',
            "edas_link": r"https://edas\.info/newPaper\.php\?c=35149",
        },
    )
    length_budget_hits = pattern_hits(
        length_budget_text,
        {
            "length_pass": r'"length_audit_pass":\s*true',
            "final_pdf_required": r'"pdf_compile_required_for_final_page_count":\s*true',
            "max_pages": r'"max_pages":\s*4\.0',
        },
    )
    compile_note_hits = pattern_hits(
        compile_note_text,
        {
            "source_zip": r"build/paper_submission_20260626/paper_submission_20260626_source\.zip",
            "main_file": r"paper\.tex",
            "four_page_check": r"four pages|4 pages",
            "pdf_compile_boundary": r"pdflatex.*unavailable|local.*TeX.*unavailable",
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
            and all(length_budget_hits.values())
            and all(compile_note_hits.values())
            and all(final_handoff_hits.values())
            and submission_lineage["matches_expected"]
            and source_archive["matches_expected"]
            and not forbidden
        ),
        "pdflatex_available": shutil.which("pdflatex") is not None,
        "chktex_available": shutil.which("chktex") is not None,
        "human_fillins_remaining": bool(placeholders),
        "create_prediction_now": False,
        "create_submission_artifact_now": False,
        "modify_submission_log_now": False,
        "run_kaggle_now": False,
        "submit_dacon_now": False,
        "next_action": (
            "Use paper.tex as the working IEEE source draft. Human author/affiliation/EDAS fields remain; "
            "no score-bearing action is opened by this package audit."
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
        "length_budget_patterns": length_budget_hits,
        "overleaf_compile_note_patterns": compile_note_hits,
        "final_handoff_patterns": final_handoff_hits,
        "submission_lineage": submission_lineage,
        "source_archive": source_archive,
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
