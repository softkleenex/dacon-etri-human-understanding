#!/usr/bin/env python3
"""Create an Overleaf/TeX-ready source archive for the paper package.

This creates a non-score-bearing ZIP archive from paper source files only. It
does not create predictions, feature matrices, submission CSVs, Kaggle actions,
submission-log rows, or DACON submissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "docs" / "paper_submission_20260626"
BUILD_DIR = ROOT / "build" / "paper_submission_20260626"
REPORTS_DIR = ROOT / "reports"
DEFAULT_ARCHIVE = BUILD_DIR / "paper_submission_20260626_source.zip"
DEFAULT_OUTPUT = REPORTS_DIR / "paper_submission_source_archive_20260625.json"
KST = timezone(timedelta(hours=9))

SOURCE_FILES = [
    PAPER_DIR / "paper.tex",
    PAPER_DIR / "references.bib",
    PAPER_DIR / "README.md",
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
    ROOT / "docs" / "PAPER_SUBMISSION_CHECKLIST_20260626.md",
]

FORBIDDEN_PATTERNS = {
    "secret_key_like": r"serviceKey\s*=|[0-9a-f]{64}",
    "dacon_token": r"DACON_TOKEN\s*=|DACON_TEAMNAME\s*=",
    "kaggle_token": r"\"key\"\s*:\s*\"[^\"]+\"",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_forbidden(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        name
        for name, pattern in FORBIDDEN_PATTERNS.items()
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    ]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    now = datetime.now(KST)
    archive = args.archive if args.archive.is_absolute() else ROOT / args.archive
    missing = [path for path in SOURCE_FILES if not path.exists()]
    forbidden_hits = [
        {"path": rel(path), "patterns": hits}
        for path in SOURCE_FILES
        if path.exists()
        for hits in [scan_forbidden(path)]
        if hits
    ]
    archive_ready = not missing and not forbidden_hits
    file_entries = [
        {
            "path": rel(path),
            "archive_name": path.name if path.parent == PAPER_DIR else f"support/{path.name}",
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "bytes": path.stat().st_size if path.exists() else None,
        }
        for path in SOURCE_FILES
    ]
    return {
        "created_at_kst": now.strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "paper_submission_source_archive",
        "purpose": "Create an Overleaf/TeX-ready source archive from paper package files.",
        "archive_path": rel(archive),
        "archive_ready": archive_ready,
        "missing_files": [rel(path) for path in missing],
        "forbidden_hits": forbidden_hits,
        "files": file_entries,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def write_archive(report: dict[str, Any], archive: Path) -> None:
    if not report["archive_ready"]:
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_handle:
        for entry in report["files"]:
            source = ROOT / entry["path"]
            zip_handle.write(source, arcname=entry["archive_name"])
        zip_handle.writestr("manifest.json", json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paper submission source archive",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is a non-score-bearing paper source package. It creates no predictions, feature matrices, submission CSVs, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Archive ready: `{report['archive_ready']}`",
        f"- Archive path: `{report['archive_path']}`",
        f"- Missing files: `{len(report['missing_files'])}`",
        f"- Forbidden hits: `{len(report['forbidden_hits'])}`",
        "",
        "## Files",
        "",
        "| archive name | source | bytes |",
        "|---|---|---:|",
    ]
    lines.extend(
        f"| `{entry['archive_name']}` | `{entry['path']}` | {entry['bytes'] if entry['bytes'] is not None else ''} |"
        for entry in report["files"]
    )
    lines.extend(
        [
            "",
            "## Missing Files",
            "",
            "```json",
            json.dumps(report["missing_files"], indent=2, sort_keys=True),
            "```",
            "",
            "## Forbidden Hits",
            "",
            "```json",
            json.dumps(report["forbidden_hits"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = args.archive if args.archive.is_absolute() else ROOT / args.archive
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args)
    write_archive(report, archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    if report["archive_ready"]:
        print(f"Wrote {rel(archive)}")
    print(f"Archive ready: {report['archive_ready']}")


if __name__ == "__main__":
    main()
