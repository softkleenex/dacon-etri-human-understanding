#!/usr/bin/env python3
"""Audit paper.tex length and structure before PDF compilation.

This is an approximate source-level check for the four-page paper. It does not
compile a PDF and creates no predictions, feature matrices, submission CSVs,
Kaggle actions, submission-log rows, or DACON submissions.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "docs" / "paper_submission_20260626"
DEFAULT_TEX = PAPER_DIR / "paper.tex"
DEFAULT_OUTPUT = PAPER_DIR / "paper_length_budget.json"
COMPILED_PDF = ROOT / "build" / "paper_submission_20260626" / "compile" / "paper.pdf"
KST = timezone(timedelta(hours=9))

MAX_PAGES = 4.0
WORD_BUDGET = 2600
ABSTRACT_WORD_BUDGET = 250


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def extract(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1) if match else ""


def strip_latex(text: str) -> str:
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{(?:table|equation)\}.*?\\end\{(?:table|equation)\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\cite\{[^}]+\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = text.replace("\\&", "&")
    return " ".join(text.split())


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", strip_latex(text)))


def section_counts(text: str) -> list[dict[str, Any]]:
    matches = list(re.finditer(r"\\section\{([^{}]+)\}", text))
    rows: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end]
        rows.append({"section": match.group(1), "words": word_count(body)})
    return rows


def pdf_page_count(path: Path) -> int | None:
    if not path.exists() or shutil.which("pdfinfo") is None:
        return None
    result = subprocess.run(["pdfinfo", str(path)], check=False, capture_output=True, text=True)
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    return int(match.group(1)) if result.returncode == 0 and match else None


def build_report(tex_path: Path) -> dict[str, Any]:
    text = tex_path.read_text(encoding="utf-8")
    abstract = extract(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text)
    body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", " ", text, flags=re.DOTALL)
    body = re.sub(r"\\begin\{IEEEkeywords\}.*?\\end\{IEEEkeywords\}", " ", body, flags=re.DOTALL)
    table_count = len(re.findall(r"\\begin\{table\}", text))
    figure_count = len(re.findall(r"\\begin\{figure\*?\}", text))
    equation_count = len(re.findall(r"\\begin\{equation\}", text))
    abstract_words = word_count(abstract)
    body_words = word_count(body)
    total_words = abstract_words + body_words
    estimated_pages = (total_words / 700.0) + (table_count * 0.30) + (equation_count * 0.12) + (figure_count * 0.45)
    actual_pdf_pages = pdf_page_count(COMPILED_PDF)
    source_budget_pass = total_words <= WORD_BUDGET and estimated_pages <= MAX_PAGES
    page_budget_pass = actual_pdf_pages <= MAX_PAGES if actual_pdf_pages is not None else source_budget_pass
    now = datetime.now(KST)
    return {
        "created_at_kst": now.strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "paper_length_budget",
        "source_tex": rel(tex_path),
        "abstract_words": abstract_words,
        "body_words": body_words,
        "total_words": total_words,
        "section_count": len(re.findall(r"\\section\{", text)),
        "subsection_count": len(re.findall(r"\\subsection\{", text)),
        "table_count": table_count,
        "figure_count": figure_count,
        "equation_count": equation_count,
        "section_word_counts": section_counts(text),
        "word_budget": WORD_BUDGET,
        "abstract_word_budget": ABSTRACT_WORD_BUDGET,
        "max_pages": MAX_PAGES,
        "estimated_pages_from_source": round(estimated_pages, 3),
        "source_budget_pass": source_budget_pass,
        "compiled_pdf": rel(COMPILED_PDF),
        "actual_pdf_pages": actual_pdf_pages,
        "actual_pdf_page_budget_pass": actual_pdf_pages is not None and actual_pdf_pages <= MAX_PAGES,
        "length_audit_pass": abstract_words <= ABSTRACT_WORD_BUDGET and page_budget_pass,
        "pdf_compile_required_for_final_page_count": actual_pdf_pages is None,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Length Budget",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "Source length is approximate. When a compiled PDF exists, its actual page count is authoritative.",
        "",
        "## Decision",
        "",
        f"- Length audit pass: `{report['length_audit_pass']}`",
        f"- Estimated pages from source: `{report['estimated_pages_from_source']}` / `{report['max_pages']}`",
        f"- Actual compiled PDF pages: `{report['actual_pdf_pages']}` / `{report['max_pages']}`",
        f"- Compiled PDF: `{report['compiled_pdf']}`",
        f"- Total words: `{report['total_words']}` / `{report['word_budget']}`",
        f"- Abstract words: `{report['abstract_words']}` / `{report['abstract_word_budget']}`",
        f"- Tables: `{report['table_count']}`",
        f"- Figures: `{report['figure_count']}`",
        f"- Equations: `{report['equation_count']}`",
        f"- Final PDF compile required: `{report['pdf_compile_required_for_final_page_count']}`",
        "",
        "## Section Word Counts",
        "",
        "| section | words |",
        "|---|---:|",
    ]
    lines.extend(f"| {row['section']} | {row['words']} |" for row in report["section_word_counts"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tex_path = args.tex if args.tex.is_absolute() else ROOT / args.tex
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(tex_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(f"Length audit pass: {report['length_audit_pass']} | estimated_pages={report['estimated_pages_from_source']}")


if __name__ == "__main__":
    main()
