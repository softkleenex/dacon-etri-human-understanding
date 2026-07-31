#!/usr/bin/env python3
"""Extract EDAS copy-paste metadata from paper.tex.

This writes non-score-bearing metadata files for the paper handoff only. It
does not create predictions, feature matrices, submission CSVs, Kaggle actions,
submission-log rows, or DACON submissions.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "docs" / "paper_submission_20260626"
DEFAULT_TEX = PAPER_DIR / "paper.tex"
DEFAULT_OUTPUT = PAPER_DIR / "edas_metadata.json"
KST = timezone(timedelta(hours=9))
EDAS_TOPIC = "Artificial Intelligence and Machine Learning"
EDAS_PAPER_ID = "1571325954"
EDAS_STATUS = "Active (has manuscript)"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def extract_required(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract {label}")
    return match.group(1)


def clean_latex_text(text: str) -> str:
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\cite\{[^}]+\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\&", "&")
    text = text.replace("--", "-")
    return " ".join(text.split())


def build_metadata(tex_path: Path) -> dict[str, Any]:
    text = tex_path.read_text(encoding="utf-8")
    author_name = clean_latex_text(
        extract_required(r"\\newcommand\{\\paperauthorname\}\{(.*?)\}", text, "paper author name")
    )
    affiliation = clean_latex_text(
        extract_required(r"\\newcommand\{\\paperaffiliation\}\{(.*?)\}", text, "paper affiliation")
    )
    email = clean_latex_text(extract_required(r"\\newcommand\{\\paperemail\}\{(.*?)\}", text, "paper email"))
    title = clean_latex_text(extract_required(r"\\title\{(.*?)\}", text, "title"))
    abstract = clean_latex_text(extract_required(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, "abstract"))
    keywords_text = clean_latex_text(
        extract_required(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", text, "IEEE keywords")
    )
    keywords = [item.strip() for item in keywords_text.split(",") if item.strip()]
    human_fillins: list[str] = []
    if "To Be Filled" in author_name:
        human_fillins.insert(0, "author name")
    now = datetime.now(KST)
    return {
        "created_at_kst": now.strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "source_tex": rel(tex_path),
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "topic": EDAS_TOPIC,
        "authors": [
            {
                "name": author_name,
                "affiliation": affiliation,
                "email": email,
                "corresponding_author": True,
            }
        ],
        "author_count": 1,
        "edas_paper_id": EDAS_PAPER_ID,
        "edas_status": EDAS_STATUS,
        "review_manuscript_uploaded": True,
        "abstract_word_count": len(abstract.split()),
        "keyword_count": len(keywords),
        "edas_submission_link": "https://edas.info/newPaper.php?c=35149",
        "ictc_submission_page": "https://ictc.org/authors_paperSubmission",
        "dacon_rules_page": "https://dacon.io/en/competitions/official/236690/overview/rules",
        "human_fillins_remaining": human_fillins,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(metadata: dict[str, Any]) -> str:
    keyword_line = ", ".join(metadata["keywords"])
    lines = [
        "# EDAS Metadata",
        "",
        f"Created: `{metadata['created_at_kst']}`",
        "",
        "This is copy-paste metadata for EDAS. It creates no predictions, feature matrices, submission CSVs, Kaggle runs, or DACON submissions.",
        "",
        "## Title",
        "",
        metadata["title"],
        "",
        "## Abstract",
        "",
        metadata["abstract"],
        "",
        "## EDAS Form Fields",
        "",
        f"- Topic: `{metadata['topic']}`",
        f"- Author count: `{metadata['author_count']}`",
        f"- Author name: `{metadata['authors'][0]['name']}`",
        f"- Affiliation: `{metadata['authors'][0]['affiliation']}`",
        f"- Email: `{metadata['authors'][0]['email']}`",
        f"- Corresponding author: `{metadata['authors'][0]['corresponding_author']}`",
        f"- EDAS paper ID: `{metadata['edas_paper_id']}`",
        f"- Status: `{metadata['edas_status']}`",
        f"- Review manuscript uploaded: `{metadata['review_manuscript_uploaded']}`",
        "",
        "## Keywords 1-6",
        "",
    ]
    lines.extend(f"- Keyword {index}: `{keyword}`" for index, keyword in enumerate(metadata["keywords"], start=1))
    lines.extend([
        "",
        f"Combined: {keyword_line}",
        "",
        "## Links",
        "",
        f"- EDAS: `{metadata['edas_submission_link']}`",
        f"- ICTC: `{metadata['ictc_submission_page']}`",
        f"- DACON rules: `{metadata['dacon_rules_page']}`",
        "",
        "## Counts",
        "",
        f"- Abstract words: `{metadata['abstract_word_count']}`",
        f"- Keywords: `{metadata['keyword_count']}`",
        "",
        "## Human Fill-Ins",
        "",
    ])
    if metadata["human_fillins_remaining"]:
        lines.extend(f"- {item}" for item in metadata["human_fillins_remaining"])
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tex_path = args.tex if args.tex.is_absolute() else ROOT / args.tex
    output = args.output if args.output.is_absolute() else ROOT / args.output
    metadata = build_metadata(tex_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(metadata), encoding="utf-8")
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(f"Extracted {metadata['abstract_word_count']} abstract words and {metadata['keyword_count']} keywords")


if __name__ == "__main__":
    main()
