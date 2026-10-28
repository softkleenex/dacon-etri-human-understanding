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
DEFAULT_RECEIPT = PAPER_DIR / "edas_receipt.json"
KST = timezone(timedelta(hours=9))
EDAS_TOPIC = "Artificial Intelligence and Machine Learning"


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


def load_edas_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "observed_at_kst",
        "paper_id",
        "registered_title",
        "review_manuscript_uploaded",
        "source",
        "status",
        "submitted_pdf",
        "submitted_pdf_sha256",
    }
    missing = sorted(required - set(receipt))
    if missing:
        raise ValueError(f"EDAS receipt is missing fields: {missing}")
    return receipt


def build_metadata(tex_path: Path, receipt_path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    text = tex_path.read_text(encoding="utf-8")
    receipt = load_edas_receipt(receipt_path)
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
        "registered_title_matches_source": title == receipt["registered_title"],
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
        "edas_paper_id": receipt["paper_id"],
        "edas_status": receipt["status"],
        "review_manuscript_uploaded": receipt["review_manuscript_uploaded"],
        "edas_receipt": rel(receipt_path),
        "edas_status_observed_at_kst": receipt["observed_at_kst"],
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
        f"- Registered title matches source: `{metadata['registered_title_matches_source']}`",
        f"- EDAS status observed: `{metadata['edas_status_observed_at_kst']}`",
        f"- EDAS receipt: `{metadata['edas_receipt']}`",
        "",
        "## Keywords 1-6",
        "",
    ]
    lines.extend(f"- Keyword {index}: `{keyword}`" for index, keyword in enumerate(metadata["keywords"], start=1))
    lines.extend(
        [
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
        ]
    )
    if metadata["human_fillins_remaining"]:
        lines.extend(f"- {item}" for item in metadata["human_fillins_remaining"])
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tex_path = args.tex if args.tex.is_absolute() else ROOT / args.tex
    output = args.output if args.output.is_absolute() else ROOT / args.output
    receipt_path = args.receipt if args.receipt.is_absolute() else ROOT / args.receipt
    metadata = build_metadata(tex_path, receipt_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(metadata), encoding="utf-8")
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(f"Extracted {metadata['abstract_word_count']} abstract words and {metadata['keyword_count']} keywords")


if __name__ == "__main__":
    main()
