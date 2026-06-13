#!/usr/bin/env python3
"""Convert repository PDFs to Markdown text files and optionally remove PDFs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def convert_pdf(pdf_path: Path) -> Path:
    md_path = pdf_path.with_suffix(".md")
    tmp_txt = pdf_path.with_suffix(".txt")
    subprocess.check_call(["pdftotext", "-layout", str(pdf_path), str(tmp_txt)])
    text = tmp_txt.read_text(errors="ignore").strip()
    tmp_txt.unlink(missing_ok=True)
    title = pdf_path.stem
    md_path.write_text(f"# {title}\n\n{text}\n")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete-pdf", action="store_true")
    parser.add_argument("paths", nargs="*", default=["docs"])
    args = parser.parse_args()

    converted = []
    reused = []
    deleted = []
    for root in args.paths:
        for pdf_path in sorted((ROOT / root).rglob("*.pdf")):
            md_path = pdf_path.with_suffix(".md")
            if md_path.exists() and md_path.stat().st_size > 0:
                reused.append((pdf_path, md_path))
            else:
                md_path = convert_pdf(pdf_path)
                converted.append((pdf_path, md_path))
            if args.delete_pdf and md_path.exists() and md_path.stat().st_size > 0:
                pdf_path.unlink()
                deleted.append(pdf_path)

    report = ["# PDF Conversion Report\n"]
    report.append("## Converted\n")
    report.extend(f"- `{pdf.relative_to(ROOT)}` -> `{md.relative_to(ROOT)}`\n" for pdf, md in converted)
    report.append("\n## Existing Markdown Reused\n")
    report.extend(f"- `{pdf.relative_to(ROOT)}` -> `{md.relative_to(ROOT)}`\n" for pdf, md in reused)
    report.append("\n## Deleted PDFs\n")
    report.extend(f"- `{pdf.relative_to(ROOT)}`\n" for pdf in deleted)
    out = ROOT / "reports" / "pdf_conversion_report.md"
    out.write_text("\n".join(report))
    print(f"converted={len(converted)} reused={len(reused)} deleted={len(deleted)}")
    print(f"Wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
