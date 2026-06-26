#!/usr/bin/env python3
"""Report-only validation-transfer gate after remote candidate parity.

This diagnostic uses existing submission logs and the post-parity public-anchor
rerank. It does not create predictions, submission artifacts, or DACON
submissions. The purpose is to prevent local-CV-only candidates from being
misread as Public 0.58 candidates after ID178/ID287/ID329 showed large transfer
gaps.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
LOG_PATH = ROOT / "submissions" / "submission_log.csv"
RERANK_PATH = ROOT / "reports" / "post_remote_parity_public_anchor_rerank_post_id339_20260618.json"
PUBLIC_058_GATE = 0.590
RECENT_ANCHOR_IDS = {178, 329}
COLLAPSE_IDS = {287}
WATCH_IDS = [337, 339, 338, 336, 179, 307, 295, 304, 297]


def clean_real_public_rows(log: pd.DataFrame) -> pd.DataFrame:
    rows = log.copy()
    rows["id_num"] = pd.to_numeric(rows["id"], errors="coerce")
    rows["local_cv"] = pd.to_numeric(rows["local_cv_score_num"], errors="coerce")
    rows["public_lb"] = pd.to_numeric(rows["public_lb_score_num"], errors="coerce")
    rows = rows[rows["public_lb"].notna() & rows["public_lb"].gt(0.1)]
    rows = rows[rows["status_clean"].isin(["scored", "cv_lb_gap_risk"])]
    rows["gap"] = rows["public_lb"] - rows["local_cv"]
    return rows


def quantiles(values: pd.Series) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


def row_by_id(rows: list[dict[str, Any]], candidate_id: int) -> dict[str, Any] | None:
    for row in rows:
        if int(row["id"]) == candidate_id:
            return row
    return None


def candidate_rows(rerank: dict[str, Any], recent_gap_floor: float) -> list[dict[str, Any]]:
    all_rows = rerank["top_all_by_proxy"]
    out = []
    for candidate_id in WATCH_IDS:
        row = row_by_id(all_rows, candidate_id)
        if row is None:
            continue
        local_cv = float(row["local_cv"])
        proxy = float(row["conservative_public_risk_proxy"])
        local_plus_recent_gap = local_cv + recent_gap_floor
        stress_estimate = max(proxy, local_plus_recent_gap)
        family = row["family"]
        quarantined = family == "failed_direction_combo_family" or candidate_id in {
            295,
            296,
            297,
            304,
            305,
            306,
            307,
            310,
            311,
        }
        out.append(
            {
                "id": candidate_id,
                "local_cv": local_cv,
                "proxy": proxy,
                "local_plus_recent_gap_floor": local_plus_recent_gap,
                "stress_estimate": stress_estimate,
                "family": family,
                "max_delta_vs_178": float(row["max_delta_vs_178"]),
                "projection_failed": float(row["projection_failed"]),
                "quarantined": quarantined,
                "passes_public_058_transfer_gate": stress_estimate <= PUBLIC_058_GATE and not quarantined,
                "submission_file": row["submission_file"],
            }
        )
    return sorted(out, key=lambda item: (item["passes_public_058_transfer_gate"] is False, item["stress_estimate"]))


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Validation-transfer gate — post remote parity",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, submission CSV, notebook artifact, submission-log row, or DACON submission.",
        "",
        "## Public gap read",
        "",
        f"- Real public rows used: `{report['public_rows_used']}`",
        f"- Safe scored gap median: `{report['gap_summary']['safe_scored']['median']:.6f}`",
        f"- Recent ID178/ID329 gap floor: `{report['recent_anchor_gap_floor']:.6f}`",
        f"- Public `0.58` plausibility gate: stress estimate `<= {PUBLIC_058_GATE:.3f}` and not quarantined",
        "",
        "Recent anchor behavior matters more than old safe rows: ID178 and ID329 both landed around Public `0.601` despite very different local CVs. That is why the gate uses a stress estimate rather than local CV alone.",
        "",
        "## Candidate gate",
        "",
        "| id | local | proxy | local+recent-gap | stress | pass 0.58 gate | family | quarantined | file |",
        "|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in report["candidate_gate"]:
        lines.append(
            f"| {row['id']} | {row['local_cv']:.6f} | {row['proxy']:.6f} | "
            f"{row['local_plus_recent_gap_floor']:.6f} | {row['stress_estimate']:.6f} | "
            f"`{row['passes_public_058_transfer_gate']}` | `{row['family']}` | "
            f"`{row['quarantined']}` | `{row['submission_file']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Any candidate passes Public `0.58` transfer gate: `{report['decision']['any_candidate_passes_public_058_gate']}`",
            f"- Submit now: `{report['decision']['submit_now']}`",
            f"- Next action: {report['decision']['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def build_report() -> dict[str, Any]:
    log = pd.read_csv(LOG_PATH)
    public_rows = clean_real_public_rows(log)
    safe_rows = public_rows[public_rows["status_clean"].eq("scored")]
    risk_rows = public_rows[public_rows["status_clean"].eq("cv_lb_gap_risk")]
    recent_rows = public_rows[public_rows["id_num"].isin(RECENT_ANCHOR_IDS)]
    collapse_rows = public_rows[public_rows["id_num"].isin(COLLAPSE_IDS)]
    if recent_rows.empty:
        raise ValueError("No recent anchor public rows available.")
    recent_gap_floor = float(recent_rows["gap"].min())

    rerank = json.loads(RERANK_PATH.read_text(encoding="utf-8"))
    candidates = candidate_rows(rerank, recent_gap_floor)
    any_pass = any(row["passes_public_058_transfer_gate"] for row in candidates)
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "validation_transfer_gate_post_remote_parity",
        "inputs": {
            "submission_log": str(LOG_PATH.relative_to(ROOT)),
            "post_parity_rerank": str(RERANK_PATH.relative_to(ROOT)),
        },
        "public_rows_used": int(len(public_rows)),
        "gap_summary": {
            "all_real_public": quantiles(public_rows["gap"]),
            "safe_scored": quantiles(safe_rows["gap"]),
            "cv_lb_gap_risk": quantiles(risk_rows["gap"]),
            "recent_id178_id329": quantiles(recent_rows["gap"]),
            "collapse_id287": quantiles(collapse_rows["gap"]) if not collapse_rows.empty else None,
        },
        "recent_anchor_gap_floor": recent_gap_floor,
        "public_058_gate": PUBLIC_058_GATE,
        "candidate_gate": candidates,
        "decision": {
            "any_candidate_passes_public_058_gate": any_pass,
            "submit_now": False,
            "no_submission_created": True,
            "no_dacon_submission_made": True,
            "next_action": (
                "Do not create ID340 from local CV alone. The next useful experiment must define a new source or "
                "validation split that lowers the transfer stress estimate, not merely blend toward lower local CV."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "validation_transfer_gate_post_remote_parity_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
