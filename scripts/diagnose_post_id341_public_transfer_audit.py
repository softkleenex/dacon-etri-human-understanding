#!/usr/bin/env python3
"""Post-ID341 public transfer audit.

Report-only diagnostic. It uses known DACON public anchors and existing base-test
prediction artifacts to prevent local-CV-only candidates from being misread as
Public 0.58/0.55 candidates. It creates no submission CSV, no notebook artifact,
no base predictions, no submission-log row, and no DACON submission.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
BASE_DIR = ROOT / "artifacts" / "base_predictions"
LOG_PATH = ROOT / "submissions" / "submission_log.csv"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KST = timezone(timedelta(hours=9))
PUBLIC_058_GATE = 0.590
CURRENT_CONTROL_ID = 341
PUBLIC_ANCHOR_IDS = [178, 329]
COLLAPSE_ID = 287
RECENT_WATCH_IDS = [334, 335, 336, 337, 338, 339, 340, 341]
ID341_PROXY_FALLBACK = 0.6038297835


def extract_base_id(path: Path) -> int | None:
    match = re.search(r"base_test_id_(\d+)_", path.name)
    return int(match.group(1)) if match else None


def find_base_test_files() -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in sorted(BASE_DIR.glob("base_test_id_*.csv")):
        candidate_id = extract_base_id(path)
        if candidate_id is not None and candidate_id not in out:
            out[candidate_id] = path
    return out


def load_log() -> pd.DataFrame:
    log = pd.read_csv(LOG_PATH)
    log["id_num"] = pd.to_numeric(log["id"], errors="coerce").astype("Int64")
    for column in ["local_cv_score_num", "public_lb_score_num"]:
        log[column] = pd.to_numeric(log[column], errors="coerce")
    return log


def log_row(log: pd.DataFrame, candidate_id: int) -> dict[str, Any]:
    rows = log[log["id_num"].eq(candidate_id)]
    if rows.empty:
        return {}
    row = rows.iloc[-1].to_dict()
    for key in ["local_cv_score", "public_lb_score", "local_cv_score_num", "public_lb_score_num", "cv_lb_gap"]:
        value = pd.to_numeric(row.get(key), errors="coerce")
        row[key] = None if pd.isna(value) else float(value)
    return row


def clean_real_public_rows(log: pd.DataFrame) -> pd.DataFrame:
    rows = log.copy()
    rows = rows[rows["public_lb_score_num"].notna()]
    rows = rows[rows["public_lb_score_num"].between(0.3, 1.0, inclusive="neither")]
    rows = rows[rows["status_clean"].isin(["scored", "cv_lb_gap_risk"])]
    rows = rows[rows["local_cv_score_num"].notna()]
    rows["gap"] = rows["public_lb_score_num"] - rows["local_cv_score_num"]
    return rows


def quantiles(values: pd.Series) -> dict[str, float | None]:
    if values.empty:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    return {
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


def load_pred(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    missing = [target for target in TARGETS if target not in df.columns]
    if missing:
        raise ValueError(f"{path} missing target columns: {missing}")
    return df[TARGETS].astype(float).to_numpy()


def projection(vec: np.ndarray, direction: np.ndarray) -> float:
    denom = float(np.dot(direction, direction))
    return float(np.dot(vec, direction) / denom) if denom > 0 else 0.0


def family_for(candidate_id: int, submission_file: str, base_file: str, model_description: str) -> str:
    text = f"{submission_file} {base_file} {model_description}".lower()
    if candidate_id in {173, 177, 178, 334, 335, 336, 337, 338, 339, 340, 341}:
        return "sequence_anchor_family"
    if "sequence" in text or "random_conv" in text or "randomconv" in text or "seqsum" in text:
        return "sequence_anchor_family"
    if "post328_combo" in text or "failed_direction" in text or "orthogonal" in text or "antivector" in text:
        return "failed_direction_combo_family"
    if "frontier" in text or "logit_bridge" in text or "qfree" in text or "q240" in text:
        return "frontier_bridge_family"
    if "calibration" in text or "isotonic" in text or "temperature" in text or "fixedtemp" in text:
        return "calibration_family"
    if "demographic" in text or "static" in text:
        return "demographic_static_family"
    if "strict" in text or "raw" in text or "feature" in text:
        return "strict_feature_family"
    return "other_family"


def read_id341_proxy() -> float:
    path = REPORTS_DIR / "id341_q2_seqsum_w00025_public_risk_gate_20260618.json"
    if not path.exists():
        return ID341_PROXY_FALLBACK
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ["candidate_proxy", "proxy", "conservative_public_risk_proxy"]:
        value = data.get(key)
        if isinstance(value, int | float):
            return float(value)
    decision = data.get("decision", {})
    for key in ["candidate_proxy", "proxy", "conservative_public_risk_proxy"]:
        value = decision.get(key)
        if isinstance(value, int | float):
            return float(value)
    return ID341_PROXY_FALLBACK


def risk_proxy(row: dict[str, Any], public_base: float, recent_gap_floor: float, id341_proxy: float) -> float:
    local = row["local_cv"]
    local_gap_floor = float(local + recent_gap_floor) if local is not None else id341_proxy
    movement_penalty = 0.18 * row["mean_delta_vs_178"] + 0.020 * row["max_delta_vs_178"]
    control_penalty = 0.12 * row["mean_delta_vs_341"] + 0.012 * row["max_delta_vs_341"]
    failed_penalty = 0.20 * max(0.0, row["projection_failed"])
    combo_penalty = 0.025 if row["family"] == "failed_direction_combo_family" else 0.0
    geometry_proxy = public_base + movement_penalty + control_penalty + failed_penalty + combo_penalty
    # ID341 is almost an ID178 clone; keep it pinned to its explicit prior proxy rather than letting geometry understate it.
    if int(row["id"]) == CURRENT_CONTROL_ID:
        geometry_proxy = max(geometry_proxy, id341_proxy)
    return float(max(geometry_proxy, local_gap_floor))


def risk_tier(row: dict[str, Any]) -> str:
    candidate_id = int(row["id"])
    if row["public_lb"] is not None:
        if candidate_id == COLLAPSE_ID or row["gap"] is not None and row["gap"] >= 0.08:
            return "known_public_gap_or_collapse"
        return "known_public_anchor"
    if candidate_id == CURRENT_CONTROL_ID:
        return "current_control_no_submit"
    if candidate_id in RECENT_WATCH_IDS:
        return "recent_sequence_probe_unsubmitted"
    if row["family"] == "failed_direction_combo_family":
        return "frozen_same_family_after_id329"
    if row["projection_failed"] > 0.04 or row["max_delta_vs_178"] > 0.16 or row["max_delta_vs_341"] > 0.16:
        return "reject_high_geometry_risk"
    if row["proxy_public_risk"] <= PUBLIC_058_GATE:
        return "would_pass_numeric_gate_review_needed"
    return "diagnostic_only"


def row_for_candidate(
    candidate_id: int,
    path: Path,
    log: pd.DataFrame,
    arr178: np.ndarray,
    arr287: np.ndarray,
    arr329: np.ndarray,
    arr341: np.ndarray,
    public_base: float,
    recent_gap_floor: float,
    id341_proxy: float,
) -> dict[str, Any]:
    info = log_row(log, candidate_id)
    arr = load_pred(path)
    failed_direction = (arr287 - arr178).reshape(-1)
    neutral_direction = (arr329 - arr178).reshape(-1)
    control_direction = (arr341 - arr178).reshape(-1)
    vec178 = (arr - arr178).reshape(-1)
    public_lb = info.get("public_lb_score_num")
    gap = info.get("cv_lb_gap")
    row = {
        "id": int(candidate_id),
        "submission_file": info.get("submission_file", ""),
        "base_test_file": str(path.relative_to(ROOT)),
        "model_description": info.get("model_description", ""),
        "local_cv": info.get("local_cv_score_num"),
        "public_lb": public_lb,
        "gap": gap,
        "status_clean": info.get("status_clean", ""),
        "artifact_exists": bool(info.get("artifact_exists", False)),
        "family": family_for(
            candidate_id,
            str(info.get("submission_file", "")),
            path.name,
            str(info.get("model_description", "")),
        ),
        "mean_delta_vs_178": float(np.mean(np.abs(arr - arr178))),
        "max_delta_vs_178": float(np.max(np.abs(arr - arr178))),
        "rmse_vs_178": float(np.sqrt(np.mean((arr - arr178) ** 2))),
        "mean_delta_vs_329": float(np.mean(np.abs(arr - arr329))),
        "max_delta_vs_329": float(np.max(np.abs(arr - arr329))),
        "mean_delta_vs_341": float(np.mean(np.abs(arr - arr341))),
        "max_delta_vs_341": float(np.max(np.abs(arr - arr341))),
        "projection_failed": projection(vec178, failed_direction),
        "projection_329": projection(vec178, neutral_direction),
        "projection_341": projection(vec178, control_direction),
    }
    row["proxy_public_risk"] = risk_proxy(row, public_base, recent_gap_floor, id341_proxy)
    row["passes_public_058_gate"] = bool(row["proxy_public_risk"] <= PUBLIC_058_GATE)
    row["risk_tier"] = risk_tier(row)
    return row


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Post-ID341 public transfer audit",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, submission CSV, notebook/history artifact, submission-log row, or DACON submission.",
        "",
        "## Public evidence read",
        "",
        f"- Plausible public rows used: `{report['public_rows_used']}`",
        f"- Recent anchor gap floor, IDs 178/329: `{report['recent_anchor_gap_floor']:.6f}`",
        f"- Local/public correlation on plausible rows: `{report['local_public_corr']:.6f}`",
        f"- Current no-submit control: ID{CURRENT_CONTROL_ID}, proxy `{report['id341_proxy']:.6f}`",
        "",
        "Plausible public evidence still says local CV is a weak selector. ID178 and ID329 both landed near Public 0.601 despite different local CVs; ID287 proves full-OOF calibration can collapse.",
        "",
        "## Recent/control candidates",
        "",
        "| id | local | public | proxy | Δ178 mean/max | Δ341 mean/max | proj failed | tier | file |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["recent_control_rows"]:
        public = "" if row["public_lb"] is None else f"{row['public_lb']:.6f}"
        local = "" if row["local_cv"] is None else f"{row['local_cv']:.6f}"
        lines.append(
            f"| {row['id']} | {local} | {public} | {row['proxy_public_risk']:.6f} | "
            f"{row['mean_delta_vs_178']:.6f}/{row['max_delta_vs_178']:.6f} | "
            f"{row['mean_delta_vs_341']:.6f}/{row['max_delta_vs_341']:.6f} | "
            f"{row['projection_failed']:+.4f} | `{row['risk_tier']}` | `{row['submission_file']}` |"
        )
    lines.extend(
        [
            "",
            "## Best numeric proxy rows after quarantine",
            "",
            "| rank | id | local | proxy | family | tier | Δ178 max | Δ341 max | file |",
            "|---:|---:|---:|---:|---|---|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(report["top_numeric_after_quarantine"], start=1):
        local = "" if row["local_cv"] is None else f"{row['local_cv']:.6f}"
        lines.append(
            f"| {rank} | {row['id']} | {local} | {row['proxy_public_risk']:.6f} | `{row['family']}` | "
            f"`{row['risk_tier']}` | {row['max_delta_vs_178']:.6f} | {row['max_delta_vs_341']:.6f} | `{row['submission_file']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Any candidate passes Public 0.58 transfer gate: `{report['decision']['any_candidate_passes_public_058_gate']}`",
            f"- Submit now: `{report['decision']['submit_now']}`",
            f"- Create submission artifact now: `{report['decision']['create_submission_artifact']}`",
            f"- Push Kaggle now: `{report['decision']['push_kaggle_now']}`",
            f"- Next action: {report['decision']['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def build_report() -> dict[str, Any]:
    log = load_log()
    public_rows = clean_real_public_rows(log)
    recent_rows = public_rows[public_rows["id_num"].astype(int).isin(PUBLIC_ANCHOR_IDS)]
    if recent_rows.empty:
        raise ValueError("Missing recent public anchors 178/329 in submission log.")
    public_base = float(recent_rows["public_lb_score_num"].min())
    recent_gap_floor = float(recent_rows["gap"].min())
    local_public_corr = float(public_rows[["local_cv_score_num", "public_lb_score_num"]].corr().iloc[0, 1])
    files = find_base_test_files()
    required = [178, 287, 329, 341]
    missing = [candidate_id for candidate_id in required if candidate_id not in files]
    if missing:
        raise FileNotFoundError(f"Missing required base_test ids: {missing}")
    arr178 = load_pred(files[178])
    arr287 = load_pred(files[287])
    arr329 = load_pred(files[329])
    arr341 = load_pred(files[341])
    id341_proxy = read_id341_proxy()
    rows = [
        row_for_candidate(
            candidate_id,
            path,
            log,
            arr178,
            arr287,
            arr329,
            arr341,
            public_base,
            recent_gap_floor,
            id341_proxy,
        )
        for candidate_id, path in sorted(files.items())
    ]
    quarantine_tiers = {
        "known_public_gap_or_collapse",
        "frozen_same_family_after_id329",
        "reject_high_geometry_risk",
    }
    numeric_after_quarantine = [
        row
        for row in rows
        if row["risk_tier"] not in quarantine_tiers and row["public_lb"] is None and row["id"] != CURRENT_CONTROL_ID
    ]
    top_numeric = sorted(numeric_after_quarantine, key=lambda row: (row["proxy_public_risk"], row["max_delta_vs_341"]))[
        :20
    ]
    recent_control_rows = [row for row in rows if row["id"] in [178, 287, 329, *RECENT_WATCH_IDS]]
    any_pass = any(row["passes_public_058_gate"] for row in top_numeric)
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_public_transfer_audit",
        "inputs": {
            "submission_log": str(LOG_PATH.relative_to(ROOT)),
            "base_predictions_dir": str(BASE_DIR.relative_to(ROOT)),
        },
        "public_rows_used": int(len(public_rows)),
        "public_gap_summary": {
            "all_plausible": quantiles(public_rows["gap"]),
            "recent_id178_id329": quantiles(recent_rows["gap"]),
        },
        "local_public_corr": local_public_corr,
        "public_anchor_ids": PUBLIC_ANCHOR_IDS,
        "collapse_id": COLLAPSE_ID,
        "current_control_id": CURRENT_CONTROL_ID,
        "id341_proxy": id341_proxy,
        "recent_anchor_gap_floor": recent_gap_floor,
        "public_058_gate": PUBLIC_058_GATE,
        "tier_counts": dict(Counter(row["risk_tier"] for row in rows)),
        "family_counts": dict(Counter(row["family"] for row in rows)),
        "recent_control_rows": sorted(recent_control_rows, key=lambda row: row["id"]),
        "top_numeric_after_quarantine": top_numeric,
        "top_all_by_proxy": sorted(rows, key=lambda row: (row["proxy_public_risk"], row["max_delta_vs_341"]))[:80],
        "decision": {
            "any_candidate_passes_public_058_gate": any_pass,
            "submit_now": False,
            "create_submission_artifact": False,
            "push_kaggle_now": False,
            "public_058_plausible_candidate_exists": False,
            "forced_probe_fallback": None,
            "manual_probe_family_status": "consumed_after_id341_id340_id337_public_micro_probe_slate",
            "next_action": (
                "Do not continue local OOF recombination or same-family shrinking. The next score-relevant work is the "
                "report-only Kaggle upstream source rebuild, or a genuinely new source family evaluated through this transfer gate."
            ),
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_public_transfer_audit_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
