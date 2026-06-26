from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dacon_submit_api import dacon_submit_api

KST = timezone(timedelta(hours=9))
COMPETITION_ID = "236690"
REPORT_JSON = Path("reports/dacon_daily_submission_slate_20260621.json")
REPORT_MD = Path("reports/dacon_daily_submission_slate_20260621.md")
LOG_PATH = Path("submissions/submission_log.csv")


@dataclass(frozen=True)
class SlateItem:
    slot: int
    submission_id: int
    filename: str
    memo: str


RECONCILED_ID341 = SlateItem(
    slot=1,
    submission_id=341,
    filename="submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv",
    memo="daily slate 1/5 id341 q2 micro controlled probe",
)

SUBMIT_ITEMS = [
    SlateItem(
        slot=2,
        submission_id=340,
        filename="submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv",
        memo="daily slate 2/5 id340 q2 controlled probe",
    ),
    SlateItem(
        slot=3,
        submission_id=337,
        filename="submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv",
        memo="daily slate 3/5 id337 s1 controlled probe",
    ),
    SlateItem(
        slot=4,
        submission_id=221,
        filename="submission_id_221_id218_public_feature_similarity_overlay_cv0.532990_unsubmitted.csv",
        memo="daily slate 4/5 id221 upstream clean-root probe",
    ),
    SlateItem(
        slot=5,
        submission_id=255,
        filename="submission_id_255_id178_anchor_fine_cap012_frontier_bridge_cv0.469816_unsubmitted.csv",
        memo="daily slate 5/5 id255 ambitious anchor bridge probe",
    ),
]


def now_kst() -> str:
    return datetime.now(tz=KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00")


def load_log() -> pd.DataFrame:
    return pd.read_csv(LOG_PATH, dtype=str).fillna("")


def save_log(df: pd.DataFrame) -> None:
    df.to_csv(LOG_PATH, index=False)


def append_note(old: str, note: str) -> str:
    if not old:
        return note
    if note in old:
        return old
    return f"{old} | {note}"


def mark_submitted(df: pd.DataFrame, submission_id: int, note: str) -> None:
    mask = df["id"].eq(str(submission_id))
    if not mask.any():
        raise RuntimeError(f"submission_log.csv missing id {submission_id}")
    df.loc[mask, "is_submitted"] = "True"
    df.loc[mask, "is_submitted_bool"] = "True"
    df.loc[mask, "status_clean"] = "submitted_unscored"
    df.loc[mask, "notes"] = df.loc[mask, "notes"].map(lambda old: append_note(old, note))


def csv_path_for(item: SlateItem) -> Path:
    path = Path("submissions") / item.filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def validate_csv(item: SlateItem) -> None:
    frame = pd.read_csv(csv_path_for(item))
    expected_cols = ["subject_id", "sleep_date", "lifelog_date", "Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    if list(frame.columns) != expected_cols:
        raise RuntimeError(f"{item.filename} header mismatch: {list(frame.columns)}")
    if frame.shape != (250, 10):
        raise RuntimeError(f"{item.filename} shape mismatch: {frame.shape}")
    na_count = int(frame.isna().sum().sum())
    if na_count:
        raise RuntimeError(f"{item.filename} has {na_count} NA values")


def validate_notebook(item: SlateItem) -> None:
    path = Path("notebooks/history") / item.filename.replace("submission_id_", "submission_").replace(".csv", ".ipynb")
    if not path.exists():
        raise FileNotFoundError(path)


def write_report(report: dict[str, Any]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report["updated_at_kst"] = now_kst()
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows = []
    for item in report["final_statuses"]:
        rows.append(
            "| {slot} | {id} | {status} | {file} | {detail} |".format(
                slot=item["slot"],
                id=item["id"],
                status=item["status"],
                file=item["file"],
                detail=str(item.get("detail", "")).replace("|", "/"),
            )
        )
    completed = sum(1 for row in report["final_statuses"] if row["status"] in {"submitted", "reconciled_submitted"})
    md = [
        "# DACON daily submission slate - 2026-06-21",
        "",
        f"Updated: {report['updated_at_kst']}",
        "",
        "| slot | id | status | file | detail |",
        "|---:|---:|---|---|---|",
        *rows,
        "",
        f"Completed final statuses: {completed}/5",
        "",
        "Public scores are intentionally left pending until confirmed from DACON leaderboard output.",
        "",
    ]
    REPORT_MD.write_text("\n".join(md), encoding="utf-8")


def submit_one(item: SlateItem, token: str, team_name: str) -> dict[str, Any]:
    path = csv_path_for(item)
    with path.open("rb") as file_obj:
        data = {
            "cpt_id": COMPETITION_ID,
            "team_name": team_name,
            "file_name": path.name,
            "memo": item.memo,
            "api_token": token,
            "api_version": 2.0,
        }
        response = requests.post(
            dacon_submit_api.__API_URL,
            data=data,
            files={"file": file_obj},
            timeout=(10, 60),
        )
    payload = response.json()
    return {"ok": payload.get("message") == "ok", "status_code": response.status_code, "payload": payload}


def main() -> int:
    token = os.environ.get("DACON_TOKEN")
    team_name = os.environ.get("DACON_TEAMNAME")
    if not token or not team_name:
        raise RuntimeError("DACON_TOKEN and DACON_TEAMNAME must be set")

    for item in [RECONCILED_ID341, *SUBMIT_ITEMS]:
        validate_csv(item)
        validate_notebook(item)

    stamp = datetime.now(tz=KST).strftime("%Y%m%d_%H%M%S")
    backup = LOG_PATH.with_name(f"{LOG_PATH.name}.before_20260621_daily_slate_{stamp}")
    shutil.copyfile(LOG_PATH, backup)

    df = load_log()
    report: dict[str, Any] = {
        "created_at_kst": now_kst(),
        "competition_id": COMPETITION_ID,
        "team_name": team_name,
        "resume_context": "ID341 had DACON HTTP 200 on 2026-06-19 before local pandas dtype failure; it is not resubmitted.",
        "log_backup": str(backup),
        "final_statuses": [],
        "api_attempts": [],
    }

    mark_submitted(
        df,
        RECONCILED_ID341.submission_id,
        "2026-06-21 reconciled as submitted from 2026-06-19 daily slate: DACON HTTP 200 observed before local pandas dtype failure; public score pending.",
    )
    report["final_statuses"].append(
        {
            "slot": RECONCILED_ID341.slot,
            "id": RECONCILED_ID341.submission_id,
            "file": RECONCILED_ID341.filename,
            "status": "reconciled_submitted",
            "detail": "not resubmitted; prior HTTP 200 protected from duplicate",
        }
    )
    save_log(df)
    write_report(report)

    for item in SUBMIT_ITEMS:
        attempt = {
            "slot": item.slot,
            "id": item.submission_id,
            "file": item.filename,
            "memo": item.memo,
            "submitted_at_kst": now_kst(),
        }
        try:
            result = submit_one(item, token, team_name)
            attempt["result"] = result
            report["api_attempts"].append(attempt)
            if not result["ok"]:
                report["final_statuses"].append(
                    {
                        "slot": item.slot,
                        "id": item.submission_id,
                        "file": item.filename,
                        "status": "failed",
                        "detail": result,
                    }
                )
                write_report(report)
                save_log(df)
                return 2
            mark_submitted(
                df,
                item.submission_id,
                f"2026-06-21 submitted via DACON daily slate slot {item.slot}/5; API message ok; public score pending.",
            )
            report["final_statuses"].append(
                {
                    "slot": item.slot,
                    "id": item.submission_id,
                    "file": item.filename,
                    "status": "submitted",
                    "detail": result,
                }
            )
            save_log(df)
            write_report(report)
        except Exception as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            report["api_attempts"].append(attempt)
            report["final_statuses"].append(
                {
                    "slot": item.slot,
                    "id": item.submission_id,
                    "file": item.filename,
                    "status": "error",
                    "detail": attempt["error"],
                }
            )
            save_log(df)
            write_report(report)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
