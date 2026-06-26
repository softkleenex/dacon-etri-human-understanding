from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_prelaunch_freshness_contract.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prelaunch_freshness_contract", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_ordered_makefile(module, path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "kaggle-upstream-prelaunch-local:",
                f"\t.venv/bin/python {module.SELF_SUPERVISED_DESIGN_CMD}",
                f"\t.venv/bin/python {module.SLEEP_PROXY_DESIGN_CMD}",
                f"\t.venv/bin/python {module.MATERIAL_QUEUE_CMD}",
                f"\t.venv/bin/python {module.MATERIAL_COVERAGE_CMD}",
                f"\t.venv/bin/python {module.PRELAUNCH_CMD}",
                f"\t.venv/bin/python {module.FRESHNESS_CMD}",
                f"\t.venv/bin/python {module.SELF_SUPERVISED_HEAVY_CMD}",
                f"\t.venv/bin/python {module.SAFE_WAIT_CMD}",
                f"\t.venv/bin/python {module.NEXT_ROUTE_TRIAGE_CMD}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_prelaunch_freshness_contract_passes_and_blocks_actions() -> None:
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["prelaunch_freshness_contract_pass"] is True
    assert decision["prelaunch_ready_for_explicit_approval"] is True
    assert decision["requires_explicit_user_approval"] is True
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert report["counts"]["prelaunch_recorded_input_paths"] >= 20
    assert report["counts"]["stale_inputs_newer_than_prelaunch"] == 0
    assert report["makefile_order_contract"]["order_ok"] is True
    assert report["no_submission_created"] is True
    assert report["no_submission_log_row_created"] is True


def test_prelaunch_freshness_contract_fails_when_input_is_newer(tmp_path: Path) -> None:
    module = load_module()
    input_path = tmp_path / "input.json"
    prelaunch_path = tmp_path / "prelaunch.json"
    makefile_path = tmp_path / "Makefile"
    write_json(input_path, {"decision": {"ok": True}})
    write_json(
        prelaunch_path,
        {
            "decision": {
                "prelaunch_ready_for_explicit_approval": True,
                "requires_explicit_user_approval": True,
                "execute_kaggle_now": False,
                "push_kaggle_now": False,
                "run_kaggle_now": False,
                "create_submission_artifact": False,
                "submit_now": False,
            },
            "inputs": {
                "dummy_input": {
                    "path": str(input_path),
                    "exists": True,
                    "parse_ok": True,
                }
            },
        },
    )
    write_ordered_makefile(module, makefile_path)
    os.utime(prelaunch_path, (1000, 1000))
    os.utime(input_path, (2000, 2000))

    report = module.build_report(prelaunch_path=prelaunch_path, makefile_path=makefile_path)

    assert report["counts"]["stale_inputs_newer_than_prelaunch"] == 1
    assert report["stale_inputs"][0]["input_key"] == "dummy_input"
    assert report["decision"]["prelaunch_freshness_contract_pass"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False


def test_prelaunch_freshness_contract_fails_when_makefile_order_breaks(tmp_path: Path) -> None:
    module = load_module()
    prelaunch_path = module.DEFAULT_PRELAUNCH
    makefile_path = tmp_path / "Makefile"
    makefile_path.write_text(
        "\n".join(
            [
                "kaggle-upstream-prelaunch-local:",
                f"\t.venv/bin/python {module.SELF_SUPERVISED_DESIGN_CMD}",
                f"\t.venv/bin/python {module.SLEEP_PROXY_DESIGN_CMD}",
                f"\t.venv/bin/python {module.MATERIAL_QUEUE_CMD}",
                f"\t.venv/bin/python {module.PRELAUNCH_CMD}",
                f"\t.venv/bin/python {module.MATERIAL_COVERAGE_CMD}",
                f"\t.venv/bin/python {module.FRESHNESS_CMD}",
                f"\t.venv/bin/python {module.SELF_SUPERVISED_HEAVY_CMD}",
                f"\t.venv/bin/python {module.SAFE_WAIT_CMD}",
                f"\t.venv/bin/python {module.NEXT_ROUTE_TRIAGE_CMD}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = module.build_report(prelaunch_path=prelaunch_path, makefile_path=makefile_path)

    assert report["makefile_order_contract"]["order_ok"] is False
    assert report["decision"]["prelaunch_freshness_contract_pass"] is False
    assert report["decision"]["submit_now"] is False


def test_prelaunch_freshness_markdown_is_unambiguous() -> None:
    module = load_module()
    markdown = module.build_markdown(json.loads(json.dumps(module.build_report())))

    assert "Prelaunch freshness contract pass: `True`" in markdown
    assert "Create submission artifact: `False`" in markdown
    assert "Submit now: `False`" in markdown
    assert "Order OK: `True`" in markdown
