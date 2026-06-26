import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_kaggle_remote_fail_closed_dryrun.py"


def load_module():
    spec = importlib.util.spec_from_file_location("remote_fail_closed_dryrun", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_fail_closed_dryrun_blocks_targets_before_kaggle_cli():
    module = load_module()
    report = module.build_report()

    assert report["decision"]["remote_fail_closed_dryrun_pass"] is True
    assert report["decision"]["requires_explicit_user_approval"] is True
    assert report["decision"]["execute_kaggle_now"] is False
    assert report["decision"]["push_kaggle_now"] is False
    assert report["decision"]["run_kaggle_now"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False
    assert report["checks"]["all_failed_before_remote"] is True
    assert report["checks"]["all_guard_messages_present"] is True
    assert report["checks"]["no_forbidden_leak_markers"] is True
    assert {row["target"] for row in report["results"]} == set(module.REMOTE_TARGETS)


def test_remote_fail_closed_markdown_records_targets():
    module = load_module()
    report = module.build_report()
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "Remote fail-closed dry-run pass: `True`" in markdown
    assert "kaggle-push-upstream-patch-dataset" in markdown
    assert "kaggle-wait-and-pull-upstream-rebuild-output" in markdown
