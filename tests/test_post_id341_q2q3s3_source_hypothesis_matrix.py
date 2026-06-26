import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q2q3s3_source_hypothesis_matrix.py"


def load_module():
    spec = importlib.util.spec_from_file_location("q2q3s3_source_hypothesis_matrix", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_q2q3s3_matrix_blocks_artifacts_and_remote_actions():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["q2q3s3_source_hypothesis_matrix_ready"] is True
    assert decision["local_q2q3s3_artifact_promotable"] is False
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False
    assert decision["requires_explicit_kaggle_approval"] is True
    assert decision["approval_phrase"] == "Kaggle upstream rebuild no-submit 실행 승인"


def test_q2q3s3_matrix_focus_targets_are_missing_but_have_id226_signal():
    module = load_module()
    report = module.build_report()

    assert report["constants"]["focus_targets"] == ["Q2", "Q3", "S3"]
    assert report["decision"]["all_focus_targets_missing_locally"] is True
    assert report["decision"]["all_focus_targets_have_historical_id226_signal"] is True
    assert report["decision"]["all_focus_targets_id226_win_all_folds_vs_341"] is True

    for _target, row in report["target_source_matrix"].items():
        assert row["local_inventory"]["local_status"] == "missing"
        assert row["clean_backbone_id226"]["target_delta_vs_178"] < 0
        assert row["clean_backbone_id226"]["fold_wins_vs_341"] == row["clean_backbone_id226"]["fold_count"]
        assert row["processed_feature_coverage"]["match_count"] > 0


def test_q2q3s3_matrix_uses_target_specific_source_families():
    module = load_module()
    report = module.build_report()
    matrix = report["target_source_matrix"]

    assert "mUsageStats" in matrix["Q2"]["raw_sources"]
    assert "mUsageStats" in matrix["Q3"]["raw_sources"]
    assert "mWifi" in matrix["Q3"]["raw_sources"]
    assert "mScreenStatus" in matrix["S3"]["raw_sources"]
    assert "wLight" in matrix["S3"]["raw_sources"]
    assert "direct row concatenation is blocked" in matrix["Q2"]["external_value"]
    assert "not as direct DACON rows" in matrix["Q3"]["external_value"]


def test_q2q3s3_markdown_mentions_no_id342_and_exact_approval(tmp_path):
    module = load_module()
    report = module.build_report()
    output = tmp_path / "matrix.md"

    output.write_text(module.build_markdown(json.loads(json.dumps(report))), encoding="utf-8")
    text = output.read_text(encoding="utf-8")

    assert "Create submission artifact: `False`" in text
    assert "Run Kaggle now: `False`" in text
    assert "Kaggle upstream rebuild no-submit 실행 승인" in text
    assert "Keep local ID342 closed" in text
