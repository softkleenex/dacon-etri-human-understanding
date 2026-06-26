import json
import os
import shutil
import subprocess
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


def run(cmd: str, cwd: Path | None = None) -> None:
    print(f"\n$ {cmd}", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("NUMEXPR_NUM_THREADS", "2")
    if cwd is not None:
        env["PYTHONPATH"] = str(cwd)
    subprocess.run(cmd, shell=True, check=True, cwd=str(cwd) if cwd else None, env=env)


is_kaggle = Path("/kaggle/input").exists()
input_root = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input" if is_kaggle else "."))
working_root = Path(
    os.environ.get(
        "KAGGLE_WORKING_DIR",
        "/kaggle/working" if is_kaggle else "build/kaggle_missingness_subject_state_smoke",
    )
)
scratch_root = Path(
    os.environ.get(
        "KAGGLE_SCRATCH_DIR",
        "/tmp/dacon_etri_missingness_subject_state" if is_kaggle else str(working_root / "scratch"),
    )
)
project_dir = scratch_root / "project"
output_dir = working_root / "kaggle_outputs"
output_dir.mkdir(parents=True, exist_ok=True)
os.environ["KAGGLE_OUTPUT_DIR"] = str(output_dir)

if project_dir.exists():
    shutil.rmtree(project_dir)
project_dir.mkdir(parents=True, exist_ok=True)

copy_names = [
    "src",
    "conf",
    "scripts",
    "data",
    "artifacts",
    "submissions",
    "reports",
    "pyproject.toml",
    "README.md",
    "dataset-metadata.json",
    "Makefile",
]
patch_script_names = [
    "run_post_id342_missingness_subject_state_heavy_no_submit.py",
    "diagnose_post_id341_q2q3s3_fold_internal_model_smoke.py",
    "diagnose_post_id341_q2q3s3_feature_generation_metadata_smoke.py",
]
runner_script_name = patch_script_names[0]


def overlay_project(source_root: Path, dst_root: Path) -> None:
    source_zip = source_root / "source.zip"
    if source_zip.exists():
        tmp = scratch_root / f"unzipped_{source_root.name}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_zip) as zf:
            zf.extractall(tmp)
        source_root = tmp
    for name in copy_names:
        src = source_root / name
        if not src.exists():
            continue
        dst = dst_root / name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for script_name in patch_script_names:
        standalone_script = source_root / script_name
        if standalone_script.exists():
            dst = dst_root / "scripts" / script_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(standalone_script, dst)


def add_candidate_root(candidate_roots: set[Path], path: Path, ancestor_index: int = 0) -> None:
    if len(path.parents) <= ancestor_index:
        return
    candidate_roots.add(path.parents[ancestor_index])


candidate_roots: set[Path] = set()
if input_root.exists():
    for source_zip in input_root.rglob("source.zip"):
        candidate_roots.add(source_zip.parent)
    for script_name in patch_script_names:
        for marker in input_root.rglob(script_name):
            if marker.parent.name == "scripts":
                add_candidate_root(candidate_roots, marker, 1)
            else:
                candidate_roots.add(marker.parent)
    for marker in input_root.rglob("pyproject.toml"):
        candidate_roots.add(marker.parent)
    for raw_file in input_root.rglob("data/raw/ch2025_data_items/ch2025_mGps.parquet"):
        add_candidate_root(candidate_roots, raw_file, 3)


def root_priority(path: Path) -> tuple[int, str]:
    text = str(path)
    if "dacon-etri-human-base" in text:
        return (0, text)
    if "dacon-etri-upstream-patch-post-id341" in text:
        return (1, text)
    if "dacon-etri-ss-subject-state-patch-id342" in text:
        return (2, text)
    return (1, text)


roots = sorted(candidate_roots, key=root_priority)
if not roots:
    roots = [Path.cwd()]
for root in roots:
    print(f"Overlaying input root: {root}")
    overlay_project(root, project_dir)


def copy_first_match(pattern: str, dst: Path) -> None:
    if dst.exists() or not input_root.exists():
        return
    matches = sorted(input_root.rglob(pattern))
    if not matches:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matches[0], dst)
    print(f"Backfilled {dst.relative_to(project_dir)} from {matches[0]}")


def copy_base_predictions() -> None:
    dst_dir = project_dir / "artifacts/base_predictions"
    if any(dst_dir.glob("base_oof_id_342_*.csv")):
        return
    matches = sorted(input_root.rglob("base_oof_id_342_*.csv")) if input_root.exists() else []
    if not matches:
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in matches[0].parent.glob("base_oof_id_*.csv"):
        shutil.copy2(path, dst_dir / path.name)
    print(f"Backfilled base_predictions from {matches[0].parent}")


copy_first_match("ch2026_metrics_train.csv", project_dir / "data/raw/ch2026_metrics_train.csv")
copy_first_match("ch2026_submission_sample.csv", project_dir / "data/raw/ch2026_submission_sample.csv")
copy_first_match(
    "id340_validation_split_manifest_20260618.json",
    project_dir / "reports/id340_validation_split_manifest_20260618.json",
)
copy_base_predictions()

required = [
    project_dir / "scripts" / runner_script_name,
    project_dir / "scripts/diagnose_post_id341_q2q3s3_fold_internal_model_smoke.py",
    project_dir / "scripts/diagnose_post_id341_q2q3s3_feature_generation_metadata_smoke.py",
    project_dir / "data/raw/ch2026_metrics_train.csv",
    project_dir / "data/raw/ch2026_submission_sample.csv",
    project_dir / "data/raw/ch2025_data_items/ch2025_mGps.parquet",
    project_dir / "data/raw/ch2025_data_items/ch2025_wHr.parquet",
    project_dir / "artifacts/base_predictions",
    project_dir / "reports/id340_validation_split_manifest_20260618.json",
]
missing = [str(path.relative_to(project_dir)) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(f"Packaged project is missing required paths: {missing}")

max_rows = int(os.environ.get("MAX_ROWS_PER_MODALITY", "0"))
batch_size = int(os.environ.get("BATCH_SIZE", "65536"))
embedding_dim = int(os.environ.get("EMBEDDING_DIM", "16"))
epochs = int(os.environ.get("EPOCHS", "5"))
seed = int(os.environ.get("SEED", "45502"))
cmd = " ".join(
    [
        f"python scripts/{runner_script_name}",
        "--mode remote_no_submit",
        f"--max-rows-per-modality {max_rows}",
        f"--batch-size {batch_size}",
        f"--embedding-dim {embedding_dim}",
        f"--epochs {epochs}",
        f"--seed {seed}",
        f"--output-dir {output_dir}",
    ]
)
run(cmd, cwd=project_dir)

if is_kaggle and project_dir.exists():
    shutil.rmtree(project_dir)

suspicious = []
for path in output_dir.rglob("*"):
    if not path.is_file():
        continue
    lower = path.name.lower()
    if path.suffix.lower() == ".csv" or "submission" in lower or "base_oof" in lower or "base_test" in lower:
        suspicious.append(str(path))
if suspicious:
    raise RuntimeError(f"Forbidden submission-like output detected: {suspicious}")

manifest = {
    "created_at_kst": datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
    "runner": "post_id342_missingness_subject_state_heavy_no_submit_wrapper",
    "mode": "remote_no_submit",
    "max_rows_per_modality": max_rows,
    "batch_size": batch_size,
    "embedding_dim": embedding_dim,
    "epochs": epochs,
    "seed": seed,
    "output_files": [str(path.relative_to(output_dir)) for path in sorted(output_dir.rglob("*")) if path.is_file()],
    "no_submission_created": True,
    "no_submission_artifact_created": True,
    "no_submission_log_row_created": True,
    "no_dacon_submission_made": True,
}
(output_dir / "post_id342_missingness_subject_state_wrapper_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, indent=2, ensure_ascii=False))
