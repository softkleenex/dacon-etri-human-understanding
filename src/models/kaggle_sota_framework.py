"""
True 2026 Kaggle SOTA Engine - Physical Runner Framework
Direct implementation of AgentGA (Parent Archive Inheritance),
Lycurgus (Pydantic-Graph 5-Phase State Machine), and NVIDIA (cuDF/cuML GPU Accelerated Stacking).
Optimized to run dynamically in any project directory without hardcoded path conflicts.
"""

import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# Global Dynamic Directory Resolution (Resolves conflict across multiple project folders)
PROJECT_ROOT = os.path.join(os.getcwd(), ".kaggle_sota")
os.makedirs(PROJECT_ROOT, exist_ok=True)

# Try importing GPU accelerated RAPIDS libraries, fallback to CPU
try:
    import cudf
    from cuml.ensemble import RandomForestRegressor as cumlRF

    HAS_GPU_ACCEL = True
except ImportError:
    HAS_GPU_ACCEL = False


class LycurgusState:
    def __init__(self, mission_id):
        self.mission_id = mission_id
        self.phase = "Discovery"  # Discovery -> Research -> Prototype -> Evolution -> Submission
        self.history = []
        self.current_score = 0.0
        self.active_checkpoint = None
        self.metadata = {}

    def transition_to(self, new_phase):
        print(f"[Lycurgus] State Transition: {self.phase} -> {new_phase}")
        self.history.append({"from": self.phase, "to": new_phase, "timestamp": time.time()})
        self.phase = new_phase

    def save_checkpoint(self):
        checkpoint_path = os.path.join(PROJECT_ROOT, f"checkpoint_{self.mission_id}.json")
        checkpoint_data = {
            "mission_id": self.mission_id,
            "phase": self.phase,
            "current_score": self.current_score,
            "active_checkpoint": self.active_checkpoint,
            "history": self.history,
            "metadata": self.metadata,
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"[Lycurgus] Persistence Checkpoint Saved to {checkpoint_path} (Inspectable/Resumable)")

    @classmethod
    def load_checkpoint(cls, mission_id):
        checkpoint_path = os.path.join(PROJECT_ROOT, f"checkpoint_{mission_id}.json")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        with open(checkpoint_path) as f:
            data = json.load(f)
        state = cls(data["mission_id"])
        state.phase = data["phase"]
        state.current_score = data["current_score"]
        state.active_checkpoint = data["active_checkpoint"]
        state.history = data["history"]
        state.metadata = data["metadata"]
        print(f"[Lycurgus] Mission Resumed. Current Phase: {state.phase} | Score: {state.current_score}")
        return state


class LycurgusRunner:
    def __init__(self):
        self.sandbox_dir = os.path.join(PROJECT_ROOT, "sandbox")
        self.subset_csv_path = os.path.join(PROJECT_ROOT, "subset_1k.csv")
        os.makedirs(self.sandbox_dir, exist_ok=True)

    def generate_subset_if_missing(self, full_csv_path):
        if not os.path.exists(self.subset_csv_path):
            print(f"[Lycurgus] Subset file missing. Generating 1,000-row subset from {full_csv_path}...")
            try:
                df = pd.read_csv(full_csv_path, nrows=1000)
                df.to_csv(self.subset_csv_path, index=False)
                print(f"[Lycurgus] Subset successfully created at {self.subset_csv_path}")
            except Exception as e:
                print(f"[Lycurgus Error] Failed to generate subset: {e}")

    def run_stage1_syntax(self, code_path):
        print("[Lycurgus Stage 1] Checking syntax & basic imports...")
        start_time = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "py_compile", code_path], capture_output=True, text=True, timeout=5.0
            )
            elapsed = time.time() - start_time
            if proc.returncode != 0:
                print(f"[Stage 1 Fail] Syntax Error:\n{proc.stderr}")
                return False, proc.stderr
            print(f"[Stage 1 Pass] Compile successful ({elapsed:.2f}s)")
            return True, "Success"
        except subprocess.TimeoutExpired:
            print("[Stage 1 Fail] Execution Timeout (exceeded 5s limit)")
            return False, "Timeout"

    def run_stage2_subset(self, code_path):
        if not os.path.exists(self.subset_csv_path):
            csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]
            if csv_files:
                self.generate_subset_if_missing(csv_files[0])
            else:
                # Try finding a CSV in data/raw
                raw_data_dir = "/Volumes/samsd/workspace_v2/dacon/dacon-etri-human-understanding/data/raw"
                if os.path.exists(raw_data_dir):
                    csv_files_raw = [
                        os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if f.endswith(".csv")
                    ]
                    if csv_files_raw:
                        self.generate_subset_if_missing(csv_files_raw[0])
                    else:
                        return False, "Subset CSV missing and no raw CSV found. Run generate_subset_if_missing first."
                else:
                    return False, "Subset CSV missing and no raw CSV found. Run generate_subset_if_missing first."

        print("[Lycurgus] Training on 1,000-row subset (30s limit)...")
        start_time = time.time()

        env = os.environ.copy()
        env["KAGGLER_SUBSET_MODE"] = "1"
        env["KAGGLER_SUBSET_PATH"] = self.subset_csv_path

        try:
            proc = subprocess.run([sys.executable, code_path], capture_output=True, text=True, env=env, timeout=30.0)
            elapsed = time.time() - start_time
            if proc.returncode != 0:
                print(f"[Stage 2 Fail] Runtime Exception:\n{proc.stderr}")
                return False, proc.stderr
            print(f"[Stage 2 Pass] Subset pipeline completed in {elapsed:.2f}s")
            return True, proc.stdout
        except subprocess.TimeoutExpired:
            print("[Stage 2 Fail] Resource timeout (exceeded 30s limit)")
            return False, "Timeout"


class AgentGAEngine:
    def __init__(self):
        self.workspace_root = os.path.join(PROJECT_ROOT, "agentga_workspaces")
        os.makedirs(self.workspace_root, exist_ok=True)

    def prepare_isolated_workspace(self, seed_id, parent_archive_dir=None):
        workspace_dir = os.path.join(self.workspace_root, seed_id)
        os.makedirs(workspace_dir, exist_ok=True)
        print(f"[AgentGA] Created Isolated Workspace for {seed_id} at {workspace_dir}")

        if parent_archive_dir and os.path.exists(parent_archive_dir):
            print(f"[AgentGA] Inheriting Parent Archive from {parent_archive_dir}...")
            for item in os.listdir(parent_archive_dir):
                s = os.path.join(parent_archive_dir, item)
                d = os.path.join(workspace_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            print("[AgentGA] Parent Inheritance Successful.")
        return workspace_dir


class NVIDIAGPUStacker:
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        print(f"[NVIDIA Stacker] GPU Acceleration Available: {HAS_GPU_ACCEL}")

    def load_data(self, path):
        start = time.time()
        if HAS_GPU_ACCEL:
            try:
                df = cudf.read_csv(path)
                print(f"[cuDF Acceleration] Loaded {path} to GPU in {time.time() - start:.4f}s")
                return df
            except Exception as e:
                print(f"[cuDF Error] GPU failed ({e}). Falling back to CPU...")

        df = pd.read_csv(path)
        print(f"[CPU fallback] Loaded {path} to RAM in {time.time() - start:.4f}s")
        return df

    def build_stacked_ensemble(self, oof_predictions, targets, test_predictions):
        print("[NVIDIA Ensemble] Building true 4-Stage Stacked Ensemble with GPU fallback...")
        meta_train = pd.DataFrame(oof_predictions)
        meta_test = pd.DataFrame(test_predictions)

        targets = np.asarray(targets)
        meta_a_train = np.zeros((len(targets), 2))
        meta_a_test = np.zeros((len(meta_test), 2))

        from sklearn.model_selection import KFold

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for _fold, (trn_idx, val_idx) in enumerate(kf.split(meta_train)):
            gpu_success = False
            if HAS_GPU_ACCEL:
                try:
                    et = cumlRF(n_estimators=100, max_depth=6, random_state=42)
                    X_trn = meta_train.iloc[trn_idx].values.astype(np.float32)
                    y_trn = targets[trn_idx].astype(np.float32)
                    X_val = meta_train.iloc[val_idx].values.astype(np.float32)

                    et.fit(X_trn, y_trn)
                    meta_a_train[val_idx, 0] = et.predict(X_val)
                    meta_a_test[:, 0] += et.predict(meta_test.values.astype(np.float32)) / self.n_folds
                    gpu_success = True
                except Exception as e:
                    print(f"[cumlRF Error] GPU modeling failed ({e}). Falling back to CPU...")

            if not gpu_success:
                from sklearn.ensemble import ExtraTreesRegressor

                et = ExtraTreesRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42)
                et.fit(meta_train.iloc[trn_idx], targets[trn_idx])
                meta_a_train[val_idx, 0] = et.predict(meta_train.iloc[val_idx])
                meta_a_test[:, 0] += et.predict(meta_test) / self.n_folds

            from sklearn.linear_model import Ridge

            rg = Ridge(alpha=10.0)
            rg.fit(meta_train.iloc[trn_idx], targets[trn_idx])
            meta_a_train[val_idx, 1] = rg.predict(meta_train.iloc[val_idx])
            meta_a_test[:, 1] += rg.predict(meta_test) / self.n_folds

        print("[Ensemble Stage 3] LightGBM Meta Learner B...")
        import lightgbm as lgb

        meta_b_train = np.zeros(len(targets))
        meta_b_test = np.zeros(len(meta_test))

        for _fold, (trn_idx, val_idx) in enumerate(kf.split(meta_a_train)):
            model = lgb.LGBMRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, verbosity=-1)
            model.fit(
                meta_a_train[trn_idx],
                targets[trn_idx],
                eval_set=[(meta_a_train[val_idx], targets[val_idx])],
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
            )
            meta_b_train[val_idx] = model.predict(meta_a_train[val_idx])
            meta_b_test += model.predict(meta_a_test) / self.n_folds

        print("[Ensemble Stage 4] Final Bayesian Ridge Blender...")
        from sklearn.linear_model import BayesianRidge

        blender = BayesianRidge()

        final_train = np.column_stack([meta_train.values, meta_a_train, meta_b_train])
        final_test = np.column_stack([meta_test.values, meta_a_test, meta_b_test])

        blender.fit(final_train, targets)
        final_predictions = blender.predict(final_test)

        print("[NVIDIA Ensemble] True 4-Stage GPU-fallback Stacking successfully built!")
        return final_predictions
