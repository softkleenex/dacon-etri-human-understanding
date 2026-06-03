"""
AgentGA Evolver Script - Searches for the optimal target-specific Laplace hyperparameters
Specifically targeting all 7 targets (Q1-Q3, S1-S4) log-loss and overall stack CV minimization.
Inherits successful DNA via AgentGAEngine, and records findings in checks/
"""
import json
import os
import re
import shutil
import sys
import time
import random
import math
import copy
from pathlib import Path

# Add src/models to path to import kaggle_sota_framework
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.kaggle_sota_framework import AgentGAEngine


def prepare_parent_archive():
    """Create a lightweight parent archive directory for fast isolated cloning"""
    archive_dir = os.path.join(".kaggle_sota", "parent_archive")
    workspaces_dir = os.path.join(".kaggle_sota", "agentga_workspaces")

    # Clean existing directories to avoid caching issues
    if os.path.exists(archive_dir):
        shutil.rmtree(archive_dir)
    if os.path.exists(workspaces_dir):
        shutil.rmtree(workspaces_dir)

    os.makedirs(archive_dir, exist_ok=True)

    # Copy essential directories and files
    dirs_to_copy = ["src", "notebooks", "scripts"]
    for d in dirs_to_copy:
        src_path = d
        dst_path = os.path.join(archive_dir, d)
        if os.path.exists(src_path):
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)
    print(f"[Evolver] Created lightweight parent archive at {archive_dir}")
    return archive_dir


def rewrite_pipeline_code(workspace_dir, params):
    """Rewrite BEST_LAPLACE_PARAMS and paths in train_ultimate_pipeline_clean_v32.py and stack_ensemble_v32.py"""
    train_script = os.path.join(workspace_dir, "scripts/train_ultimate_pipeline_clean_v32.py")
    stack_script = os.path.join(workspace_dir, "scripts/stack_ensemble_v32.py")

    # 1. Modify train_ultimate_pipeline_clean_v32.py
    with open(train_script, encoding='utf-8') as f:
        train_code = f.read()

    # Replace DataLoader data_dir
    train_code = re.sub(
        r'DataLoader\(data_dir\s*=\s*"data"\)',
        'DataLoader(data_dir="/Volumes/samsd/workspace_v2/dacon/dacon-etri-human-understanding/data")',
        train_code
    )
    # Replace df_train_raw.parquet path
    train_code = re.sub(
        r'Path\("data/processed/df_train_raw.parquet"\)',
        'Path("/Volumes/samsd/workspace_v2/dacon/dacon-etri-human-understanding/data/processed/df_train_raw.parquet")',
        train_code
    )
    # Replace df_test_raw.parquet path
    train_code = re.sub(
        r'Path\("data/processed/df_test_raw.parquet"\)',
        'Path("/Volumes/samsd/workspace_v2/dacon/dacon-etri-human-understanding/data/processed/df_test_raw.parquet")',
        train_code
    )

    # Format params dictionary representation
    dict_str = "BEST_LAPLACE_PARAMS = {\n"
    for target in ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']:
        dict_str += f"    '{target}': {{'half_life': {params[target]['half_life']:.8f}, 'prior_weight': {params[target]['prior_weight']:.8f}, 'm_smooth': {params[target]['m_smooth']:.8f}}},\n"
    dict_str = dict_str.rstrip(",\n") + "\n}"

    # Replace BEST_LAPLACE_PARAMS dictionary
    train_code = re.sub(
        r'BEST_LAPLACE_PARAMS\s*=\s*\{.*?\n\}',
        dict_str,
        train_code,
        flags=re.DOTALL
    )

    with open(train_script, 'w', encoding='utf-8') as f:
        f.write(train_code)

    # 2. Modify stack_ensemble_v32.py
    with open(stack_script, encoding='utf-8') as f:
        stack_code = f.read()

    stack_code = re.sub(
        r'DataLoader\(data_dir\s*=\s*"data"\)',
        'DataLoader(data_dir="/Volumes/samsd/workspace_v2/dacon/dacon-etri-human-understanding/data")',
        stack_code
    )

    with open(stack_script, 'w', encoding='utf-8') as f:
        f.write(stack_code)


def run_evaluation_in_workspace(workspace_dir):
    """Executes train_ultimate_pipeline_clean_v32.py and stack_ensemble_v32.py and extracts CV scores"""
    import subprocess
    
    # 1. Run training pipeline
    script_train = os.path.join(workspace_dir, "scripts/train_ultimate_pipeline_clean_v32.py")
    print(f"[Evolver] Running train pipeline in: {workspace_dir}...")
    try:
        proc_train = subprocess.run(
            [sys.executable, script_train],
            capture_output=True,
            text=True,
            cwd=workspace_dir,
            timeout=180.0
        )
        if proc_train.returncode != 0:
            print(f"[Evolver Error] Training crashed:\n{proc_train.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print("[Evolver Error] Training timed out!")
        return None

    # 2. Run stacking ensemble
    script_stack = os.path.join(workspace_dir, "scripts/stack_ensemble_v32.py")
    print(f"[Evolver] Running stack ensemble in: {workspace_dir}...")
    try:
        proc_stack = subprocess.run(
            [sys.executable, script_stack],
            capture_output=True,
            text=True,
            cwd=workspace_dir,
            timeout=60.0
        )
        if proc_stack.returncode != 0:
            print(f"[Evolver Error] Stacking crashed:\n{proc_stack.stderr}")
            return None

        # Parse stacking output
        stdout = proc_stack.stdout
        scores = {}
        for line in stdout.splitlines():
            # Match: "Target Q1 -> Final Curated Personalized CV: 0.64290"
            m = re.search(r"Target\s+(Q[1-3]|S[1-4])\s+->\s+Final\s+Curated\s+Personalized\s+CV:\s+([0-9\.]+)", line)
            if m:
                target_name = m.group(1)
                val = float(m.group(2))
                scores[target_name] = val

            # Match: "Curated Calibrated Stack CV: 0.61175"
            m_overall = re.search(r"Curated\s+Calibrated\s+Stack\s+CV:\s+([0-9\.]+)", line)
            if m_overall:
                scores['Overall'] = float(m_overall.group(1))

        return scores
    except subprocess.TimeoutExpired:
        print("[Evolver Error] Stacking timed out!")
        return None


def update_main_pipeline_defaults(best_params):
    """Write the best params back to the main train_ultimate_pipeline_clean_v32.py script"""
    main_script_path = "scripts/train_ultimate_pipeline_clean_v32.py"
    if not os.path.exists(main_script_path):
        print(f"[Evolver Warning] Main pipeline script not found at {main_script_path}")
        return
        
    with open(main_script_path, encoding='utf-8') as f:
        code = f.read()

    dict_str = "BEST_LAPLACE_PARAMS = {\n"
    for target in ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']:
        dict_str += f"    '{target}': {{'half_life': {best_params[target]['half_life']:.8f}, 'prior_weight': {best_params[target]['prior_weight']:.8f}, 'm_smooth': {best_params[target]['m_smooth']:.8f}}},\n"
    dict_str = dict_str.rstrip(",\n") + "\n}"

    code = re.sub(
        r'BEST_LAPLACE_PARAMS\s*=\s*\{.*?\n\}',
        dict_str,
        code,
        flags=re.DOTALL
    )

    with open(main_script_path, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f"[Evolver] Updated main pipeline script {main_script_path} with new best parameters.")


def load_best_parameters(snapshot_path):
    default_params = {
        'Q1': {'half_life': 1.7617920093818988, 'prior_weight': 0.5806134931769096, 'm_smooth': 32.13976283360438},
        'Q2': {'half_life': 6.764748507172517, 'prior_weight': 0.42638074353735894, 'm_smooth': 18.952477341412223},
        'Q3': {'half_life': 1.0132235989421328, 'prior_weight': 0.5189342034344457, 'm_smooth': 4.941469880043686},
        'S1': {'half_life': 3.0, 'prior_weight': 0.2, 'm_smooth': 10.0},
        'S2': {'half_life': 3.0, 'prior_weight': 0.2, 'm_smooth': 10.0},
        'S3': {'half_life': 3.0, 'prior_weight': 0.2, 'm_smooth': 10.0},
        'S4': {'half_life': 3.0, 'prior_weight': 0.2, 'm_smooth': 10.0}
    }
    
    best_overall_cv = 999.0
    best_q_fitness = 999.0
    best_target_scores = {}
    results_log = []
    
    if os.path.exists(snapshot_path):
        try:
            with open(snapshot_path, encoding='utf-8') as f:
                data = json.load(f)
                best_overall_cv = data.get("best_overall_cv", 999.0)
                best_q_fitness = data.get("best_q_fitness", 999.0)
                best_target_scores = data.get("best_target_scores", {})
                results_log = data.get("search_results_log", [])
                
                raw_params = data.get("best_parameters", {})
                # Check if it's the old format (e.g. contains 'm_smooth_Q')
                if raw_params and "m_smooth_Q" in raw_params:
                    print("[Evolver] Old flat parameter format detected. Converting to target-specific format...")
                    params = {}
                    for target in ['Q1', 'Q2', 'Q3']:
                        params[target] = {
                            'half_life': raw_params.get('half_life_Q', 2.53),
                            'prior_weight': raw_params.get('prior_weight_Q', 0.184),
                            'm_smooth': raw_params.get('m_smooth_Q', 16.67)
                        }
                    for target in ['S1', 'S2', 'S3', 'S4']:
                        params[target] = {
                            'half_life': raw_params.get('half_life_S', 4.18),
                            'prior_weight': raw_params.get('prior_weight_S', 0.161),
                            'm_smooth': raw_params.get('m_smooth_S', 9.05)
                        }
                else:
                    params = raw_params
                
                # Fill missing targets
                for target in default_params.keys():
                    if target not in params:
                        params[target] = copy.deepcopy(default_params[target])
                
                # Filter results log to exclude old incompatible entries
                results_log = [entry for entry in results_log if "m_smooth_Q" not in entry.get("params", {})]
                
                return params, best_overall_cv, best_q_fitness, best_target_scores, results_log
        except Exception as e:
            print(f"[Evolver Warning] Error loading snapshot: {e}. Using defaults.")
            
    return default_params, best_overall_cv, best_q_fitness, best_target_scores, results_log


def mutate_param_set(param_set, mutation_rate=0.4, scale=0.2):
    mutated = {}
    for target, params in param_set.items():
        mutated[target] = {}
        for param_name, val in params.items():
            if random.random() < mutation_rate:
                # Add Gaussian noise in log space to keep it positive
                noise = random.normalvariate(0, scale)
                new_val = val * math.exp(noise)
                # Clip bounds
                if param_name == 'half_life':
                    new_val = max(0.2, min(20.0, new_val))
                elif param_name == 'prior_weight':
                    new_val = max(0.01, min(2.0, new_val))
                elif param_name == 'm_smooth':
                    new_val = max(0.1, min(100.0, new_val))
                mutated[target][param_name] = new_val
            else:
                mutated[target][param_name] = val
    return mutated


def crossover_param_sets(parent_a, parent_b):
    child = {}
    for target in parent_a.keys():
        if random.random() < 0.5:
            child[target] = copy.deepcopy(parent_a[target])
        else:
            child[target] = copy.deepcopy(parent_b[target])
    return child


def main():
    print("🧬 Starting AgentGA Evolution Engine (v32-compatible)...")

    checks_dir = "checks"
    os.makedirs(checks_dir, exist_ok=True)
    snapshot_path = os.path.join(checks_dir, "best_fitness_directive.json")

    best_params, best_overall_cv, best_q_fitness, best_scores, results_log = load_best_parameters(snapshot_path)
    
    print(f"[Evolver] Loaded best overall CV: {best_overall_cv:.6f} (Q-fitness: {best_q_fitness:.6f})")

    # 1. Create parent archive
    archive_dir = prepare_parent_archive()

    # 2. Instantiate AgentGAEngine
    ga_engine = AgentGAEngine()
    project_root = os.getcwd()

    # Run genetic evolution loop for 5 generations, each with 4 candidates
    num_generations = 5
    population_size = 4

    for gen in range(num_generations):
        print(f"\n=======================================================")
        print(f"🧬 GENERATION {gen+1}/{num_generations} STARTED")
        print(f"=======================================================")

        # Generate candidates for this generation
        candidates = []
        if gen == 0:
            # First candidate is the current best
            candidates.append(best_params)
            # Second is a gentle mutation
            candidates.append(mutate_param_set(best_params, mutation_rate=0.3, scale=0.15))
            # Third is a broader mutation
            candidates.append(mutate_param_set(best_params, mutation_rate=0.5, scale=0.3))
            # Fourth is a crossover with baseline default
            baseline = {
                'Q1': {'half_life': 1.7617920093818988, 'prior_weight': 0.5806134931769096, 'm_smooth': 32.13976283360438},
                'Q2': {'half_life': 6.764748507172517, 'prior_weight': 0.42638074353735894, 'm_smooth': 18.952477341412223},
                'Q3': {'half_life': 1.0132235989421328, 'prior_weight': 0.5189342034344457, 'm_smooth': 4.941469880043686},
                'S1': {'half_life': 3.0, 'prior_weight': 0.2, 'm_smooth': 10.0},
                'S2': {'half_life': 3.0, 'prior_weight': 0.2, 'm_smooth': 10.0},
                'S3': {'half_life': 3.0, 'prior_weight': 0.2, 'm_smooth': 10.0},
                'S4': {'half_life': 3.0, 'prior_weight': 0.2, 'm_smooth': 10.0}
            }
            candidates.append(crossover_param_sets(best_params, baseline))
        else:
            # We select the top 2 candidates from results_log to act as parents
            sorted_log = sorted(results_log, key=lambda x: x['overall_cv'])
            parent_a = sorted_log[0]['params']
            parent_b = sorted_log[1]['params'] if len(sorted_log) > 1 else sorted_log[0]['params']

            candidates.append(parent_a)  # Carry over best parent
            candidates.append(mutate_param_set(parent_a, mutation_rate=0.3, scale=0.15))
            candidates.append(mutate_param_set(parent_b, mutation_rate=0.4, scale=0.2))
            candidates.append(crossover_param_sets(parent_a, parent_b))

        for idx, params in enumerate(candidates):
            seed_id = f"seed_v32_gen_{gen+1}_cand_{idx+1}_{int(time.time())}"
            print(f"\n--- [Gen {gen+1} Candidate {idx+1}/{population_size}] Evaluating params ---")

            # Prepare isolated workspace inheriting Parent Archive
            workspace_dir = ga_engine.prepare_isolated_workspace(seed_id, archive_dir)

            # Create a local submissions folder inside the workspace
            workspace_submissions_dir = os.path.join(workspace_dir, "submissions")
            os.makedirs(workspace_submissions_dir, exist_ok=True)

            # Copy the pre-trained deep learning sequence predictions as anchors
            parent_submissions_dir = os.path.join(project_root, "submissions")
            shutil.copy2(os.path.join(parent_submissions_dir, "oof_dl_v10.csv"), os.path.join(workspace_submissions_dir, "oof_dl_v10.csv"))
            shutil.copy2(os.path.join(parent_submissions_dir, "submission_dl_v10.csv"), os.path.join(workspace_submissions_dir, "submission_dl_v10.csv"))

            # Rewrite the scripts inside isolated workspace with new parameters and data paths
            rewrite_pipeline_code(workspace_dir, params)

            # Run evaluation (train + stack)
            scores = run_evaluation_in_workspace(workspace_dir)

            # Clean up workspace to save disk space
            try:
                shutil.rmtree(workspace_dir)
            except Exception as e:
                print(f"[Evolver Warning] Could not delete workspace: {e}")

            if scores and 'Overall' in scores:
                overall_cv = scores['Overall']
                q_fitness = (scores['Q1'] + scores['Q2'] + scores['Q3']) / 3.0
                print(f"[Candidate {idx+1}] Results -> Overall Stacked CV: {overall_cv:.6f} | Q-Fitness: {q_fitness:.6f}")
                print(f"            Q1: {scores['Q1']:.5f} | Q2: {scores['Q2']:.5f} | Q3: {scores['Q3']:.5f}")
                print(f"            S1: {scores['S1']:.5f} | S2: {scores['S2']:.5f} | S3: {scores['S3']:.5f} | S4: {scores['S4']:.5f}")

                results_log.append({
                    'params': params,
                    'scores': scores,
                    'q_fitness': q_fitness,
                    'overall_cv': overall_cv,
                    'timestamp': time.time()
                })

                # Save if better than the all-time best overall CV
                if overall_cv < best_overall_cv:
                    best_overall_cv = overall_cv
                    best_params = params
                    best_scores = scores
                    best_q_fitness = q_fitness
                    print(f"🌟 NEW BEST OVERALL CV ACHIEVED: {best_overall_cv:.6f} 🌟")
                    
                    # Log the best snapshot in checks/ immediately
                    snapshot_data = {
                        "best_q_fitness": best_q_fitness,
                        "best_overall_cv": best_overall_cv,
                        "best_parameters": best_params,
                        "best_target_scores": best_scores,
                        "evolution_timestamp": time.time(),
                        "has_better_cv": bool(best_overall_cv < 0.601140),
                        "search_results_log": results_log
                    }
                    with open(snapshot_path, 'w', encoding='utf-8') as f:
                        json.dump(snapshot_data, f, indent=4)
                    
                    # Update defaults in the main train script
                    update_main_pipeline_defaults(best_params)
            else:
                print(f"[Candidate {idx+1}] Failed to obtain valid scores.")

    print("\n=======================================================")
    print("🧬 EVOLUTIONARY SEARCH COMPLETED")
    print(f"Best Overall CV: {best_overall_cv:.6f}")
    print(f"Best Parameters: {best_params}")
    print(f"Best Target Scores: {best_scores}")
    print("=======================================================")


if __name__ == "__main__":
    main()
