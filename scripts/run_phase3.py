#!/usr/bin/env python3
"""Phase 3: Generate figures, run controls, check replication criteria.

Runs locally after syncing activations/vectors from GPU VM.

Usage:
    python -m scripts.run_phase3                    # all models
    python -m scripts.run_phase3 --model qwen-7b-base  # single model
    python -m scripts.run_phase3 --skip-controls    # faster, just figures
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
from configs.models import MODEL_REGISTRY, get_model_config, get_vectors_dir, get_activations_dir, get_figures_dir, ALL_MODEL_NAMES
from configs.emotions import (
    ALL_EMOTIONS, EMOTION_TO_CLUSTER, EMOTION_CLUSTERS,
    IMPLICIT_SCENARIOS, INTENSITY_TEMPLATES, ACTIVITIES,
    VALENCE_LABELS,
)
from core.vectors import load_vectors
from analysis.statistics import (
    balanced_silhouette, valence_auc, implicit_accuracy,
    intensity_spearman, check_replication_criteria,
)
from analysis.random_controls import (
    random_vector_silhouette, shuffled_label_silhouette, noun_vector_control,
)
from figures.generate_all import generate_stream1_per_model, generate_stream1_comparison, generate_stream2


def run_replication_check(model_name: str) -> dict:
    """Check replication criteria for one model."""
    cfg = get_model_config(model_name)
    act_dir = get_activations_dir(model_name)
    vec_dir = get_vectors_dir(model_name)

    try:
        vectors, labels, cluster_map = load_vectors(cfg, "emotion")
    except FileNotFoundError:
        print(f"  No vectors for {model_name}")
        return {}

    cluster_ids = [cluster_map[l] for l in labels]

    # Load scenario activations
    scenario_path = act_dir / f"scenarios_layer{cfg.analysis_layer}.npy"
    names_path = act_dir / "scenario_names.json"
    scenario_acts = None
    scenario_names = None
    if scenario_path.exists() and names_path.exists():
        scenario_acts = np.load(scenario_path)
        scenario_names = json.loads(names_path.read_text())

    # Intensity
    intensity_results = intensity_spearman(
        vectors, labels, act_dir, INTENSITY_TEMPLATES, cfg.analysis_layer
    )

    results = check_replication_criteria(
        cfg, vectors, labels, cluster_map, VALENCE_LABELS,
        scenario_acts, scenario_names, intensity_results,
    )

    return results


def run_controls(model_name: str) -> dict:
    """Run random controls for one model."""
    cfg = get_model_config(model_name)
    try:
        vectors, labels, cluster_map = load_vectors(cfg, "emotion")
    except FileNotFoundError:
        return {}

    cluster_ids = [cluster_map[l] for l in labels]

    print(f"  Random vector control...")
    rv = random_vector_silhouette(len(labels), cfg.hidden_dim, cluster_ids, real_vectors=vectors)

    print(f"  Shuffled label control...")
    sl = shuffled_label_silhouette(vectors, cluster_ids, n_repeats=1000)

    return {"random_vectors": rv, "shuffled_labels": sl}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    models = [args.model] if args.model else ALL_MODEL_NAMES

    all_results = {}

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        # Check data exists
        cfg = get_model_config(model_name)
        vec_dir = get_vectors_dir(model_name)
        vec_path = vec_dir / f"emotion_vectors_layer{cfg.analysis_layer}.npy"
        if not vec_path.exists():
            print(f"  No vectors found, skipping")
            continue

        # 1. Replication criteria
        print(f"\n  --- Replication criteria ---")
        repl = run_replication_check(model_name)
        all_results[model_name] = {"replication": repl}

        for key, val in repl.items():
            if isinstance(val, dict) and "passed" in val:
                status = "PASS" if val["passed"] else "FAIL"
                print(f"  {key}: {status} (value={val.get('value', 'N/A')})")

        if repl.get("all_passed"):
            print(f"  => ALL CRITERIA PASSED")
        else:
            print(f"  => SOME CRITERIA FAILED")

        # 2. Controls
        if not args.skip_controls:
            print(f"\n  --- Random controls ---")
            controls = run_controls(model_name)
            all_results[model_name]["controls"] = controls
            if "random_vectors" in controls:
                print(f"  Random vector silhouette: {controls['random_vectors']['mean']:.4f}")
            if "shuffled_labels" in controls:
                print(f"  Shuffled label silhouette: {controls['shuffled_labels']['mean']:.4f}")
                print(f"  Real vs shuffled p-value: {controls['shuffled_labels']['p_value']:.4f}")

        # 3. Figures
        if not args.skip_figures:
            print(f"\n  --- Generating figures ---")
            generate_stream1_per_model(model_name)

    # Cross-model comparison
    if not args.skip_figures and len(models) > 1:
        print(f"\n  --- Cross-model comparison figures ---")
        generate_stream1_comparison()

    # Stream 2 (needs)
    if not args.skip_figures:
        for model_name in models:
            cfg = get_model_config(model_name)
            vec_dir = get_vectors_dir(model_name)
            combined_path = vec_dir / f"need_combined_vectors_layer{cfg.analysis_layer}.npy"
            if combined_path.exists():
                print(f"\n  --- Stream 2 figures for {model_name} ---")
                generate_stream2(model_name)

    # Save all results
    results_dir = get_figures_dir(1)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "phase3_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Go/No-Go checks
    print(f"\n{'='*60}")
    print(f"  GO/NO-GO CHECKS")
    print(f"{'='*60}")

    for model_name in models:
        if model_name not in all_results:
            continue
        repl = all_results[model_name].get("replication", {})
        if repl.get("all_passed"):
            print(f"  {model_name}: GO (all replication criteria passed)")
        else:
            failed = [k for k, v in repl.items() if isinstance(v, dict) and not v.get("passed", True)]
            print(f"  {model_name}: CAUTION (failed: {failed})")


if __name__ == "__main__":
    main()
