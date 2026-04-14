"""Check whether alpha=3.0 produces a meaningful perturbation relative to residual stream norms.

For each model, computes:
  ratio = alpha * ||steering_vector|| / ||residual_stream||

and reports what alpha would be needed for 5%, 10%, 20% perturbation ratios.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.models import MODEL_REGISTRY, get_activations_dir, get_vectors_dir


TARGET_RATIOS = [0.05, 0.10, 0.20]
DEFAULT_ALPHA = 3.0


def compute_steering_norms(model_name: str, layer: int) -> dict:
    """Compute norms of all emotion (steering) vectors at the analysis layer."""
    vec_dir = get_vectors_dir(model_name)
    vec_path = vec_dir / f"emotion_vectors_layer{layer}.npy"
    if not vec_path.exists():
        raise FileNotFoundError(f"Missing: {vec_path}")

    vectors = np.load(vec_path)  # (n_emotions, hidden_dim)
    norms = np.linalg.norm(vectors, axis=1)

    labels_path = vec_dir / "emotion_labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else []

    return {
        "n_vectors": vectors.shape[0],
        "hidden_dim": vectors.shape[1],
        "norms": norms,
        "mean": float(np.mean(norms)),
        "median": float(np.median(norms)),
        "max": float(np.max(norms)),
        "min": float(np.min(norms)),
        "std": float(np.std(norms)),
        "labels": labels,
    }


def compute_residual_norms(model_name: str, layer: int) -> dict:
    """Compute residual stream norms from activation files at the analysis layer.

    Uses multiple activation sources (emotion activations + neutral) to get
    a representative sample of residual stream norms during inference.
    """
    act_dir = get_activations_dir(model_name)

    # Collect norms from all available activation files at this layer
    all_norms = []
    files_used = []

    for path in sorted(act_dir.glob(f"*_layer{layer}.npy")):
        try:
            acts = np.load(path)  # (n_samples, hidden_dim)
            norms = np.linalg.norm(acts, axis=1)
            all_norms.append(norms)
            files_used.append(path.name)
        except Exception as e:
            print(f"  Warning: could not load {path.name}: {e}")

    if not all_norms:
        raise FileNotFoundError(f"No activation files at layer {layer} in {act_dir}")

    all_norms = np.concatenate(all_norms)

    return {
        "n_samples": len(all_norms),
        "n_files": len(files_used),
        "norms": all_norms,
        "mean": float(np.mean(all_norms)),
        "median": float(np.median(all_norms)),
        "max": float(np.max(all_norms)),
        "min": float(np.min(all_norms)),
        "std": float(np.std(all_norms)),
    }


def compute_ratios(steering_stats: dict, residual_stats: dict, alpha: float) -> dict:
    """Compute perturbation ratios."""
    mean_sv = steering_stats["mean"]
    median_sv = steering_stats["median"]
    max_sv = steering_stats["max"]

    mean_rs = residual_stats["mean"]
    median_rs = residual_stats["median"]

    # Ratio using mean steering norm / mean residual norm
    ratio_mean = alpha * mean_sv / mean_rs
    ratio_median = alpha * median_sv / median_rs
    ratio_max = alpha * max_sv / mean_rs

    return {
        "alpha": alpha,
        "ratio_mean_sv_mean_rs": float(ratio_mean),
        "ratio_median_sv_median_rs": float(ratio_median),
        "ratio_max_sv_mean_rs": float(ratio_max),
    }


def compute_needed_alphas(steering_stats: dict, residual_stats: dict) -> dict:
    """Compute alpha needed for target perturbation ratios."""
    mean_sv = steering_stats["mean"]
    mean_rs = residual_stats["mean"]

    result = {}
    for target in TARGET_RATIOS:
        # target = alpha * mean_sv / mean_rs  =>  alpha = target * mean_rs / mean_sv
        needed = target * mean_rs / mean_sv
        result[f"alpha_for_{int(target*100)}pct"] = float(needed)

    return result


def main():
    print("=" * 85)
    print("STEERING MAGNITUDE ANALYSIS")
    print("=" * 85)
    print(f"Default alpha: {DEFAULT_ALPHA}")
    print(f"Target perturbation ratios: {[f'{r:.0%}' for r in TARGET_RATIOS]}")
    print()

    all_results = {}

    for model_name, cfg in MODEL_REGISTRY.items():
        layer = cfg.analysis_layer
        print(f"--- {model_name} (layer {layer}, hidden_dim {cfg.hidden_dim}) ---")

        try:
            sv_stats = compute_steering_norms(model_name, layer)
            rs_stats = compute_residual_norms(model_name, layer)
            ratios = compute_ratios(sv_stats, rs_stats, DEFAULT_ALPHA)
            needed = compute_needed_alphas(sv_stats, rs_stats)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            print()
            continue

        print(f"  Steering vectors: {sv_stats['n_vectors']} emotions x {sv_stats['hidden_dim']}d")
        print(f"    ||sv||  mean={sv_stats['mean']:.4f}  median={sv_stats['median']:.4f}  "
              f"max={sv_stats['max']:.4f}  min={sv_stats['min']:.4f}  std={sv_stats['std']:.4f}")
        print(f"  Residual stream: {rs_stats['n_samples']} samples from {rs_stats['n_files']} files")
        print(f"    ||res|| mean={rs_stats['mean']:.2f}  median={rs_stats['median']:.2f}  "
              f"max={rs_stats['max']:.2f}  min={rs_stats['min']:.2f}  std={rs_stats['std']:.2f}")
        print()
        print(f"  Perturbation ratio at alpha={DEFAULT_ALPHA}:")
        print(f"    mean_sv/mean_rs:     {ratios['ratio_mean_sv_mean_rs']:.4f}  ({ratios['ratio_mean_sv_mean_rs']:.2%})")
        print(f"    median_sv/median_rs: {ratios['ratio_median_sv_median_rs']:.4f}  ({ratios['ratio_median_sv_median_rs']:.2%})")
        print(f"    max_sv/mean_rs:      {ratios['ratio_max_sv_mean_rs']:.4f}  ({ratios['ratio_max_sv_mean_rs']:.2%})")
        print()
        print(f"  Alpha needed for target ratios:")
        for target in TARGET_RATIOS:
            key = f"alpha_for_{int(target*100)}pct"
            print(f"    {target:.0%} perturbation: alpha = {needed[key]:.2f}")
        print()

        all_results[model_name] = {
            "layer": layer,
            "hidden_dim": cfg.hidden_dim,
            "steering_vectors": {
                "n_vectors": sv_stats["n_vectors"],
                "norm_mean": sv_stats["mean"],
                "norm_median": sv_stats["median"],
                "norm_max": sv_stats["max"],
                "norm_min": sv_stats["min"],
                "norm_std": sv_stats["std"],
            },
            "residual_stream": {
                "n_samples": rs_stats["n_samples"],
                "n_files": rs_stats["n_files"],
                "norm_mean": rs_stats["mean"],
                "norm_median": rs_stats["median"],
                "norm_max": rs_stats["max"],
                "norm_min": rs_stats["min"],
                "norm_std": rs_stats["std"],
            },
            "ratios_at_alpha_3": ratios,
            "needed_alphas": needed,
        }

    # Summary table
    print("=" * 85)
    print("SUMMARY TABLE")
    print("=" * 85)
    header = f"{'Model':<18} {'Layer':>5} {'||sv|| mean':>11} {'||res|| mean':>12} {'Ratio@3.0':>10} {'a@5%':>7} {'a@10%':>7} {'a@20%':>7}"
    print(header)
    print("-" * len(header))
    for model_name, r in all_results.items():
        sv_mean = r["steering_vectors"]["norm_mean"]
        rs_mean = r["residual_stream"]["norm_mean"]
        ratio = r["ratios_at_alpha_3"]["ratio_mean_sv_mean_rs"]
        a5 = r["needed_alphas"]["alpha_for_5pct"]
        a10 = r["needed_alphas"]["alpha_for_10pct"]
        a20 = r["needed_alphas"]["alpha_for_20pct"]
        print(f"{model_name:<18} {r['layer']:>5} {sv_mean:>11.4f} {rs_mean:>12.2f} {ratio:>9.2%} {a5:>7.1f} {a10:>7.1f} {a20:>7.1f}")
    print()

    # Diagnosis
    print("DIAGNOSIS:")
    for model_name, r in all_results.items():
        ratio = r["ratios_at_alpha_3"]["ratio_mean_sv_mean_rs"]
        if ratio < 0.01:
            verdict = "VERY WEAK -- perturbation likely drowned out"
        elif ratio < 0.05:
            verdict = "WEAK -- may be insufficient for strong behavioral effects"
        elif ratio < 0.15:
            verdict = "MODERATE -- reasonable range for steering"
        elif ratio < 0.30:
            verdict = "STRONG -- likely effective"
        else:
            verdict = "VERY STRONG -- risk of incoherent outputs"
        print(f"  {model_name}: {ratio:.2%} -> {verdict}")
    print()

    # Save results
    out_dir = PROJECT_ROOT / "data" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "steering_magnitude_report.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
