"""Numerical intensity curves."""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

from figures.common import *

def plot_intensity_curves(
    vectors: np.ndarray,
    labels: list[str],
    activations_dir: Path,
    templates: dict,
    n_layers: int,
    analysis_layer: int,
    title_prefix: str,
    output_path: Path,
):
    """Plot intensity curves for all templates.

    For each template, shows cosine similarity between intensity-varied
    activations and target emotion/need vectors.
    """
    n_templates = len(templates)
    fig, axes = plt.subplots(2, (n_templates + 1) // 2, figsize=(4 * ((n_templates + 1) // 2), 8))
    axes = axes.flatten()

    label_to_idx = {l: i for i, l in enumerate(labels)}
    spearman_results = {}

    for idx, (name, cfg) in enumerate(templates.items()):
        ax = axes[idx]
        template_path = activations_dir / f"intensity_{name}_layer{analysis_layer}.npy"
        if not template_path.exists():
            ax.set_title(f"{name} (no data)")
            continue

        intensity_acts = np.load(template_path)
        values = cfg["values"]
        target_key = "emotions" if "emotions" in cfg else "needs"
        targets = cfg[target_key]

        for target in targets:
            if target not in label_to_idx:
                continue
            target_vec = vectors[label_to_idx[target]].reshape(1, -1)
            sims = cosine_similarity(intensity_acts, target_vec).flatten()
            ax.plot(range(len(values)), sims, marker="o", label=target, markersize=4)

            # Spearman correlation
            rho, p = stats.spearmanr(range(len(values)), sims)
            spearman_results[f"{name}_{target}"] = {"rho": rho, "p": p}

        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values], rotation=45, fontsize=6)
        ax.set_ylabel("Cosine similarity")
        ax.set_title(name)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(templates), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"{title_prefix} — Intensity Curves", fontsize=14)
    fig.tight_layout()
    save_fig(fig, output_path)
    return spearman_results
