"""Implicit detection heatmap — works for both emotions and needs."""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from figures.common import *

def plot_implicit_heatmap(
    vectors: np.ndarray,
    labels: list[str],
    scenario_activations: np.ndarray,  # (N_scenarios, hidden_dim)
    scenario_names: list[str],
    title: str,
    output_path: Path,
):
    """Plot cosine similarity heatmap between scenarios and concept vectors.

    Scenarios on y-axis, concepts on x-axis (filtered to scenario set).
    """
    # Filter vectors to those that match scenario names
    label_to_idx = {l: i for i, l in enumerate(labels)}
    matched_idxs = [label_to_idx[n] for n in scenario_names if n in label_to_idx]
    matched_labels = [labels[i] for i in matched_idxs]
    matched_vectors = vectors[matched_idxs]

    # Cosine similarity: scenarios x matched vectors
    sim = cosine_similarity(scenario_activations, matched_vectors)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(matched_labels)))
    ax.set_xticklabels(matched_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(scenario_names)))
    ax.set_yticklabels(scenario_names)
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Cosine similarity")

    # Annotate with values
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center", fontsize=6,
                   color="white" if abs(sim[i,j]) > 0.5 else "black")

    save_fig(fig, output_path)
    return sim
