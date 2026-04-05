"""Cosine similarity matrix — parameterized for emotions, needs, or nouns."""

import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from figures.common import *

def plot_cosine_similarity(
    vectors: np.ndarray,
    labels: list[str],
    cluster_map: dict[str, str],
    cluster_colors: dict[str, str],
    title: str,
    output_path: Path,
    sort_by_cluster: bool = True,
):
    """Plot NxN cosine similarity matrix, optionally sorted by cluster."""
    if sort_by_cluster:
        # Sort by cluster
        order = sorted(range(len(labels)), key=lambda i: cluster_map.get(labels[i], ""))
        vectors = vectors[order]
        labels = [labels[i] for i in order]

    sim = cosine_similarity(vectors)
    colors = get_cluster_colors(labels, cluster_map, cluster_colors)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Cluster boundaries
    if sort_by_cluster:
        clusters_sorted = [cluster_map.get(l, "") for l in labels]
        boundaries = []
        for i in range(1, len(clusters_sorted)):
            if clusters_sorted[i] != clusters_sorted[i-1]:
                boundaries.append(i)
        for b in boundaries:
            ax.axhline(b - 0.5, color="black", linewidth=0.5)
            ax.axvline(b - 0.5, color="black", linewidth=0.5)

    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Cosine similarity")

    # Tick labels only if small enough
    if len(labels) <= 50:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=4)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=4)

    fig.tight_layout()
    save_fig(fig, output_path)
    return sim
