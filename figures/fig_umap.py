"""UMAP visualization with k-means clusters."""

import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import umap

from figures.common import *

def plot_umap(
    vectors: np.ndarray,
    labels: list[str],
    cluster_map: dict[str, str],
    cluster_colors: dict[str, str],
    title: str,
    output_path: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
):
    """UMAP 2D projection colored by cluster."""
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    embedding = reducer.fit_transform(vectors)

    colors = get_cluster_colors(labels, cluster_map, cluster_colors)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot by cluster for legend
    plotted_clusters = set()
    for i, label in enumerate(labels):
        cluster = cluster_map.get(label, "Unknown")
        color = cluster_colors.get(cluster, "#333333")
        show_label = cluster not in plotted_clusters
        ax.scatter(embedding[i, 0], embedding[i, 1], c=color, s=20, alpha=0.7,
                  label=cluster if show_label else None)
        plotted_clusters.add(cluster)

    ax.set_title(title)
    ax.legend(loc="best", fontsize=7, markerscale=1.5)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    fig.tight_layout()
    save_fig(fig, output_path)
    return embedding
