"""PCA projection (PC1=valence, PC2=arousal)."""

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

from figures.common import *

def plot_pca(
    vectors: np.ndarray,
    labels: list[str],
    cluster_map: dict[str, str],
    cluster_colors: dict[str, str],
    title: str,
    output_path: Path,
    annotate: bool = False,
):
    """2D PCA scatter plot colored by cluster."""
    pca = PCA(n_components=2)
    projected = pca.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(12, 10))

    plotted_clusters = set()
    for i, label in enumerate(labels):
        cluster = cluster_map.get(label, "Unknown")
        color = cluster_colors.get(cluster, "#333333")
        show_label = cluster not in plotted_clusters
        ax.scatter(projected[i, 0], projected[i, 1], c=color, s=20, alpha=0.7,
                  label=cluster if show_label else None)
        plotted_clusters.add(cluster)

        if annotate and len(labels) <= 50:
            ax.annotate(label, (projected[i, 0], projected[i, 1]), fontsize=5, alpha=0.6)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=7, markerscale=1.5)

    fig.tight_layout()
    save_fig(fig, output_path)
    return pca, projected
