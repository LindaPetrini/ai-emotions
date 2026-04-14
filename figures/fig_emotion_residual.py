"""Emotion-residual clustering: do need residuals still cluster after projecting out emotion space?"""

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from figures.common import *
from analysis.statistics import balanced_silhouette
from configs.models import MODEL_REGISTRY, get_model_config
from core.vectors import load_vectors

def compute_emotion_residuals(
    emotion_vectors: np.ndarray,
    need_vectors: np.ndarray,
    variance_threshold: float = 0.90,
) -> tuple[np.ndarray, int]:
    """Project out emotion subspace from need vectors.

    1. PCA on emotion vectors to find top K components capturing variance_threshold
    2. Project need vectors onto this subspace
    3. Return residuals = need_vectors - projections

    Returns: (residuals, n_components_removed)
    """
    pca = PCA()
    pca.fit(emotion_vectors)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, variance_threshold) + 1)
    k = min(k, len(pca.components_))

    # Project need vectors onto emotion subspace
    components = pca.components_[:k]  # (k, hidden_dim)
    projections = need_vectors @ components.T @ components  # (n_needs, hidden_dim)
    residuals = need_vectors - projections

    return residuals, k


def plot_emotion_residual_clustering(
    emotion_vectors: np.ndarray,
    need_vectors: np.ndarray,
    need_labels: list[str],
    need_cluster_map: dict,
    title: str,
    output_path: Path,
    variance_threshold: float = 0.90,
):
    """THE key figure: do need residuals cluster after removing emotion subspace?"""
    residuals, k = compute_emotion_residuals(emotion_vectors, need_vectors, variance_threshold)

    # Cluster residuals
    cluster_ids = [need_cluster_map.get(l, "Unknown") for l in need_labels]
    unique_clusters = list(set(cluster_ids))

    if len(unique_clusters) < 2:
        print("  Not enough clusters for silhouette")
        return

    sil = silhouette_score(residuals, cluster_ids)

    # PCA of residuals
    pca = PCA(n_components=2)
    projected = pca.fit_transform(residuals)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: original need PCA
    pca_orig = PCA(n_components=2)
    proj_orig = pca_orig.fit_transform(need_vectors)

    ax = axes[0]
    plotted = set()
    for i, label in enumerate(need_labels):
        cluster = need_cluster_map.get(label, "")
        color = NEED_CLUSTER_COLORS.get(cluster, "#333")
        show = cluster not in plotted
        ax.scatter(proj_orig[i, 0], proj_orig[i, 1], c=color, s=20, alpha=0.7,
                  label=cluster if show else None)
        plotted.add(cluster)

    sil_orig = silhouette_score(need_vectors, cluster_ids)
    ax.set_title(f"Original need vectors\nSilhouette = {sil_orig:.4f}")
    ax.legend(fontsize=6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Right: residual PCA
    ax = axes[1]
    plotted = set()
    for i, label in enumerate(need_labels):
        cluster = need_cluster_map.get(label, "")
        color = NEED_CLUSTER_COLORS.get(cluster, "#333")
        show = cluster not in plotted
        ax.scatter(projected[i, 0], projected[i, 1], c=color, s=20, alpha=0.7,
                  label=cluster if show else None)
        plotted.add(cluster)

    ax.set_title(f"After removing emotion subspace ({k} PCs, {variance_threshold*100:.0f}% var)\nResidual silhouette = {sil:.4f}")
    ax.legend(fontsize=6)
    ax.set_xlabel("Residual PC1")
    ax.set_ylabel("Residual PC2")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    save_fig(fig, output_path)

    return {"residual_silhouette": sil, "original_silhouette": sil_orig, "n_pcs_removed": k}


def plot_direction_vs_valence(
    emotion_vectors: np.ndarray,
    direction_vectors: np.ndarray,
    need_labels: list[str],
    need_cluster_map: dict,
    title: str,
    output_path: Path,
):
    """Correlation between need direction vectors and emotion PC1 (valence)."""
    from scipy import stats as scipy_stats

    # Get emotion PC1
    pca = PCA(n_components=1)
    pca.fit(emotion_vectors)
    valence_axis = pca.components_[0]

    # Project direction vectors onto valence
    projections = direction_vectors @ valence_axis
    norms = np.linalg.norm(direction_vectors, axis=1)

    # Correlation
    r, p = scipy_stats.pearsonr(np.abs(projections), norms)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = get_cluster_colors(need_labels, need_cluster_map, NEED_CLUSTER_COLORS)
    ax.scatter(projections, norms, c=colors, s=30, alpha=0.7)

    for i, label in enumerate(need_labels):
        if i % 5 == 0:  # Annotate some
            ax.annotate(label, (projections[i], norms[i]), fontsize=5, alpha=0.6)

    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Projection onto emotion valence (PC1)")
    ax.set_ylabel("Direction vector norm")
    ax.set_title(f"{title}\nr = {r:.3f}, p = {p:.2e}")

    fig.tight_layout()
    save_fig(fig, output_path)
    return {"pearson_r": r, "pearson_p": p}


def plot_variance_threshold_sweep(
    model_names: list[str],
    output_path: Path,
    thresholds: list[float] = None,
    k_per_cluster: int = 6,
    n_bootstrap: int = 100,
):
    """Sensitivity analysis: residual silhouette vs variance threshold for all models.

    For each model and threshold, projects out the emotion subspace capturing that
    fraction of variance, then computes balanced silhouette on the residuals.

    Args:
        model_names: List of model short names from MODEL_REGISTRY.
        output_path: Where to save the figure (PDF).
        thresholds: Variance thresholds to sweep. Defaults to
            [0.50, 0.60, 0.70, 0.80, 0.90, 0.95].
        k_per_cluster: Samples per cluster for balanced silhouette.
        n_bootstrap: Bootstrap iterations for confidence intervals.
    """
    if thresholds is None:
        thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name in model_names:
        cfg = get_model_config(model_name)

        try:
            emo_vectors, emo_labels, _ = load_vectors(cfg, "emotion")
            need_vectors, need_labels, need_clusters = load_vectors(cfg, "need_combined")
        except FileNotFoundError:
            print(f"  Missing vectors for {model_name}, skipping")
            continue

        cluster_ids = [need_clusters.get(l, "Unknown") for l in need_labels]

        means = []
        ci_lows = []
        ci_highs = []
        n_pcs = []

        for thresh in thresholds:
            residuals, k = compute_emotion_residuals(emo_vectors, need_vectors, thresh)
            sil = balanced_silhouette(
                residuals, cluster_ids,
                k_per_cluster=k_per_cluster,
                n_bootstrap=n_bootstrap,
            )
            means.append(sil["mean"])
            ci_lows.append(sil["ci_low"])
            ci_highs.append(sil["ci_high"])
            n_pcs.append(k)

        means = np.array(means)
        ci_lows = np.array(ci_lows)
        ci_highs = np.array(ci_highs)

        linestyle = "--" if cfg.is_instruct else "-"
        line = ax.plot(thresholds, means, label=model_name,
                       linestyle=linestyle, marker="o", markersize=5)
        color = line[0].get_color()
        ax.fill_between(thresholds, ci_lows, ci_highs, alpha=0.15, color=color)

        # Annotate number of PCs removed at each threshold
        for i, (t, k) in enumerate(zip(thresholds, n_pcs)):
            ax.annotate(f"{k}", (t, means[i]), textcoords="offset points",
                        xytext=(0, 8), fontsize=6, ha="center", color=color, alpha=0.7)

    ax.axhline(0, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Variance threshold (fraction of emotion variance removed)")
    ax.set_ylabel("Residual balanced silhouette")
    ax.set_title("Need clustering after emotion removal:\nSensitivity to variance threshold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, output_path)

    return {"thresholds": thresholds, "models": model_names}
