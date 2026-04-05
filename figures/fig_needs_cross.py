"""Cross-analysis figures for needs vs emotions."""

import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from figures.common import *

def plot_need_x_emotion(
    emotion_vectors: np.ndarray,
    emotion_labels: list[str],
    need_vectors: np.ndarray,
    need_labels: list[str],
    emotion_cluster_map: dict,
    need_cluster_map: dict,
    title: str,
    output_path: Path,
):
    """Cross-similarity matrix: needs (rows) x emotions (cols)."""
    sim = cosine_similarity(need_vectors, emotion_vectors)

    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Cosine similarity")

    if len(need_labels) <= 90:
        ax.set_yticks(range(len(need_labels)))
        ax.set_yticklabels(need_labels, fontsize=4)
    if len(emotion_labels) <= 50:
        ax.set_xticks(range(len(emotion_labels)))
        ax.set_xticklabels(emotion_labels, rotation=90, fontsize=4)

    fig.tight_layout()
    save_fig(fig, output_path)
    return sim


def plot_need_emotion_alignment(
    emotion_vectors: np.ndarray,
    emotion_labels: list[str],
    need_vectors: np.ndarray,
    need_labels: list[str],
    need_cluster_map: dict,
    title: str,
    output_path: Path,
    top_k: int = 3,
):
    """Per-need cosine similarity with top-K most similar emotions. Heatmap."""
    sim = cosine_similarity(need_vectors, emotion_vectors)

    # For each need, find top-K emotions
    top_emotions = set()
    for i in range(len(need_labels)):
        top_idx = np.argsort(sim[i])[-top_k:]
        for j in top_idx:
            top_emotions.add(j)

    top_emotion_indices = sorted(top_emotions)
    filtered_sim = sim[:, top_emotion_indices]
    filtered_labels = [emotion_labels[i] for i in top_emotion_indices]

    fig, ax = plt.subplots(figsize=(max(12, len(filtered_labels) * 0.3), 10))
    im = ax.imshow(filtered_sim, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")

    ax.set_xticks(range(len(filtered_labels)))
    ax.set_xticklabels(filtered_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(need_labels)))
    ax.set_yticklabels(need_labels, fontsize=5)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    fig.tight_layout()
    save_fig(fig, output_path)


def plot_met_unmet_pca(
    met_vectors: np.ndarray,
    unmet_vectors: np.ndarray,
    need_labels: list[str],
    need_cluster_map: dict,
    title: str,
    output_path: Path,
):
    """PCA showing met vs unmet for each need. Met=filled, unmet=open markers."""
    combined = np.vstack([met_vectors, unmet_vectors])
    pca = PCA(n_components=2)
    projected = pca.fit_transform(combined)

    n = len(need_labels)
    met_proj = projected[:n]
    unmet_proj = projected[n:]

    fig, ax = plt.subplots(figsize=(12, 10))

    plotted = set()
    for i, label in enumerate(need_labels):
        cluster = need_cluster_map.get(label, "")
        color = NEED_CLUSTER_COLORS.get(cluster, "#333")
        is_llm = cluster.startswith("LLM")
        marker_met = "D" if is_llm else "o"
        marker_unmet = "d" if is_llm else "^"

        show_label = cluster not in plotted
        ax.scatter(met_proj[i, 0], met_proj[i, 1], c=color, marker=marker_met,
                  s=30, alpha=0.8, label=f"{cluster} (met)" if show_label else None)
        ax.scatter(unmet_proj[i, 0], unmet_proj[i, 1], c=color, marker=marker_unmet,
                  s=30, alpha=0.4, edgecolors=color, facecolors="none")
        plotted.add(cluster)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}%)")
    ax.set_ylabel(f"PC2 ({var2:.1f}%)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, output_path)
