"""Cross-model comparison figures."""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from figures.common import *
from configs.models import MODEL_REGISTRY, get_vectors_dir, get_activations_dir
from analysis.statistics import balanced_silhouette

def plot_layer_emergence(
    model_names: list[str],
    title: str,
    output_path: Path,
    n_clusters_per_emotion: int = 6,
    n_bootstrap: int = 100,
):
    """Silhouette score vs normalized layer depth for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name in model_names:
        cfg = MODEL_REGISTRY[model_name]
        vec_dir = get_vectors_dir(model_name)

        labels_path = vec_dir / "cluster_labels.json"
        if not labels_path.exists():
            continue
        cluster_map = json.loads(labels_path.read_text())

        emotion_labels_path = vec_dir / "emotion_labels.json"
        if not emotion_labels_path.exists():
            continue
        emotion_labels = json.loads(emotion_labels_path.read_text())
        cluster_ids = [cluster_map[e] for e in emotion_labels]

        silhouettes = []
        layers_norm = []

        for layer in range(cfg.n_layers):
            vec_path = vec_dir / f"emotion_vectors_layer{layer}.npy"
            if not vec_path.exists():
                continue
            vectors = np.load(vec_path)

            sil = balanced_silhouette(
                vectors,
                cluster_ids,
                k_per_cluster=n_clusters_per_emotion,
                n_bootstrap=n_bootstrap,
                seed=42,
            )
            silhouettes.append(sil["mean"])
            layers_norm.append(layer / cfg.n_layers)

        if silhouettes:
            linestyle = "--" if cfg.is_instruct else "-"
            ax.plot(layers_norm, silhouettes, label=model_name, linestyle=linestyle, marker=".", markersize=4)

    ax.set_xlabel("Normalized layer depth")
    ax.set_ylabel("Balanced silhouette score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, output_path)


def plot_cross_architecture_rsa(
    model_names: list[str],
    title: str,
    output_path: Path,
):
    """RSA: Spearman correlation between cosine-sim matrices across model pairs."""
    sim_matrices = {}

    for model_name in model_names:
        cfg = MODEL_REGISTRY[model_name]
        vec_dir = get_vectors_dir(model_name)
        vec_path = vec_dir / f"emotion_vectors_layer{cfg.analysis_layer}.npy"
        if not vec_path.exists():
            continue
        vectors = np.load(vec_path)
        sim = cosine_similarity(vectors)
        # Extract upper triangle
        triu_idx = np.triu_indices_from(sim, k=1)
        sim_matrices[model_name] = sim[triu_idx]

    names = list(sim_matrices.keys())
    n = len(names)
    rsa_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            rho, _ = stats.spearmanr(sim_matrices[names[i]], sim_matrices[names[j]])
            rsa_matrix[i, j] = rho

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(rsa_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{rsa_matrix[i,j]:.3f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="Spearman rho")
    ax.set_title(title)
    fig.tight_layout()
    save_fig(fig, output_path)
    return rsa_matrix


def plot_base_vs_instruct_pca(
    base_name: str,
    instruct_name: str,
    title: str,
    output_path: Path,
):
    """Overlay base and instruct PCA projections for same architecture."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name, marker, alpha_val in [(base_name, "o", 0.5), (instruct_name, "^", 0.7)]:
        cfg = MODEL_REGISTRY[model_name]
        vec_dir = get_vectors_dir(model_name)
        vec_path = vec_dir / f"emotion_vectors_layer{cfg.analysis_layer}.npy"
        labels_path = vec_dir / "emotion_labels.json"
        cluster_path = vec_dir / "cluster_labels.json"

        if not vec_path.exists():
            continue

        vectors = np.load(vec_path)
        labels = json.loads(labels_path.read_text())
        cluster_map = json.loads(cluster_path.read_text())

        pca = PCA(n_components=2)
        projected = pca.fit_transform(vectors)

        colors = get_cluster_colors(labels, cluster_map, EMOTION_CLUSTER_COLORS)
        ax.scatter(projected[:, 0], projected[:, 1], c=colors, marker=marker,
                  s=15, alpha=alpha_val, label=model_name)

    ax.set_title(title)
    ax.legend()
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    save_fig(fig, output_path)


def plot_probing_summary(
    model_names: list[str],
    metrics: dict,  # {model_name: {metric_name: value}}
    title: str,
    output_path: Path,
):
    """Bar chart of probing quality metrics grouped by model."""
    metric_names = ["silhouette", "implicit_acc", "intensity_rho", "preference_r"]
    x = np.arange(len(metric_names))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model_name in enumerate(model_names):
        if model_name not in metrics:
            continue
        vals = [metrics[model_name].get(m, 0) for m in metric_names]
        ax.bar(x + i * width, vals, width, label=model_name)

    ax.set_xticks(x + width * len(model_names) / 2)
    ax.set_xticklabels(["Balanced\nSilhouette", "Implicit\nAccuracy", "Intensity\nSpearman rho", "Preference\nPearson r"])
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_fig(fig, output_path)
