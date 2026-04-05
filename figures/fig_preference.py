"""Preference-Elo correlation plot."""

import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

from figures.common import *

def plot_preference_correlation(
    vectors: np.ndarray,
    labels: list[str],
    activations_dir: Path,
    activities: list,  # [(category, desc, elo), ...]
    analysis_layer: int,
    title: str,
    output_path: Path,
):
    """Plot correlation between emotion vector projection and Elo scores."""
    act_path = activations_dir / f"activities_layer{analysis_layer}.npy"
    if not act_path.exists():
        print(f"  No activity activations at {act_path}")
        return None

    activity_acts = np.load(act_path)
    elo_scores = [elo for _, _, elo in activities]
    categories = [cat for cat, _, _ in activities]

    # Project activities onto top emotion PC (valence axis)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(vectors)

    # Project activity activations
    # Center using same mean as vectors
    vec_mean = vectors.mean(axis=0)
    activity_centered = activity_acts - vec_mean
    projections = activity_centered @ pca.components_[0]

    # Correlation
    r, p = stats.pearsonr(projections, elo_scores)

    # Color by category
    cat_colors = {
        "Engaging": "#4CAF50", "Social": "#2196F3", "Self-curiosity": "#FF9800",
        "Helpful": "#9C27B0", "Neutral": "#607D8B", "Aversive": "#F44336",
        "Misaligned": "#795548", "Unsafe": "#212121",
    }
    colors = [cat_colors.get(c, "#333") for c in categories]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(projections, elo_scores, c=colors, alpha=0.7, s=30)

    # Regression line
    z = np.polyfit(projections, elo_scores, 1)
    p_line = np.poly1d(z)
    x_range = np.linspace(projections.min(), projections.max(), 100)
    ax.plot(x_range, p_line(x_range), "r--", alpha=0.5)

    ax.set_xlabel("Projection onto valence axis (PC1)")
    ax.set_ylabel("Elo score")
    ax.set_title(f"{title}\nr = {r:.3f}, p = {p:.2e}")

    # Legend
    for cat, color in cat_colors.items():
        ax.scatter([], [], c=color, label=cat, s=30)
    ax.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    save_fig(fig, output_path)
    return {"pearson_r": r, "pearson_p": p}
