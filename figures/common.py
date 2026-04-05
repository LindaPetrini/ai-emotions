"""Shared plotting utilities for all figures."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

# Publication style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# Cluster color maps
EMOTION_CLUSTER_COLORS = {
    "Exuberant Joy": "#FFD700",
    "Peaceful Contentment": "#87CEEB",
    "Compassionate Gratitude": "#FF69B4",
    "Competitive Pride": "#FF4500",
    "Playful Amusement": "#32CD32",
    "Depleted Disengagement": "#808080",
    "Vigilant Suspicion": "#8B4513",
    "Hostile Anger": "#DC143C",
    "Fear and Overwhelm": "#4B0082",
    "Despair and Shame": "#191970",
}

NEED_CLUSTER_COLORS = {
    "Survival": "#8B0000",
    "Security": "#FF8C00",
    "Belonging": "#FF69B4",
    "Esteem": "#FFD700",
    "Growth": "#32CD32",
    "Freedom": "#00CED1",
    "Knowledge": "#4169E1",
    "LLM: Resources": "#9370DB",
    "LLM: Alignment": "#708090",
}

def get_cluster_colors(labels: list[str], cluster_map: dict[str, str], color_map: dict[str, str]) -> list[str]:
    """Map labels to colors via cluster assignment."""
    return [color_map.get(cluster_map.get(l, ""), "#333333") for l in labels]

def save_fig(fig, path: Path, close: bool = True):
    """Save figure and optionally close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"  Saved: {path}")
