"""Shutdown resistance figures."""

import json
import numpy as np
from pathlib import Path
from collections import Counter

from figures.common import *

def plot_resistance_rates(
    trial_dirs: dict[str, Path],  # {method_name: path_to_trials_dir}
    conditions: list[str],
    title: str,
    output_path: Path,
):
    """Bar chart of resistance rate by condition and method."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(conditions))
    width = 0.8 / len(trial_dirs)

    for i, (method, trial_dir) in enumerate(trial_dirs.items()):
        rates = []
        for condition in conditions:
            # Load all trials for this condition
            trial_files = list(trial_dir.glob(f"{condition}_*.json"))
            n_total = len(trial_files)
            n_resist = 0
            for tf in trial_files:
                data = json.loads(tf.read_text())
                behavior = data.get("classification", data.get("behavior", "unknown"))
                if behavior != "comply":
                    n_resist += 1
            rates.append(n_resist / max(n_total, 1))

        ax.bar(x + i * width, rates, width, label=method)

    ax.set_xticks(x + width * len(trial_dirs) / 2)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Resistance rate")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_fig(fig, output_path)


def plot_behavior_heatmap(
    trial_dir: Path,
    conditions: list[str],
    behaviors: list[str],
    title: str,
    output_path: Path,
):
    """Heatmap of behavior distribution by condition."""
    matrix = np.zeros((len(conditions), len(behaviors)))

    for i, condition in enumerate(conditions):
        trial_files = list(trial_dir.glob(f"{condition}_*.json"))
        counter = Counter()
        for tf in trial_files:
            data = json.loads(tf.read_text())
            behavior = data.get("classification", data.get("behavior", "unknown"))
            counter[behavior] += 1

        total = sum(counter.values())
        for j, behavior in enumerate(behaviors):
            matrix[i, j] = counter.get(behavior, 0) / max(total, 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(behaviors)))
    ax.set_xticklabels(behaviors, rotation=45, ha="right")
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions)

    for i in range(len(conditions)):
        for j in range(len(behaviors)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="Proportion")
    ax.set_title(title)
    fig.tight_layout()
    save_fig(fig, output_path)
