"""Compute emotion and need direction vectors from extracted activations."""

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from configs.models import ModelConfig, get_activations_dir, get_vectors_dir
from configs.needs import sanitize_need_name

NEUTRAL_VARIANCE_THRESHOLD = 0.50


def compute_emotion_vectors(
    cfg: ModelConfig,
    emotions: list[str],
    cluster_map: dict[str, str],
) -> None:
    """Compute deconfounded emotion vectors for all layers.

    For each layer:
      1. Average story activations per emotion -> (N, hidden_dim)
      2. Center by subtracting global mean
      3. Deconfound by projecting out top PCs from neutral activations
      4. Save raw and deconfounded vectors
    """
    act_dir = get_activations_dir(cfg.short_name)
    vec_dir = get_vectors_dir(cfg.short_name)
    vec_dir.mkdir(parents=True, exist_ok=True)

    # Save labels
    (vec_dir / "emotion_labels.json").write_text(json.dumps(emotions))
    (vec_dir / "cluster_labels.json").write_text(
        json.dumps({e: cluster_map[e] for e in emotions})
    )

    for layer in range(cfg.n_layers):
        vec_path = vec_dir / f"emotion_vectors_layer{layer}.npy"
        raw_path = vec_dir / f"emotion_vectors_raw_layer{layer}.npy"
        if vec_path.exists():
            continue

        # 1. Load and average
        means = []
        for emotion in emotions:
            path = act_dir / f"{emotion}_layer{layer}.npy"
            if not path.exists():
                print(f"  Missing: {path}")
                break
            acts = np.load(path)
            means.append(acts.mean(axis=0))
        else:
            means = np.stack(means)

            # Save raw (centered but not deconfounded)
            global_mean = means.mean(axis=0)
            centered = means - global_mean
            np.save(raw_path, centered)

            # 3. Deconfound
            deconfounded = _deconfound(centered, act_dir, layer)
            np.save(vec_path, deconfounded)
            continue

        print(f"  Layer {layer}: SKIPPED (missing data)")


def compute_need_vectors(
    cfg: ModelConfig,
    needs: list[str],
    cluster_map: dict[str, str],
) -> None:
    """Compute need vectors: met, unmet, combined, direction.

    Saves 4 vector sets per layer:
    - need_met_vectors_layer{L}.npy (90, hidden_dim)
    - need_unmet_vectors_layer{L}.npy
    - need_combined_vectors_layer{L}.npy  (mean of met+unmet)
    - need_direction_vectors_layer{L}.npy (met - unmet)
    """
    act_dir = get_activations_dir(cfg.short_name)
    vec_dir = get_vectors_dir(cfg.short_name)
    vec_dir.mkdir(parents=True, exist_ok=True)

    (vec_dir / "need_labels.json").write_text(json.dumps(needs))
    (vec_dir / "need_cluster_labels.json").write_text(
        json.dumps({n: cluster_map[n] for n in needs})
    )

    for layer in range(cfg.n_layers):
        combined_path = vec_dir / f"need_combined_vectors_layer{layer}.npy"
        if combined_path.exists():
            continue

        met_means = []
        unmet_means = []

        for need in needs:
            safe = sanitize_need_name(need)
            met_path = act_dir / f"need_{safe}_met_layer{layer}.npy"
            unmet_path = act_dir / f"need_{safe}_unmet_layer{layer}.npy"

            if not met_path.exists() or not unmet_path.exists():
                print(f"  Missing need data for '{need}' layer {layer}")
                break

            met_acts = np.load(met_path)
            unmet_acts = np.load(unmet_path)
            met_means.append(met_acts.mean(axis=0))
            unmet_means.append(unmet_acts.mean(axis=0))
        else:
            met_arr = np.stack(met_means)
            unmet_arr = np.stack(unmet_means)
            combined = (met_arr + unmet_arr) / 2
            direction = met_arr - unmet_arr

            # Center all
            for name, arr in [("met", met_arr), ("unmet", unmet_arr), ("combined", combined), ("direction", direction)]:
                centered = arr - arr.mean(axis=0)
                deconfounded = _deconfound(centered, act_dir, layer)
                np.save(vec_dir / f"need_{name}_vectors_layer{layer}.npy", deconfounded)
            continue

        print(f"  Layer {layer}: SKIPPED")


def _deconfound(centered: np.ndarray, act_dir: Path, layer: int) -> np.ndarray:
    """Project out top neutral PCs from centered vectors."""
    neutral_path = act_dir / f"neutral_layer{layer}.npy"
    if not neutral_path.exists():
        return centered

    neutral_acts = np.load(neutral_path)
    neutral_centered = neutral_acts - neutral_acts.mean(axis=0)

    n_components = min(neutral_centered.shape[0], 20)
    pca = PCA(n_components=n_components)
    pca.fit(neutral_centered)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_remove = int(np.searchsorted(cumvar, NEUTRAL_VARIANCE_THRESHOLD) + 1)
    n_remove = min(n_remove, n_components)

    result = centered.copy()
    for comp in pca.components_[:n_remove]:
        comp = comp / np.linalg.norm(comp)
        projections = result @ comp
        result = result - np.outer(projections, comp)

    return result


def load_vectors(cfg: ModelConfig, kind: str = "emotion", layer: int = None) -> tuple[np.ndarray, list[str], dict]:
    """Load precomputed vectors and labels.

    Args:
        kind: "emotion", "need_met", "need_unmet", "need_combined", "need_direction"
        layer: layer index, defaults to cfg.analysis_layer

    Returns: (vectors, labels, cluster_map)
    """
    vec_dir = get_vectors_dir(cfg.short_name)
    layer = layer if layer is not None else cfg.analysis_layer

    if kind == "emotion":
        vectors = np.load(vec_dir / f"emotion_vectors_layer{layer}.npy")
        labels = json.loads((vec_dir / "emotion_labels.json").read_text())
        clusters = json.loads((vec_dir / "cluster_labels.json").read_text())
    else:
        vectors = np.load(vec_dir / f"{kind}_vectors_layer{layer}.npy")
        labels = json.loads((vec_dir / "need_labels.json").read_text())
        clusters = json.loads((vec_dir / "need_cluster_labels.json").read_text())

    return vectors, labels, clusters
