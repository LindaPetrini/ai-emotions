"""Random vector controls — mandatory for every quantitative claim.

Controls:
1. Random unit vectors of same norm/dimensionality
2. Shuffled cluster labels
3. Noun-vector control (semantic coherence)
4. Random story-pair differences (for direction vectors)
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from configs.models import ModelConfig, get_vectors_dir, get_activations_dir
from analysis.statistics import balanced_silhouette


def random_vector_silhouette(
    n_vectors: int,
    hidden_dim: int,
    cluster_ids: list[str],
    n_repeats: int = 100,
    k_per_cluster: int = 6,
    seed: int = 42,
) -> dict:
    """Compute silhouette for random unit vectors with same cluster structure.

    Returns: {"mean": float, "std": float, "ci_low": float, "ci_high": float}
    """
    rng = np.random.RandomState(seed)
    samples = []

    for i in range(n_repeats):
        random_vecs = rng.randn(n_vectors, hidden_dim)
        random_vecs = random_vecs / np.linalg.norm(random_vecs, axis=1, keepdims=True)

        sil = balanced_silhouette(random_vecs, cluster_ids, k_per_cluster, n_bootstrap=1, seed=seed + i)
        samples.append(sil["mean"])

    samples = np.array(samples)
    return {
        "mean": float(np.nanmean(samples)),
        "std": float(np.nanstd(samples)),
        "ci_low": float(np.nanpercentile(samples, 2.5)),
        "ci_high": float(np.nanpercentile(samples, 97.5)),
    }


def shuffled_label_silhouette(
    vectors: np.ndarray,
    cluster_ids: list[str],
    n_repeats: int = 1000,
    k_per_cluster: int = 6,
    seed: int = 42,
) -> dict:
    """Compute silhouette with shuffled cluster labels (permutation test).

    Returns: {"mean": float, "std": float, "p_value": float, "real_value": float}
    """
    rng = np.random.RandomState(seed)

    # Real silhouette
    real = balanced_silhouette(vectors, cluster_ids, k_per_cluster, n_bootstrap=1, seed=seed)
    real_val = real["mean"]

    samples = []
    for _ in range(n_repeats):
        shuffled = list(cluster_ids)
        rng.shuffle(shuffled)
        sil = balanced_silhouette(vectors, shuffled, k_per_cluster, n_bootstrap=1, seed=seed)
        samples.append(sil["mean"])

    samples = np.array(samples)
    p_value = float(np.mean(samples >= real_val))

    return {
        "mean": float(np.nanmean(samples)),
        "std": float(np.nanstd(samples)),
        "p_value": p_value,
        "real_value": real_val,
    }


def noun_vector_control(
    cfg: ModelConfig,
    noun_list: list[str],
    noun_clusters: dict[str, str],
    layer: int = None,
) -> dict:
    """Compute silhouette for noun vectors grouped into arbitrary clusters.

    THIS IS THE CRITICAL CONTROL: if noun silhouette >= emotion silhouette,
    the clustering is an artifact of the mean-difference pipeline.

    Args:
        noun_list: list of noun names
        noun_clusters: {noun: cluster_name} arbitrary grouping
        layer: analysis layer

    Returns: {"noun_silhouette": float, "is_artifact": bool, "details": dict}
    """
    layer = layer or cfg.analysis_layer
    act_dir = get_activations_dir(cfg.short_name)

    # Load noun activations and compute mean vectors
    noun_vectors = []
    valid_nouns = []
    valid_clusters = []

    for noun in noun_list:
        path = act_dir / f"{noun}_layer{layer}.npy"
        if not path.exists():
            continue
        acts = np.load(path)
        noun_vectors.append(acts.mean(axis=0))
        valid_nouns.append(noun)
        valid_clusters.append(noun_clusters.get(noun, "unknown"))

    if len(valid_nouns) < 10:
        return {"noun_silhouette": float("nan"), "is_artifact": False, "details": {"error": "too few nouns"}}

    noun_vectors = np.stack(noun_vectors)

    # Center and deconfound same as emotions
    noun_vectors = noun_vectors - noun_vectors.mean(axis=0)

    sil = balanced_silhouette(noun_vectors, valid_clusters, k_per_cluster=6)

    # Compare with emotion silhouette
    vec_dir = get_vectors_dir(cfg.short_name)
    emo_path = vec_dir / f"emotion_vectors_layer{layer}.npy"
    if emo_path.exists():
        emo_vectors = np.load(emo_path)
        emo_labels = json.loads((vec_dir / "emotion_labels.json").read_text())
        emo_clusters = json.loads((vec_dir / "cluster_labels.json").read_text())
        emo_cluster_ids = [emo_clusters[l] for l in emo_labels]
        emo_sil = balanced_silhouette(emo_vectors, emo_cluster_ids)

        is_artifact = sil["mean"] >= emo_sil["mean"]
    else:
        emo_sil = None
        is_artifact = False

    return {
        "noun_silhouette": sil,
        "emotion_silhouette": emo_sil,
        "is_artifact": is_artifact,
        "n_nouns": len(valid_nouns),
    }


def random_direction_control(
    met_activations_dir: Path,
    unmet_activations_dir: Path,
    needs: list[str],
    cluster_ids: list[str],
    n_layers: int,
    analysis_layer: int,
    n_repeats: int = 100,
    seed: int = 42,
) -> dict:
    """Control for direction vectors: randomly shuffle which stories are 'met' vs 'unmet'.

    If shuffled direction vectors cluster as well as real ones,
    the met/unmet distinction is not meaningful.
    """
    from configs.needs import sanitize_need_name

    rng = np.random.RandomState(seed)
    layer = analysis_layer

    # Load all story activations
    all_stories = {}  # need -> (met_acts, unmet_acts)
    for need in needs:
        safe = sanitize_need_name(need)
        met_path = met_activations_dir / f"need_{safe}_met_layer{layer}.npy"
        unmet_path = met_activations_dir / f"need_{safe}_unmet_layer{layer}.npy"

        if met_path.exists() and unmet_path.exists():
            all_stories[need] = (np.load(met_path), np.load(unmet_path))

    if len(all_stories) < 10:
        return {"error": "too few needs with both conditions"}

    valid_needs = list(all_stories.keys())
    valid_clusters = [cluster_ids[i] for i, n in enumerate(needs) if n in valid_needs]

    # Real direction vectors
    real_directions = []
    for need in valid_needs:
        met, unmet = all_stories[need]
        real_directions.append(met.mean(axis=0) - unmet.mean(axis=0))
    real_directions = np.stack(real_directions)

    real_sil = balanced_silhouette(real_directions, valid_clusters, k_per_cluster=3)

    # Shuffled controls
    shuffled_sils = []
    for _ in range(n_repeats):
        shuffled_directions = []
        for need in valid_needs:
            met, unmet = all_stories[need]
            combined = np.vstack([met, unmet])
            n_met = met.shape[0]
            perm = rng.permutation(combined.shape[0])
            shuffled_met = combined[perm[:n_met]]
            shuffled_unmet = combined[perm[n_met:]]
            shuffled_directions.append(shuffled_met.mean(axis=0) - shuffled_unmet.mean(axis=0))

        shuffled_dirs = np.stack(shuffled_directions)
        sil = balanced_silhouette(shuffled_dirs, valid_clusters, k_per_cluster=3, n_bootstrap=1)
        shuffled_sils.append(sil["mean"])

    shuffled_sils = np.array(shuffled_sils)
    p_value = float(np.mean(shuffled_sils >= real_sil["mean"]))

    return {
        "real_silhouette": real_sil,
        "shuffled_mean": float(np.nanmean(shuffled_sils)),
        "shuffled_std": float(np.nanstd(shuffled_sils)),
        "p_value": p_value,
    }


def run_all_controls(cfg: ModelConfig, emotions: list[str], cluster_map: dict, output_path: Path) -> dict:
    """Run all random controls for a model and save results."""
    vec_dir = get_vectors_dir(cfg.short_name)

    vectors = np.load(vec_dir / f"emotion_vectors_layer{cfg.analysis_layer}.npy")
    cluster_ids = [cluster_map[e] for e in emotions]

    results = {}

    # 1. Random vectors
    print("  Running random vector control...")
    results["random_vectors"] = random_vector_silhouette(
        len(emotions), cfg.hidden_dim, cluster_ids
    )

    # 2. Shuffled labels
    print("  Running shuffled label control...")
    results["shuffled_labels"] = shuffled_label_silhouette(vectors, cluster_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Controls saved to {output_path}")
    return results
