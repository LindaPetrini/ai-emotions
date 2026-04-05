"""Base vs instruct model comparison analysis."""

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from configs.models import MODEL_REGISTRY, get_vectors_dir
from analysis.statistics import balanced_silhouette, valence_auc


def compare_pair(base_name: str, instruct_name: str) -> dict:
    """Compare base and instruct variants of the same architecture.

    Returns comprehensive comparison metrics.
    """
    results = {"base": base_name, "instruct": instruct_name}

    for name, role in [(base_name, "base"), (instruct_name, "instruct")]:
        cfg = MODEL_REGISTRY[name]
        vec_dir = get_vectors_dir(name)

        vec_path = vec_dir / f"emotion_vectors_layer{cfg.analysis_layer}.npy"
        if not vec_path.exists():
            results[role] = {"error": "vectors not found"}
            continue

        vectors = np.load(vec_path)
        labels = json.loads((vec_dir / "emotion_labels.json").read_text())
        cluster_map = json.loads((vec_dir / "cluster_labels.json").read_text())
        cluster_ids = [cluster_map[l] for l in labels]

        sil = balanced_silhouette(vectors, cluster_ids)

        # PCA variance explained
        pca = PCA(n_components=10)
        pca.fit(vectors)

        results[role] = {
            "silhouette_mean": sil["mean"],
            "silhouette_ci": [sil["ci_low"], sil["ci_high"]],
            "pca_var_explained": pca.explained_variance_ratio_.tolist(),
            "n_vectors": vectors.shape[0],
            "hidden_dim": vectors.shape[1],
        }

    # Cross-model RSA
    base_cfg = MODEL_REGISTRY[base_name]
    inst_cfg = MODEL_REGISTRY[instruct_name]
    base_vec_path = get_vectors_dir(base_name) / f"emotion_vectors_layer{base_cfg.analysis_layer}.npy"
    inst_vec_path = get_vectors_dir(instruct_name) / f"emotion_vectors_layer{inst_cfg.analysis_layer}.npy"

    if base_vec_path.exists() and inst_vec_path.exists():
        base_vecs = np.load(base_vec_path)
        inst_vecs = np.load(inst_vec_path)

        base_sim = cosine_similarity(base_vecs)
        inst_sim = cosine_similarity(inst_vecs)

        triu = np.triu_indices_from(base_sim, k=1)
        rho, p = stats.spearmanr(base_sim[triu], inst_sim[triu])

        results["rsa"] = {"spearman_rho": float(rho), "p_value": float(p)}

        # Per-emotion cosine similarity between base and instruct
        per_emotion_cos = np.array([
            cosine_similarity(base_vecs[i:i+1], inst_vecs[i:i+1])[0, 0]
            for i in range(base_vecs.shape[0])
        ])
        results["per_emotion_cosine"] = {
            "mean": float(per_emotion_cos.mean()),
            "std": float(per_emotion_cos.std()),
            "min": float(per_emotion_cos.min()),
            "max": float(per_emotion_cos.max()),
        }

    return results


def compare_all_pairs() -> dict:
    """Compare all base-instruct pairs."""
    pairs = [
        ("qwen-7b-base", "qwen-7b-inst"),
        ("llama-8b-base", "llama-8b-inst"),
    ]

    results = {}
    for base, instruct in pairs:
        arch = base.split("-")[0]
        results[arch] = compare_pair(base, instruct)

    return results
