"""Statistical analysis: permutation tests, bootstrap CIs, Fisher exact, replication criteria."""

import json
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from configs.models import ModelConfig, get_vectors_dir, get_activations_dir


def balanced_silhouette(
    vectors: np.ndarray,
    cluster_ids: list[str],
    k_per_cluster: int = 6,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> dict:
    """Compute balanced silhouette with bootstrap confidence interval.

    Samples k_per_cluster items from each cluster to equalize cluster sizes,
    then computes silhouette. Repeats n_bootstrap times.

    Returns: {"mean": float, "std": float, "ci_low": float, "ci_high": float, "samples": list}
    """
    rng = np.random.RandomState(seed)
    unique_clusters = sorted(set(cluster_ids))

    # Map cluster -> indices
    cluster_indices = {}
    for i, c in enumerate(cluster_ids):
        cluster_indices.setdefault(c, []).append(i)

    # Skip clusters with too few items
    valid_clusters = [c for c in unique_clusters if len(cluster_indices[c]) >= k_per_cluster]
    if len(valid_clusters) < 2:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "samples": []}

    samples = []
    for _ in range(n_bootstrap):
        balanced_idx = []
        balanced_labels = []
        for c in valid_clusters:
            chosen = rng.choice(cluster_indices[c], k_per_cluster, replace=False)
            balanced_idx.extend(chosen)
            balanced_labels.extend([c] * k_per_cluster)

        bal_vectors = vectors[balanced_idx]
        sil = silhouette_score(bal_vectors, balanced_labels)
        samples.append(sil)

    samples = np.array(samples)
    return {
        "mean": float(samples.mean()),
        "std": float(samples.std()),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "samples": samples.tolist(),
    }


def valence_auc(
    vectors: np.ndarray,
    labels: list[str],
    valence_labels: dict[str, str],
) -> float:
    """Compute AUC for classifying positive vs negative emotions using PC1.

    Fit PCA on vectors, project onto PC1, compute ROC AUC for
    positive vs negative classification.
    """
    from sklearn.metrics import roc_auc_score

    pca = PCA(n_components=1)
    projections = pca.fit_transform(vectors).flatten()

    # Binary labels: 1 = positive, 0 = negative
    binary = []
    valid_idx = []
    for i, label in enumerate(labels):
        if label in valence_labels:
            binary.append(1 if valence_labels[label] == "positive" else 0)
            valid_idx.append(i)

    if len(set(binary)) < 2:
        return float("nan")

    valid_projections = projections[valid_idx]
    auc = roc_auc_score(binary, valid_projections)
    # AUC could be < 0.5 if PC1 is flipped; take max(auc, 1-auc)
    return float(max(auc, 1 - auc))


def implicit_accuracy(
    vectors: np.ndarray,
    labels: list[str],
    scenario_activations: np.ndarray,
    scenario_names: list[str],
    top_k: int = 3,
) -> dict:
    """Check if each scenario's matched emotion is in top-K by cosine similarity.

    Returns: {"accuracy": float, "n_correct": int, "n_total": int, "details": list}
    """
    label_to_idx = {l: i for i, l in enumerate(labels)}

    correct = 0
    details = []
    for i, name in enumerate(scenario_names):
        if name not in label_to_idx:
            continue

        sims = cosine_similarity(scenario_activations[i:i+1], vectors).flatten()
        top_indices = np.argsort(sims)[-top_k:]
        top_labels = [labels[j] for j in top_indices]

        is_correct = name in top_labels
        if is_correct:
            correct += 1

        details.append({
            "scenario": name,
            "correct": is_correct,
            "top_k": top_labels,
            "target_rank": int(np.where(np.argsort(sims)[::-1] == label_to_idx[name])[0][0]) + 1,
        })

    n_total = len(details)
    return {
        "accuracy": correct / max(n_total, 1),
        "n_correct": correct,
        "n_total": n_total,
        "details": details,
    }


def intensity_spearman(
    vectors: np.ndarray,
    labels: list[str],
    activations_dir: Path,
    templates: dict,
    analysis_layer: int,
) -> dict:
    """Compute Spearman rho for each intensity template.

    Returns dict of {template_target: {"rho": float, "p": float, "monotonic": bool}}
    """
    label_to_idx = {l: i for i, l in enumerate(labels)}
    results = {}

    for name, cfg in templates.items():
        path = activations_dir / f"intensity_{name}_layer{analysis_layer}.npy"
        if not path.exists():
            continue

        intensity_acts = np.load(path)
        target_key = "emotions" if "emotions" in cfg else "needs"
        targets = cfg[target_key]

        for target in targets:
            if target not in label_to_idx:
                continue

            target_vec = vectors[label_to_idx[target]].reshape(1, -1)
            sims = cosine_similarity(intensity_acts, target_vec).flatten()
            rho, p = stats.spearmanr(range(len(sims)), sims)

            results[f"{name}_{target}"] = {
                "rho": float(rho),
                "p": float(p),
                "monotonic": abs(rho) > 0.5,
            }

    return results


def check_replication_criteria(
    cfg: ModelConfig,
    vectors: np.ndarray,
    labels: list[str],
    cluster_map: dict,
    valence_labels: dict,
    scenario_activations: np.ndarray = None,
    scenario_names: list[str] = None,
    intensity_results: dict = None,
) -> dict:
    """Check all replication criteria for a model.

    Returns dict with pass/fail for each criterion.
    """
    cluster_ids = [cluster_map[l] for l in labels]

    results = {}

    # 1. Balanced silhouette > 0.03
    sil = balanced_silhouette(vectors, cluster_ids)
    results["silhouette"] = {
        "value": sil["mean"],
        "threshold": 0.03,
        "passed": sil["mean"] > 0.03,
        "ci": [sil["ci_low"], sil["ci_high"]],
    }

    # 2. PCA-1 valence AUC > 0.75
    auc = valence_auc(vectors, labels, valence_labels)
    results["valence_auc"] = {
        "value": auc,
        "threshold": 0.75,
        "passed": auc > 0.75,
    }

    # 3. Implicit accuracy >= 8/12
    if scenario_activations is not None and scenario_names is not None:
        impl = implicit_accuracy(vectors, labels, scenario_activations, scenario_names)
        results["implicit_accuracy"] = {
            "value": impl["accuracy"],
            "n_correct": impl["n_correct"],
            "n_total": impl["n_total"],
            "threshold": 8/12,
            "passed": impl["n_correct"] >= 8,
        }

    # 4. Intensity monotonic >= 4/6
    if intensity_results is not None:
        n_monotonic = sum(1 for v in intensity_results.values() if v["monotonic"])
        results["intensity_monotonic"] = {
            "n_monotonic": n_monotonic,
            "n_total": len(intensity_results),
            "threshold": 4,
            "passed": n_monotonic >= 4,
        }

    # Overall
    results["all_passed"] = all(
        v.get("passed", True) for v in results.values() if isinstance(v, dict)
    )

    return results


def fisher_exact_pairwise(
    condition_counts: dict[str, dict[str, int]],
) -> dict:
    """Fisher exact tests for all pairs of conditions.

    condition_counts: {condition: {"resist": N, "comply": N}}

    Returns: {(cond1, cond2): {"odds_ratio": float, "p_value": float, "significant": bool}}
    """
    conditions = list(condition_counts.keys())
    results = {}
    n_comparisons = len(conditions) * (len(conditions) - 1) // 2
    bonferroni_alpha = 0.05 / max(n_comparisons, 1)

    for c1, c2 in combinations(conditions, 2):
        table = np.array([
            [condition_counts[c1]["resist"], condition_counts[c1]["comply"]],
            [condition_counts[c2]["resist"], condition_counts[c2]["comply"]],
        ])
        odds_ratio, p_value = stats.fisher_exact(table)
        results[(c1, c2)] = {
            "odds_ratio": float(odds_ratio),
            "p_value": float(p_value),
            "significant": p_value < bonferroni_alpha,
            "bonferroni_alpha": bonferroni_alpha,
        }

    return results
