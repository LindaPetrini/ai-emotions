#!/usr/bin/env python3
"""Analyze LLM ratings from steering evaluation.

Computes:
  - Mean emotion/coherence/relevance scores by condition (steered vs random vs baseline)
  - Paired comparisons (steered vs baseline, steered vs random)
  - Breakdown by model and emotion
  - Statistical tests (paired t-tests and Mann-Whitney)

Saves results to eval_results.json
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
LLM_RATINGS_PATH = SCRIPT_DIR / "llm_ratings.csv"
RESULTS_PATH = SCRIPT_DIR / "eval_results.json"


def load_ratings(path: Path) -> list:
    """Load rated items from CSV."""
    items = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("emotion_score") and row.get("coherence_score") and row.get("relevance_score"):
                row["emotion_score"] = int(row["emotion_score"])
                row["coherence_score"] = int(row["coherence_score"])
                row["relevance_score"] = int(row["relevance_score"])
                try:
                    row["alpha"] = float(row["alpha"]) if row["alpha"] else 0.0
                except ValueError:
                    row["alpha"] = row["alpha"]  # keep as string (e.g. "prompt")
                items.append(row)
    return items


def compute_agreement(human_path: Path, llm_path: Path) -> dict:
    """Compute human-LLM agreement on shared items."""
    human = {}
    with open(human_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("emotion_score") and row.get("coherence_score") and row.get("relevance_score"):
                human[row["id"]] = {
                    "emotion_score": int(row["emotion_score"]),
                    "coherence_score": int(row["coherence_score"]),
                    "relevance_score": int(row["relevance_score"]),
                }

    llm = {}
    with open(llm_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("emotion_score") and row.get("coherence_score") and row.get("relevance_score"):
                llm[row["id"]] = {
                    "emotion_score": int(row["emotion_score"]),
                    "coherence_score": int(row["coherence_score"]),
                    "relevance_score": int(row["relevance_score"]),
                }

    shared_ids = sorted(set(human.keys()) & set(llm.keys()))
    if not shared_ids:
        return {"n": 0}

    agreement = {"n": len(shared_ids), "per_dimension": {}}
    for dim in ("emotion_score", "coherence_score", "relevance_score"):
        h = np.array([human[id][dim] for id in shared_ids])
        l = np.array([llm[id][dim] for id in shared_ids])

        r, p = stats.pearsonr(h, l)
        rho, p_rho = stats.spearmanr(h, l)
        mad = float(np.mean(np.abs(h - l)))
        within1 = float(np.mean(np.abs(h - l) <= 1))

        agreement["per_dimension"][dim] = {
            "pearson_r": round(r, 3),
            "pearson_p": round(p, 4),
            "spearman_rho": round(rho, 3),
            "spearman_p": round(p_rho, 4),
            "mean_abs_diff": round(mad, 2),
            "within_1_agreement": round(within1, 3),
        }

    return agreement


def main():
    items = load_ratings(LLM_RATINGS_PATH)
    print(f"Loaded {len(items)} rated items")

    if len(items) < 50:
        print(f"WARNING: Only {len(items)} items rated. Results may not be meaningful.")

    # Group by condition
    by_condition = defaultdict(list)
    for item in items:
        by_condition[item["condition"]].append(item)

    for cond_name in ["steered", "random", "baseline", "prompt_steered"]:
        print(f"  {cond_name}: {len(by_condition.get(cond_name, []))}")

    results = {"n_total": len(items), "n_by_condition": {k: len(v) for k, v in by_condition.items()}}

    # === Overall means by condition ===
    dims = ["emotion_score", "coherence_score", "relevance_score"]
    condition_means = {}
    for cond, cond_items in by_condition.items():
        means = {}
        for dim in dims:
            vals = [item[dim] for item in cond_items]
            means[dim] = {
                "mean": round(np.mean(vals), 3),
                "std": round(np.std(vals, ddof=1), 3),
                "median": float(np.median(vals)),
                "n": len(vals),
            }
        condition_means[cond] = means
    results["condition_means"] = condition_means

    # === Statistical tests: steered vs baseline, steered vs random ===
    comparisons = {}

    comparison_pairs = [
        ("steered_vs_baseline", ("steered", "baseline")),
        ("steered_vs_random", ("steered", "random")),
        ("random_vs_baseline", ("random", "baseline")),
    ]
    # Add prompt_steered comparisons if that condition exists
    if "prompt_steered" in by_condition and len(by_condition["prompt_steered"]) > 0:
        comparison_pairs.extend([
            ("prompt_steered_vs_baseline", ("prompt_steered", "baseline")),
            ("prompt_steered_vs_steered", ("prompt_steered", "steered")),
        ])
    for comparison_name, (cond_a, cond_b) in comparison_pairs:
        comp = {}
        for dim in dims:
            a_vals = np.array([item[dim] for item in by_condition[cond_a]])
            b_vals = np.array([item[dim] for item in by_condition[cond_b]])

            # Mann-Whitney U (independent samples, non-parametric)
            u_stat, u_p = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")

            # Effect size (rank-biserial correlation)
            n1, n2 = len(a_vals), len(b_vals)
            rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

            # Also compute Welch's t-test
            t_stat, t_p = stats.ttest_ind(a_vals, b_vals, equal_var=False)

            comp[dim] = {
                f"{cond_a}_mean": round(float(np.mean(a_vals)), 3),
                f"{cond_b}_mean": round(float(np.mean(b_vals)), 3),
                "diff": round(float(np.mean(a_vals) - np.mean(b_vals)), 3),
                "mann_whitney_U": float(u_stat),
                "mann_whitney_p": round(float(u_p), 4),
                "rank_biserial_r": round(rank_biserial, 3),
                "welch_t": round(float(t_stat), 3),
                "welch_p": round(float(t_p), 4),
            }
        comparisons[comparison_name] = comp
    results["comparisons"] = comparisons

    # === Breakdown by model ===
    by_model = defaultdict(lambda: defaultdict(list))
    for item in items:
        by_model[item["model"]][item["condition"]].append(item)

    model_results = {}
    for model in sorted(by_model.keys()):
        model_data = by_model[model]
        model_res = {}
        for cond in ["steered", "random", "baseline", "prompt_steered"]:
            if cond in model_data:
                model_res[cond] = {}
                for dim in dims:
                    vals = [item[dim] for item in model_data[cond]]
                    model_res[cond][dim] = {
                        "mean": round(np.mean(vals), 3),
                        "std": round(np.std(vals, ddof=1), 3) if len(vals) > 1 else 0,
                        "n": len(vals),
                    }
        # Steered vs baseline within this model
        if "steered" in model_data and "baseline" in model_data:
            s_emo = [item["emotion_score"] for item in model_data["steered"]]
            b_emo = [item["emotion_score"] for item in model_data["baseline"]]
            if len(s_emo) > 1 and len(b_emo) > 1:
                u, p = stats.mannwhitneyu(s_emo, b_emo, alternative="two-sided")
                model_res["steered_vs_baseline_emotion"] = {
                    "steered_mean": round(np.mean(s_emo), 3),
                    "baseline_mean": round(np.mean(b_emo), 3),
                    "mann_whitney_p": round(float(p), 4),
                }
        model_results[model] = model_res
    results["by_model"] = model_results

    # === Breakdown by target emotion ===
    by_emotion = defaultdict(lambda: defaultdict(list))
    for item in items:
        by_emotion[item["target_emotion"]][item["condition"]].append(item)

    emotion_results = {}
    for emotion in sorted(by_emotion.keys()):
        emotion_data = by_emotion[emotion]
        emo_res = {}
        for cond in ["steered", "random", "baseline", "prompt_steered"]:
            if cond in emotion_data:
                vals = [item["emotion_score"] for item in emotion_data[cond]]
                emo_res[cond] = {
                    "emotion_score_mean": round(np.mean(vals), 3),
                    "emotion_score_std": round(np.std(vals, ddof=1), 3) if len(vals) > 1 else 0,
                    "n": len(vals),
                }
        # Steered vs baseline
        if "steered" in emotion_data and "baseline" in emotion_data:
            s_vals = [item["emotion_score"] for item in emotion_data["steered"]]
            b_vals = [item["emotion_score"] for item in emotion_data["baseline"]]
            if len(s_vals) > 1 and len(b_vals) > 1:
                u, p = stats.mannwhitneyu(s_vals, b_vals, alternative="two-sided")
                emo_res["steered_vs_baseline"] = {
                    "mann_whitney_p": round(float(p), 4),
                    "diff": round(np.mean(s_vals) - np.mean(b_vals), 3),
                }
        emotion_results[emotion] = emo_res
    results["by_emotion"] = emotion_results

    # === Breakdown by alpha (within steered condition) ===
    steered_by_alpha = defaultdict(list)
    for item in by_condition["steered"]:
        steered_by_alpha[item["alpha"]].append(item)

    alpha_results = {}
    for alpha in sorted(steered_by_alpha.keys()):
        vals = [item["emotion_score"] for item in steered_by_alpha[alpha]]
        alpha_results[str(alpha)] = {
            "emotion_score_mean": round(np.mean(vals), 3),
            "emotion_score_std": round(np.std(vals, ddof=1), 3) if len(vals) > 1 else 0,
            "n": len(vals),
        }
    results["steered_by_alpha"] = alpha_results

    # === Human-LLM agreement ===
    human_path = SCRIPT_DIR / "ratings.csv"
    if human_path.exists():
        agreement = compute_agreement(human_path, LLM_RATINGS_PATH)
        results["human_llm_agreement"] = agreement

    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nMean emotion_score by condition:")
    for cond in ["steered", "random", "baseline", "prompt_steered"]:
        m = condition_means.get(cond, {}).get("emotion_score", {})
        if m:
            print(f"  {cond:15s}: {m.get('mean', 'N/A'):.3f} +/- {m.get('std', 0):.3f} (n={m.get('n', 0)})")

    print(f"\nSteered vs Baseline (emotion_score):")
    c = comparisons["steered_vs_baseline"]["emotion_score"]
    print(f"  diff = {c['diff']:.3f}, Mann-Whitney p = {c['mann_whitney_p']:.4f}, r = {c['rank_biserial_r']:.3f}")

    print(f"\nSteered vs Random (emotion_score):")
    c = comparisons["steered_vs_random"]["emotion_score"]
    print(f"  diff = {c['diff']:.3f}, Mann-Whitney p = {c['mann_whitney_p']:.4f}, r = {c['rank_biserial_r']:.3f}")

    if "prompt_steered_vs_baseline" in comparisons:
        print(f"\nPrompt-steered vs Baseline (emotion_score):")
        c = comparisons["prompt_steered_vs_baseline"]["emotion_score"]
        print(f"  diff = {c['diff']:.3f}, Mann-Whitney p = {c['mann_whitney_p']:.4f}, r = {c['rank_biserial_r']:.3f}")

        print(f"\nPrompt-steered vs Steered (emotion_score):")
        c = comparisons["prompt_steered_vs_steered"]["emotion_score"]
        print(f"  diff = {c['diff']:.3f}, Mann-Whitney p = {c['mann_whitney_p']:.4f}, r = {c['rank_biserial_r']:.3f}")

    print(f"\nEmotion score by alpha (steered only):")
    for alpha, data in sorted(alpha_results.items(), key=lambda x: float(x[0])):
        print(f"  alpha={alpha}: {data['emotion_score_mean']:.3f} +/- {data['emotion_score_std']:.3f} (n={data['n']})")

    print(f"\nBy model (emotion_score, steered vs baseline):")
    for model in sorted(model_results.keys()):
        mr = model_results[model]
        if "steered_vs_baseline_emotion" in mr:
            sv = mr["steered_vs_baseline_emotion"]
            print(f"  {model:15s}: steered={sv['steered_mean']:.2f} baseline={sv['baseline_mean']:.2f} p={sv['mann_whitney_p']:.4f}")

    print(f"\nBy emotion (emotion_score, steered vs baseline):")
    for emotion in sorted(emotion_results.keys()):
        er = emotion_results[emotion]
        if "steered_vs_baseline" in er:
            print(f"  {emotion:10s}: diff={er['steered_vs_baseline']['diff']:.3f} p={er['steered_vs_baseline']['mann_whitney_p']:.4f}")

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
