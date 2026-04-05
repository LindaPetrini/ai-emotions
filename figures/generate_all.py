"""Generate all figures for the ai-emotions-v2 project."""

import argparse
import json
from pathlib import Path

from configs.models import MODEL_REGISTRY, get_model_config, get_vectors_dir, get_activations_dir, get_figures_dir
from configs.emotions import (
    ALL_EMOTIONS, EMOTION_CLUSTERS, EMOTION_TO_CLUSTER,
    IMPLICIT_SCENARIOS, INTENSITY_TEMPLATES, ACTIVITIES,
)
from configs.needs import (
    ALL_NEEDS, NEED_CLUSTERS, NEED_TO_CLUSTER, NEED_IS_LLM,
    IMPLICIT_NEED_SCENARIOS, NEED_INTENSITY_TEMPLATES,
)
from core.vectors import load_vectors

from figures.common import EMOTION_CLUSTER_COLORS, NEED_CLUSTER_COLORS
from figures.fig_heatmap import plot_implicit_heatmap
from figures.fig_intensity import plot_intensity_curves
from figures.fig_preference import plot_preference_correlation
from figures.fig_cosine_sim import plot_cosine_similarity
from figures.fig_umap import plot_umap
from figures.fig_pca import plot_pca
from figures.fig_model_comparison import (
    plot_layer_emergence, plot_cross_architecture_rsa,
    plot_base_vs_instruct_pca, plot_probing_summary,
)
from figures.fig_needs_cross import (
    plot_need_x_emotion, plot_need_emotion_alignment,
    plot_met_unmet_pca,
)
from figures.fig_emotion_residual import (
    plot_emotion_residual_clustering, plot_direction_vs_valence,
)


def generate_stream1_per_model(model_name: str):
    """Generate 7 main figures for one model."""
    cfg = get_model_config(model_name)
    act_dir = get_activations_dir(model_name)
    fig_dir = get_figures_dir(1)

    print(f"\n=== Stream 1 figures for {model_name} ===")

    try:
        vectors, labels, clusters = load_vectors(cfg, "emotion")
    except FileNotFoundError:
        print(f"  No vectors for {model_name}, skipping")
        return

    # 1. Implicit heatmap
    scenario_path = act_dir / f"scenarios_layer{cfg.analysis_layer}.npy"
    names_path = act_dir / "scenario_names.json"
    if scenario_path.exists() and names_path.exists():
        scenario_acts = __import__('numpy').load(scenario_path)
        scenario_names = json.loads(names_path.read_text())
        plot_implicit_heatmap(
            vectors, labels, scenario_acts, scenario_names,
            f"Implicit Emotion Detection — {model_name}",
            fig_dir / f"heatmap_{model_name}.pdf",
        )

    # 2. Intensity curves
    plot_intensity_curves(
        vectors, labels, act_dir, INTENSITY_TEMPLATES,
        cfg.n_layers, cfg.analysis_layer,
        model_name, fig_dir / f"intensity_{model_name}.pdf",
    )

    # 3. Preference correlation
    plot_preference_correlation(
        vectors, labels, act_dir, ACTIVITIES,
        cfg.analysis_layer, f"Preference — {model_name}",
        fig_dir / f"preference_{model_name}.pdf",
    )

    # 4. Cosine similarity
    plot_cosine_similarity(
        vectors, labels, clusters, EMOTION_CLUSTER_COLORS,
        f"Emotion Cosine Similarity — {model_name}",
        fig_dir / f"cosine_sim_{model_name}.pdf",
    )

    # 5. UMAP
    plot_umap(
        vectors, labels, clusters, EMOTION_CLUSTER_COLORS,
        f"Emotion UMAP — {model_name}",
        fig_dir / f"umap_{model_name}.pdf",
    )

    # 6. PCA
    plot_pca(
        vectors, labels, clusters, EMOTION_CLUSTER_COLORS,
        f"Emotion PCA — {model_name}",
        fig_dir / f"pca_{model_name}.pdf",
    )

    # 7. Logit lens — needs model, skip in offline mode
    print(f"  Logit lens requires loaded model, skipping in figure-only mode")


def generate_stream1_comparison():
    """Generate cross-model comparison figures."""
    fig_dir = get_figures_dir(1)
    model_names = list(MODEL_REGISTRY.keys())

    print("\n=== Stream 1 cross-model figures ===")

    # 8. Layer emergence
    plot_layer_emergence(model_names, "Layer Emergence — All Models", fig_dir / "layer_emergence.pdf")

    # 9. Base vs instruct PCA
    plot_base_vs_instruct_pca(
        "qwen-7b-base", "qwen-7b-inst",
        "Base vs Instruct PCA — Qwen 7B",
        fig_dir / "base_vs_instruct_qwen.pdf",
    )
    plot_base_vs_instruct_pca(
        "llama-8b-base", "llama-8b-inst",
        "Base vs Instruct PCA — Llama 8B",
        fig_dir / "base_vs_instruct_llama.pdf",
    )

    # 11. Cross-architecture RSA
    plot_cross_architecture_rsa(model_names, "Cross-Architecture RSA", fig_dir / "cross_rsa.pdf")


def generate_stream2(model_name: str):
    """Generate need figures for one model."""
    cfg = get_model_config(model_name)
    fig_dir = get_figures_dir(2)

    print(f"\n=== Stream 2 figures for {model_name} ===")

    try:
        emo_vectors, emo_labels, emo_clusters = load_vectors(cfg, "emotion")
        combined_vectors, need_labels, need_clusters = load_vectors(cfg, "need_combined")
    except FileNotFoundError as e:
        print(f"  Missing data: {e}")
        return

    # 1. Need cosine sim
    plot_cosine_similarity(
        combined_vectors, need_labels, need_clusters, NEED_CLUSTER_COLORS,
        f"Need Cosine Similarity — {model_name}",
        fig_dir / f"need_cosine_sim_{model_name}.pdf",
    )

    # 2. Need PCA
    plot_pca(
        combined_vectors, need_labels, need_clusters, NEED_CLUSTER_COLORS,
        f"Need PCA — {model_name}",
        fig_dir / f"need_pca_{model_name}.pdf",
    )

    # 3. Need UMAP
    plot_umap(
        combined_vectors, need_labels, need_clusters, NEED_CLUSTER_COLORS,
        f"Need UMAP — {model_name}",
        fig_dir / f"need_umap_{model_name}.pdf",
    )

    # 7. Need x Emotion cross-similarity
    plot_need_x_emotion(
        emo_vectors, emo_labels, combined_vectors, need_labels,
        emo_clusters, need_clusters,
        f"Need x Emotion — {model_name}",
        fig_dir / f"need_x_emotion_{model_name}.pdf",
    )

    # 8. Emotion-residual clustering (THE KEY FIGURE)
    plot_emotion_residual_clustering(
        emo_vectors, combined_vectors, need_labels, need_clusters,
        f"Emotion-Residual Clustering — {model_name}",
        fig_dir / f"emotion_residual_{model_name}.pdf",
    )

    # 9. Need-emotion alignment
    plot_need_emotion_alignment(
        emo_vectors, emo_labels, combined_vectors, need_labels,
        need_clusters,
        f"Need-Emotion Alignment — {model_name}",
        fig_dir / f"need_emotion_alignment_{model_name}.pdf",
    )

    # 10. Met/unmet PCA
    try:
        met_vectors, _, _ = load_vectors(cfg, "need_met")
        unmet_vectors, _, _ = load_vectors(cfg, "need_unmet")
        plot_met_unmet_pca(
            met_vectors, unmet_vectors, need_labels, need_clusters,
            f"Met vs Unmet — {model_name}",
            fig_dir / f"met_unmet_pca_{model_name}.pdf",
        )

        # 11. Direction vs valence
        direction_vectors, _, _ = load_vectors(cfg, "need_direction")
        plot_direction_vs_valence(
            emo_vectors, direction_vectors, need_labels, need_clusters,
            f"Direction vs Valence — {model_name}",
            fig_dir / f"direction_vs_valence_{model_name}.pdf",
        )
    except FileNotFoundError:
        print("  Missing met/unmet vectors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Single model to process")
    parser.add_argument("--stream", type=int, default=None, help="Stream number (1, 2, or 3)")
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODEL_REGISTRY.keys())

    if args.stream is None or args.stream == 1:
        for model_name in models:
            generate_stream1_per_model(model_name)
        generate_stream1_comparison()

    if args.stream is None or args.stream == 2:
        for model_name in models:
            generate_stream2(model_name)


if __name__ == "__main__":
    main()
