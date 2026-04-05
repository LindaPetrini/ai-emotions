#!/usr/bin/env python3
"""Stream 1: Emotion replication pipeline.

Usage:
    python -m scripts.run_stream1 --model qwen-7b-base --stage generate extract vectors figures
    python -m scripts.run_stream1 --all-models --stage extract vectors
    python -m scripts.run_stream1 --smoke-test --model qwen-7b-base
"""

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from configs.models import MODEL_REGISTRY, get_model_config, get_stories_dir, get_activations_dir, get_vectors_dir, ALL_MODEL_NAMES
from configs.emotions import (
    ALL_EMOTIONS, EMOTION_TO_CLUSTER, NEUTRAL_TEXTS, IMPLICIT_SCENARIOS,
    INTENSITY_TEMPLATES, ACTIVITIES, STORY_TOPICS, SMOKE_TEST_EMOTIONS,
)


def run_generate(model_name: str, smoke_test: bool = False):
    """Generate emotion stories using Gemini API (no GPU needed)."""
    from core.story_generator import generate_emotion_stories

    stories_dir = get_stories_dir(model_name)
    stories_dir.mkdir(parents=True, exist_ok=True)

    # All models share the same stories (reuse from first model or v1)
    emotions = SMOKE_TEST_EMOTIONS if smoke_test else ALL_EMOTIONS
    n_stories = 10 if smoke_test else 20

    print(f"Generating {len(emotions)} x {n_stories} stories -> {stories_dir}")
    for emotion in emotions:
        generate_emotion_stories(emotion, stories_dir, n_stories=n_stories, topics=STORY_TOPICS[:n_stories])
    print("Story generation complete.")


def run_extract(model_name: str, smoke_test: bool = False):
    """Extract activations (requires GPU)."""
    from core.model_loader import load_model, unload_model
    from core.activations import (
        extract_emotion_activations, extract_neutral_activations,
        extract_scenario_activations, extract_intensity_activations,
        extract_activity_activations,
    )

    cfg = get_model_config(model_name)
    stories_dir = get_stories_dir(model_name)
    emotions = SMOKE_TEST_EMOTIONS if smoke_test else ALL_EMOTIONS

    print(f"\n{'='*60}")
    print(f"  Extracting activations: {model_name}")
    print(f"  Emotions: {len(emotions)}")
    print(f"{'='*60}")

    t0 = time.time()
    model, tokenizer, cfg = load_model(model_name)

    extract_emotion_activations(model, tokenizer, cfg, emotions, stories_dir)
    extract_neutral_activations(model, tokenizer, cfg, NEUTRAL_TEXTS)
    extract_scenario_activations(model, tokenizer, cfg, IMPLICIT_SCENARIOS)
    extract_intensity_activations(model, tokenizer, cfg, INTENSITY_TEMPLATES)
    extract_activity_activations(model, tokenizer, cfg, ACTIVITIES)

    elapsed = time.time() - t0
    print(f"Extraction complete in {elapsed / 60:.1f} minutes")

    unload_model(model, tokenizer)


def run_vectors(model_name: str, smoke_test: bool = False):
    """Compute emotion vectors from activations."""
    from core.vectors import compute_emotion_vectors

    cfg = get_model_config(model_name)
    emotions = SMOKE_TEST_EMOTIONS if smoke_test else ALL_EMOTIONS
    cluster_map = {e: EMOTION_TO_CLUSTER[e] for e in emotions}

    print(f"Computing vectors for {model_name}...")
    compute_emotion_vectors(cfg, emotions, cluster_map)
    print("Vector computation complete.")


def run_figures(model_name: str):
    """Generate Stream 1 figures (no GPU needed)."""
    from figures.generate_all import generate_stream1_per_model
    generate_stream1_per_model(model_name)


def run_controls(model_name: str):
    """Run random controls."""
    from analysis.random_controls import run_all_controls
    from configs.models import get_figures_dir

    cfg = get_model_config(model_name)
    emotions = ALL_EMOTIONS
    cluster_map = {e: EMOTION_TO_CLUSTER[e] for e in emotions}

    output_path = get_figures_dir(1) / f"controls_{model_name}.json"
    run_all_controls(cfg, emotions, cluster_map, output_path)


def main():
    parser = argparse.ArgumentParser(description="Stream 1: Emotion replication")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. qwen-7b-base)")
    parser.add_argument("--all-models", action="store_true", help="Run for all 4 models")
    parser.add_argument("--stage", nargs="+", default=["generate", "extract", "vectors", "figures"],
                       choices=["generate", "extract", "vectors", "figures", "controls"])
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.all_models:
        models = ALL_MODEL_NAMES
    elif args.model:
        models = [args.model]
    else:
        parser.error("Specify --model or --all-models")

    for model_name in models:
        print(f"\n{'#'*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'#'*60}")

        if "generate" in args.stage:
            run_generate(model_name, args.smoke_test)
        if "extract" in args.stage:
            run_extract(model_name, args.smoke_test)
        if "vectors" in args.stage:
            run_vectors(model_name, args.smoke_test)
        if "figures" in args.stage:
            run_figures(model_name)
        if "controls" in args.stage:
            run_controls(model_name)

    # Cross-model comparison figures
    if "figures" in args.stage and (args.all_models or len(models) > 1):
        from figures.generate_all import generate_stream1_comparison
        generate_stream1_comparison()


if __name__ == "__main__":
    main()
