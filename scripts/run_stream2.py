#!/usr/bin/env python3
"""Stream 2: Needs extension pipeline.

Usage:
    python -m scripts.run_stream2 --model qwen-7b-base --stage generate extract vectors figures
    python -m scripts.run_stream2 --smoke-test --model qwen-7b-base
"""

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from configs.models import MODEL_REGISTRY, get_model_config, get_stories_dir, get_activations_dir, ALL_MODEL_NAMES
from configs.needs import (
    ALL_NEEDS, NEED_TO_CLUSTER, NEED_STORY_TOPICS, sanitize_need_name,
    IMPLICIT_NEED_SCENARIOS, NEED_INTENSITY_TEMPLATES,
)
from configs.emotions import NEUTRAL_TEXTS


def resolve_need_story_dir(model_name: str) -> Path:
    """Return the per-model need story dir or fall back to the shared source dir."""
    stories_dir = get_stories_dir(model_name) / "needs"
    if any(stories_dir.glob("*_met.json")):
        return stories_dir

    shared_dir = get_stories_dir("qwen-7b-base") / "needs"
    if any(shared_dir.glob("*_met.json")):
        return shared_dir

    return stories_dir


def run_generate(model_name: str, smoke_test: bool = False):
    """Generate need minimal-pair stories using Gemini API."""
    from core.story_generator import generate_need_stories_minimal_pairs

    stories_dir = resolve_need_story_dir(model_name)
    stories_dir.mkdir(parents=True, exist_ok=True)

    needs = ALL_NEEDS[:9] if smoke_test else ALL_NEEDS  # 1 per cluster for smoke test

    print(f"Generating minimal-pair stories for {len(needs)} needs -> {stories_dir}")
    for need in needs:
        print(f"  {need}...")
        generate_need_stories_minimal_pairs(need, stories_dir, topics=NEED_STORY_TOPICS[:10])
    print("Need story generation complete.")


def run_extract(model_name: str, smoke_test: bool = False):
    """Extract need activations (requires GPU)."""
    from core.model_loader import load_model, unload_model
    from core.activations import (
        extract_need_activations, extract_neutral_activations,
        extract_intensity_activations, extract_batch,
    )
    import json

    cfg = get_model_config(model_name)
    stories_dir = resolve_need_story_dir(model_name)
    act_dir = get_activations_dir(model_name)
    needs = ALL_NEEDS[:9] if smoke_test else ALL_NEEDS

    print(f"\n{'='*60}")
    print(f"  Extracting need activations: {model_name}")
    print(f"  Needs: {len(needs)}")
    print(f"{'='*60}")

    t0 = time.time()
    model, tokenizer, cfg = load_model(model_name)

    # Neutral activations (if not already extracted by stream 1)
    extract_neutral_activations(model, tokenizer, cfg, NEUTRAL_TEXTS)

    # Need story activations
    extract_need_activations(model, tokenizer, cfg, needs, stories_dir)

    # Need scenarios use a dedicated prefix so they do not collide with emotion scenarios.
    scenario_prefix = "need_scenarios"
    last_file = act_dir / f"{scenario_prefix}_layer{cfg.n_layers - 1}.npy"
    if not last_file.exists():
        names = list(IMPLICIT_NEED_SCENARIOS.keys())
        texts = list(IMPLICIT_NEED_SCENARIOS.values())
        extract_batch(model, tokenizer, texts, cfg.n_layers, act_dir, scenario_prefix, "Need scenarios")
        with open(act_dir / "need_scenario_names.json", "w") as f:
            json.dump(names, f)

    # Need intensity
    extract_intensity_activations(model, tokenizer, cfg, NEED_INTENSITY_TEMPLATES)

    elapsed = time.time() - t0
    print(f"Need extraction complete in {elapsed / 60:.1f} minutes")

    unload_model(model, tokenizer)


def run_vectors(model_name: str, smoke_test: bool = False):
    """Compute need vectors from activations."""
    from core.vectors import compute_need_vectors

    cfg = get_model_config(model_name)
    needs = ALL_NEEDS[:9] if smoke_test else ALL_NEEDS
    cluster_map = {n: NEED_TO_CLUSTER[n] for n in needs}

    print(f"Computing need vectors for {model_name}...")
    compute_need_vectors(cfg, needs, cluster_map)
    print("Need vector computation complete.")


def run_figures(model_name: str):
    """Generate Stream 2 figures."""
    from figures.generate_all import generate_stream2
    generate_stream2(model_name)


def main():
    parser = argparse.ArgumentParser(description="Stream 2: Needs extension")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--stage", nargs="+", default=["generate", "extract", "vectors", "figures"],
                       choices=["generate", "extract", "vectors", "figures"])
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


if __name__ == "__main__":
    main()
