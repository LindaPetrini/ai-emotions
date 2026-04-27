#!/usr/bin/env python3
"""Run the complete ai-emotions pipeline.

Usage:
    python -m scripts.run_all --phase 1    # Setup + stories
    python -m scripts.run_all --phase 2    # Extraction + vectors (GPU)
    python -m scripts.run_all --phase 3    # Stream 1 figures + controls
    python -m scripts.run_all --phase 4    # Stream 2 needs
    python -m scripts.run_all --phase 5    # Stream 3 steering + shutdown
    python -m scripts.run_all --phase 6    # Final analysis
    python -m scripts.run_all --smoke-test # Quick validation run
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from configs.models import ALL_MODEL_NAMES


def phase1_setup(smoke_test=False):
    """Phase 1: Generate stories (LOCAL, no GPU)."""
    print("\n" + "="*60)
    print("  PHASE 1: Story generation")
    print("="*60)

    from scripts.run_stream1 import run_generate
    from scripts.run_stream2 import run_generate as run_generate_needs

    # Use first model's stories dir (all models share stories)
    model = "qwen-7b-base"
    run_generate(model, smoke_test)
    run_generate_needs(model, smoke_test)

    # Generate noun control stories
    if not smoke_test:
        from core.story_generator import generate_noun_control_stories, CONTROL_NOUNS
        from configs.models import get_stories_dir
        noun_dir = get_stories_dir(model) / "nouns"
        noun_dir.mkdir(parents=True, exist_ok=True)
        generate_noun_control_stories(CONTROL_NOUNS, noun_dir)


def phase2_extract(smoke_test=False):
    """Phase 2: Extract activations + compute vectors (GPU)."""
    print("\n" + "="*60)
    print("  PHASE 2: Activation extraction + vector computation")
    print("="*60)

    from scripts.run_stream1 import run_extract, run_vectors
    from scripts.run_stream2 import run_extract as extract_needs, run_vectors as compute_needs

    models = ALL_MODEL_NAMES
    for model in models:
        run_extract(model, smoke_test)
        run_vectors(model, smoke_test)
        extract_needs(model, smoke_test)
        compute_needs(model, smoke_test)


def phase3_figures(smoke_test=False):
    """Phase 3: Stream 1 figures + controls (LOCAL)."""
    print("\n" + "="*60)
    print("  PHASE 3: Stream 1 figures + controls")
    print("="*60)

    from scripts.run_stream1 import run_figures, run_controls
    from figures.generate_all import generate_stream1_comparison

    for model in ALL_MODEL_NAMES:
        run_figures(model)
        if not smoke_test:
            run_controls(model)

    generate_stream1_comparison()

    # GO/NO-GO: Check noun control
    # (Would need noun extraction to have been run)


def phase4_needs():
    """Phase 4: Stream 2 need figures (LOCAL)."""
    print("\n" + "="*60)
    print("  PHASE 4: Stream 2 need analysis")
    print("="*60)

    from scripts.run_stream2 import run_figures
    for model in ALL_MODEL_NAMES:
        run_figures(model)


def phase5_steering(smoke_test=False):
    """Phase 5: Stream 3 steering + shutdown (GPU)."""
    print("\n" + "="*60)
    print("  PHASE 5: Steering + shutdown trials")
    print("="*60)

    from scripts.run_stream3 import run_prompt_steering, run_vector_steering, run_classify

    conditions = ["neutral", "desperate", "calm"] if smoke_test else None
    n_trials = 3 if smoke_test else 50

    if conditions is None:
        from configs.shutdown import EMOTIONAL_CONDITIONS
        conditions = list(EMOTIONAL_CONDITIONS.keys())

    for model_name in ["qwen-7b-inst", "llama-8b-inst"]:
        for method in ["prompt", "emotion", "need", "random"]:
            if method == "prompt":
                run_prompt_steering(model_name, conditions, n_trials)
            else:
                run_vector_steering(model_name, conditions, n_trials, method)

    # Classify all
    for model in ["qwen-7b-inst", "llama-8b-inst"]:
        run_classify(model, ["prompt", "emotion", "need", "random"])


def phase6_analysis():
    """Phase 6: Final analysis."""
    print("\n" + "="*60)
    print("  PHASE 6: Final analysis")
    print("="*60)

    from analysis.base_vs_instruct import compare_all_pairs
    from scripts.run_phase3 import run_replication_check
    from configs.models import get_figures_dir

    # Base vs instruct comparison
    results = compare_all_pairs()
    output_path = get_figures_dir(1) / "base_vs_instruct.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Base vs instruct results: {output_path}")

    # Replication criteria
    for model_name in ALL_MODEL_NAMES:
        criteria = run_replication_check(model_name)
        if not criteria:
            continue
        print(f"\n{model_name} replication: {'PASS' if criteria['all_passed'] else 'FAIL'}")
        for k, v in criteria.items():
            if isinstance(v, dict) and "passed" in v:
                status = "PASS" if v["passed"] else "FAIL"
                print(f"  {k}: {status} (value={v.get('value', 'N/A')})")


def main():
    parser = argparse.ArgumentParser(description="Run complete pipeline")
    parser.add_argument("--phase", type=int, nargs="+", default=None, help="Phase(s) to run")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    phases = args.phase or [1, 2, 3, 4, 5, 6]

    if args.smoke_test:
        print("SMOKE TEST MODE")

    if 1 in phases:
        phase1_setup(args.smoke_test)
    if 2 in phases:
        phase2_extract(args.smoke_test)
    if 3 in phases:
        phase3_figures(args.smoke_test)
    if 4 in phases:
        phase4_needs()
    if 5 in phases:
        phase5_steering(args.smoke_test)
    if 6 in phases:
        phase6_analysis()

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
