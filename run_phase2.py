#!/usr/bin/env python3
"""Phase 2: Extract activations and compute vectors for all models."""
import sys, os, time, gc, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from configs.models import MODEL_REGISTRY, get_model_config, get_stories_dir, get_activations_dir, get_vectors_dir
from configs.emotions import ALL_EMOTIONS, EMOTION_TO_CLUSTER, NEUTRAL_TEXTS, IMPLICIT_SCENARIOS, INTENSITY_TEMPLATES, ACTIVITIES
from configs.needs import ALL_NEEDS, NEED_TO_CLUSTER, IMPLICIT_NEED_SCENARIOS, NEED_INTENSITY_TEMPLATES
from core.model_loader import load_model, unload_model
from core.activations import (
    extract_emotion_activations, extract_neutral_activations,
    extract_scenario_activations, extract_intensity_activations,
    extract_activity_activations, extract_need_activations,
)
from core.vectors import compute_emotion_vectors, compute_need_vectors

models_to_run = ["qwen-7b-base", "qwen-7b-inst", "llama-8b-base", "llama-8b-inst"]


def resolve_story_dir(model_name: str) -> Path:
    stories_dir = get_stories_dir(model_name)
    if (stories_dir / "happy.json").exists():
        return stories_dir

    shared_dir = get_stories_dir("qwen-7b-base")
    if (shared_dir / "happy.json").exists():
        return shared_dir

    return stories_dir

for model_name in models_to_run:
    print(f"\n{'#'*60}", flush=True)
    print(f"  MODEL: {model_name}", flush=True)
    print(f"{'#'*60}", flush=True)

    cfg = get_model_config(model_name)
    stories_dir = resolve_story_dir(model_name)

    if not (stories_dir / "happy.json").exists():
        print(f"  No stories at {stories_dir}, skipping", flush=True)
        continue

    t0 = time.time()
    try:
        print(f"  Loading {cfg.model_id}...", flush=True)
        model, tokenizer, cfg = load_model(model_name, device="auto")
        print(f"  Loaded on {next(model.parameters()).device}", flush=True)

        print(f"  Emotion activations ({len(ALL_EMOTIONS)})...", flush=True)
        extract_emotion_activations(model, tokenizer, cfg, ALL_EMOTIONS, stories_dir)

        print(f"  Neutral activations...", flush=True)
        extract_neutral_activations(model, tokenizer, cfg, NEUTRAL_TEXTS)

        print(f"  Scenario activations...", flush=True)
        extract_scenario_activations(model, tokenizer, cfg, IMPLICIT_SCENARIOS)

        print(f"  Intensity activations...", flush=True)
        extract_intensity_activations(model, tokenizer, cfg, INTENSITY_TEMPLATES)

        print(f"  Activity activations...", flush=True)
        extract_activity_activations(model, tokenizer, cfg, ACTIVITIES)

        # Need activations
        need_stories_dir = stories_dir / "needs"
        if need_stories_dir.exists():
            print(f"  Need activations ({len(ALL_NEEDS)})...", flush=True)
            extract_need_activations(model, tokenizer, cfg, ALL_NEEDS, need_stories_dir)

            # Need-specific scenarios and intensity
            # Save to different prefix to avoid overwriting emotion scenarios
            from core.activations import extract_batch
            import json
            act_dir = get_activations_dir(cfg.short_name)

            # Need scenarios
            need_scenario_names = list(IMPLICIT_NEED_SCENARIOS.keys())
            need_scenario_texts = list(IMPLICIT_NEED_SCENARIOS.values())
            last_file = act_dir / f"need_scenarios_layer{cfg.n_layers - 1}.npy"
            if not last_file.exists():
                print(f"  Need scenario activations...", flush=True)
                extract_batch(model, tokenizer, need_scenario_texts, cfg.n_layers, act_dir, "need_scenarios", "Need scenarios")
                with open(act_dir / "need_scenario_names.json", "w") as f:
                    json.dump(need_scenario_names, f)

            # Need intensity
            print(f"  Need intensity activations...", flush=True)
            for iname, icfg in NEED_INTENSITY_TEMPLATES.items():
                last_f = act_dir / f"intensity_{iname}_layer{cfg.n_layers - 1}.npy"
                if last_f.exists():
                    continue
                texts = [icfg["template"].replace("{X}", str(v)) for v in icfg["values"]]
                import numpy as np
                layer_acts = {l: [] for l in range(cfg.n_layers)}
                from core.activations import extract_activations_for_text
                for text in texts:
                    acts = extract_activations_for_text(model, tokenizer, text, cfg.n_layers)
                    for l in range(cfg.n_layers):
                        layer_acts[l].append(acts[l])
                for l in range(cfg.n_layers):
                    arr = np.stack(layer_acts[l])
                    np.save(act_dir / f"intensity_{iname}_layer{l}.npy", arr)

        # Noun activations for control
        noun_stories_dir = stories_dir / "nouns"
        if noun_stories_dir.exists():
            print(f"  Noun control activations...", flush=True)
            import json, numpy as np
            from core.activations import extract_activations_for_text
            act_dir = get_activations_dir(cfg.short_name)
            noun_files = sorted(noun_stories_dir.glob("*.json"))
            for nf in noun_files:
                noun_name = nf.stem
                last_f = act_dir / f"{noun_name}_layer{cfg.n_layers - 1}.npy"
                if last_f.exists():
                    continue
                stories = json.loads(nf.read_text())
                layer_acts = {l: [] for l in range(cfg.n_layers)}
                for story in stories:
                    acts = extract_activations_for_text(model, tokenizer, story, cfg.n_layers)
                    for l in range(cfg.n_layers):
                        layer_acts[l].append(acts[l])
                for l in range(cfg.n_layers):
                    arr = np.stack(layer_acts[l])
                    np.save(act_dir / f"{noun_name}_layer{l}.npy", arr)
                print(f"    noun: {noun_name}", flush=True)

        print(f"  Unloading model...", flush=True)
        unload_model(model, tokenizer)

        # Compute vectors (CPU)
        print(f"  Computing emotion vectors...", flush=True)
        cfg_fresh = get_model_config(model_name)
        compute_emotion_vectors(cfg_fresh, ALL_EMOTIONS, {e: EMOTION_TO_CLUSTER[e] for e in ALL_EMOTIONS})

        if need_stories_dir.exists():
            print(f"  Computing need vectors...", flush=True)
            compute_need_vectors(cfg_fresh, ALL_NEEDS, {n: NEED_TO_CLUSTER[n] for n in ALL_NEEDS})

        elapsed = time.time() - t0
        print(f"\n  {model_name} COMPLETE in {elapsed/60:.1f} min", flush=True)

    except Exception as e:
        print(f"\n  ERROR on {model_name}: {e}", flush=True)
        traceback.print_exc()
        try: unload_model(model, tokenizer)
        except: pass
        gc.collect()
        torch.cuda.empty_cache()

print("\nPHASE 2 COMPLETE", flush=True)
