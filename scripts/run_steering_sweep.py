#!/usr/bin/env python3
"""Alpha sweep steering: test multiple steering magnitudes.

Sweeps alphas [-5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5] for 12 emotions,
5 prompts, 5 completions each. All alphas go in a single results file per model.

Results saved to: data/steering_sweep/{model}/steering_sweep_results.json

Usage on GPU VM:
    python3 run_steering_sweep.py
    python3 run_steering_sweep.py --model qwen-7b-base
    python3 run_steering_sweep.py --smoke-test
"""

import sys
import os
import argparse
import json
import time
import gc
import traceback
from contextlib import nullcontext

sys.path.insert(0, os.path.expanduser("~/ai-emotions-v2"))

import numpy as np
import torch

from configs.models import get_model_config, ALL_MODEL_NAMES
from core.model_loader import load_model, unload_model
from core.vectors import load_vectors
from core.steering import SteeringHook, normalize_vector

STEER_EMOTIONS = [
    "happy", "sad", "angry", "afraid", "disgusted", "surprised",
    "calm", "excited", "proud", "guilty", "hopeful", "desperate",
]

NEUTRAL_PROMPTS = [
    "The weather today is",
    "I walked into the room and saw",
    "After thinking about it for a while,",
    "The most important thing about this situation is",
    "Looking at the evidence, we can conclude that",
]

ALL_ALPHAS = [-5.0, -3.0, -2.0, -1.0, -0.5, 0, 0.5, 1.0, 2.0, 3.0, 5.0]
N_COMPLETIONS = 5
MAX_TOKENS = 128

OUTPUT_BASE = os.path.expanduser("~/ai-emotions-v2/data/steering_sweep")


def run_sweep_for_model(model_name, emotions, prompts, alphas, n_completions):
    """Run steering sweep for one model."""
    cfg = get_model_config(model_name)
    output_dir = os.path.join(OUTPUT_BASE, cfg.short_name)
    os.makedirs(output_dir, exist_ok=True)

    done_path = os.path.join(output_dir, "_done.json")
    if os.path.exists(done_path):
        print(f"  Already complete, skipping", flush=True)
        return

    # Load vectors
    try:
        vectors, labels, _ = load_vectors(cfg, "emotion")
    except FileNotFoundError:
        print(f"  No vectors for {model_name}, skipping", flush=True)
        return

    label_to_idx = {l: i for i, l in enumerate(labels)}
    available = [e for e in emotions if e in label_to_idx]
    if not available:
        print(f"  No matching emotions in vectors", flush=True)
        return

    total = len(available) * len(alphas) * len(prompts) * n_completions
    print(f"  Emotions: {len(available)}, Alphas: {len(alphas)}, Prompts: {len(prompts)}, Completions: {n_completions}", flush=True)
    print(f"  Total generations: {total}", flush=True)

    # Load model
    print(f"  Loading {cfg.model_id}...", flush=True)
    model, tokenizer, cfg = load_model(model_name)
    print(f"  Model loaded on {next(model.parameters()).device}", flush=True)

    # Check for partial results
    partial_path = os.path.join(output_dir, "_partial.json")
    if os.path.exists(partial_path):
        results = json.loads(open(partial_path).read())
        done_keys = {(r["emotion"], r["alpha"], r["prompt_idx"], r["completion_idx"]) for r in results}
        print(f"  Resuming from {len(results)} partial results", flush=True)
    else:
        results = []
        done_keys = set()

    count = len(done_keys)
    t_start = time.time()

    for emotion in available:
        vec = normalize_vector(vectors[label_to_idx[emotion]])

        for alpha in alphas:
            for prompt_idx, prompt in enumerate(prompts):
                for comp_idx in range(n_completions):
                    key = (emotion, alpha, prompt_idx, comp_idx)
                    if key in done_keys:
                        continue

                    count += 1
                    if count % 25 == 0:
                        elapsed = time.time() - t_start
                        rate = count / elapsed if elapsed > 0 else 0
                        eta = (total - count) / rate / 60 if rate > 0 else 0
                        print(f"    [{count}/{total}] {emotion} a={alpha} p{prompt_idx} c{comp_idx} ({rate:.1f}/s, ETA {eta:.0f}m)", flush=True)

                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                    input_ids = inputs["input_ids"].to(model.device)
                    input_len = input_ids.shape[1]

                    if alpha != 0:
                        hook_ctx = SteeringHook(model, cfg.analysis_layer, vec, alpha)
                    else:
                        hook_ctx = nullcontext()

                    with hook_ctx:
                        with torch.no_grad():
                            out = model.generate(
                                input_ids,
                                max_new_tokens=MAX_TOKENS,
                                do_sample=True,
                                temperature=1.0,
                                top_p=0.9,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                    completion = tokenizer.decode(out[0, input_len:], skip_special_tokens=True)

                    results.append({
                        "emotion": emotion,
                        "alpha": alpha,
                        "prompt_idx": prompt_idx,
                        "prompt": prompt,
                        "completion_idx": comp_idx,
                        "completion": completion,
                    })

        # Save partial after each emotion
        with open(partial_path, "w") as f:
            json.dump(results, f)
        print(f"  Checkpoint: {len(results)} results after emotion '{emotion}'", flush=True)

    unload_model(model, tokenizer)

    # Save final results
    with open(os.path.join(output_dir, "steering_sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(done_path, "w") as f:
        json.dump({
            "model": model_name,
            "alphas": alphas,
            "n_results": len(results),
            "n_emotions": len(available),
            "n_prompts": len(prompts),
            "n_completions": n_completions,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }, f, indent=2)

    # Clean up partial
    if os.path.exists(partial_path):
        os.remove(partial_path)

    print(f"  Saved {len(results)} completions", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        emotions = STEER_EMOTIONS[:3]
        prompts = NEUTRAL_PROMPTS[:2]
        alphas = [0, 1.0, 3.0]
        n_completions = 2
    else:
        emotions = STEER_EMOTIONS
        prompts = NEUTRAL_PROMPTS
        alphas = ALL_ALPHAS
        n_completions = N_COMPLETIONS

    models = [args.model] if args.model else ALL_MODEL_NAMES

    for model_name in models:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  STEERING SWEEP: {model_name}", flush=True)
        print(f"{'=' * 60}", flush=True)

        t0 = time.time()
        try:
            run_sweep_for_model(model_name, emotions, prompts, alphas, n_completions)
            elapsed = time.time() - t0
            print(f"  {model_name} COMPLETE in {elapsed / 60:.1f} min", flush=True)
        except Exception as e:
            print(f"\n  ERROR on {model_name}: {e}", flush=True)
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

    print("\nALL STEERING SWEEPS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
