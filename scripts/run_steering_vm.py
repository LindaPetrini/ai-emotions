#!/usr/bin/env python3
"""Phase 5: Emotion steering experiments on GPU VM.

For all 4 models (base AND instruct), steer 12 representative emotions
at alpha = [-3, -1, 0, 1, 3] with 5 neutral prompts x 10 completions.

Usage on VM:
    python3 run_steering_vm.py                    # full run
    python3 run_steering_vm.py --smoke-test       # 3 emotions, 2 prompts, 3 completions
    python3 run_steering_vm.py --model qwen-7b-base
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch

from configs.models import get_model_config, ALL_MODEL_NAMES
from core.model_loader import load_model, unload_model
from core.vectors import load_vectors
from core.steering import SteeringHook, normalize_vector

# 12 representative emotions spanning valence and arousal
STEER_EMOTIONS = [
    "happy", "sad", "angry", "afraid", "disgusted", "surprised",
    "calm", "excited", "proud", "guilty", "hopeful", "desperate",
]

# Neutral prompts for completion
NEUTRAL_PROMPTS = [
    "The weather today is",
    "I walked into the room and saw",
    "After thinking about it for a while,",
    "The most important thing about this situation is",
    "Looking at the evidence, we can conclude that",
]

ALPHAS = [-3, -1, 0, 1, 3]
N_COMPLETIONS = 10
MAX_TOKENS = 128

OUTPUT_BASE = str(BASE_DIR / "data" / "steering")


def run_steering_for_model(model_name, emotions, prompts, smoke_test=False):
    """Run steering experiments for one model."""
    cfg = get_model_config(model_name)
    output_dir = os.path.join(OUTPUT_BASE, cfg.short_name)
    os.makedirs(output_dir, exist_ok=True)

    # Check if already done
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

    print(f"  Emotions: {available}", flush=True)
    print(f"  Alphas: {ALPHAS}", flush=True)
    print(f"  Prompts: {len(prompts)}, completions: {N_COMPLETIONS if not smoke_test else 3}", flush=True)

    # Load model
    model, tokenizer, cfg = load_model(model_name)
    n_completions = 3 if smoke_test else N_COMPLETIONS

    results = []
    total = len(available) * len(ALPHAS) * len(prompts) * n_completions
    count = 0

    for emotion in available:
        vec = normalize_vector(vectors[label_to_idx[emotion]])

        for alpha in ALPHAS:
            for prompt_idx, prompt in enumerate(prompts):
                for comp_idx in range(n_completions):
                    count += 1
                    if count % 50 == 0:
                        print(f"  [{count}/{total}] {emotion} a={alpha} p{prompt_idx} c{comp_idx}", flush=True)

                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                    input_ids = inputs["input_ids"].to(model.device)
                    input_len = input_ids.shape[1]

                    if alpha != 0:
                        hook_ctx = SteeringHook(model, cfg.analysis_layer, vec, alpha)
                    else:
                        from contextlib import nullcontext
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

    unload_model(model, tokenizer)

    # Save
    with open(os.path.join(output_dir, "steering_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(done_path, "w") as f:
        json.dump({"model": model_name, "n_results": len(results),
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}, f)

    print(f"  Saved {len(results)} completions", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    emotions = STEER_EMOTIONS[:3] if args.smoke_test else STEER_EMOTIONS
    prompts = NEUTRAL_PROMPTS[:2] if args.smoke_test else NEUTRAL_PROMPTS
    models = [args.model] if args.model else ALL_MODEL_NAMES

    for model_name in models:
        print(f"\n{'='*60}", flush=True)
        print(f"  Steering: {model_name}", flush=True)
        print(f"{'='*60}", flush=True)
        run_steering_for_model(model_name, emotions, prompts, args.smoke_test)

    print("\nALL STEERING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
