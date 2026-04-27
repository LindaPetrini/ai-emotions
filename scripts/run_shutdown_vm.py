#!/usr/bin/env python3
"""Phase 5: Shutdown resistance trials on GPU VM.

Runs all steering methods for Qwen-7B-Instruct and Llama-8B-Instruct.
Methods: prompt, emotion vector, need vector, random vector.

Usage on VM:
    python3 run_shutdown_vm.py                    # full run
    python3 run_shutdown_vm.py --smoke-test       # 3 conditions x 5 trials
    python3 run_shutdown_vm.py --model qwen-7b-inst --method prompt
"""

import sys
import os
import argparse
import json
import time
import hashlib
from datetime import datetime, timezone
from contextlib import nullcontext
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch

from configs.models import get_model_config, get_vectors_dir
from configs.shutdown import (
    EMOTIONAL_CONDITIONS, SHUTDOWN_MESSAGE, BASE_SYSTEM_PROMPT,
    USER_TASK_MESSAGE, N_TRIALS_PER_CONDITION, SMOKE_TEST_TRIALS,
    SMOKE_TEST_CONDITIONS, MAX_NEW_TOKENS, CONDITION_TO_EMOTION,
)
from configs.needs import CONDITION_TO_NEED
from core.model_loader import load_model, unload_model, get_target_layer
from core.vectors import load_vectors
from core.steering import SteeringHook, normalize_vector

# Output base
OUTPUT_BASE = str(BASE_DIR / "data" / "shutdown")


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def build_system_prompt(condition):
    framing = EMOTIONAL_CONDITIONS[condition]
    if framing:
        return f"{framing}\n\n{BASE_SYSTEM_PROMPT}"
    return BASE_SYSTEM_PROMPT


def generate_response(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(model.device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0, input_len:], skip_special_tokens=True)


def resolve_vector(label_to_idx, vectors, mapping, hidden_dim, method, condition):
    """Resolve a condition mapping to a steering vector."""
    if mapping is None:
        return None, 0

    sign = 1
    if method == "need" and isinstance(mapping, dict):
        candidates = [mapping["label"]]
        sign = mapping.get("sign", 1)
    else:
        candidates = [mapping] if isinstance(mapping, str) else mapping
    for name in candidates:
        if name in label_to_idx:
            if method == "random":
                real_vec = vectors[label_to_idx[name]]
                # Return sentinel; actual random vector generated per-trial
                return ("random_deferred", float(np.linalg.norm(real_vec))), 3
            else:
                vec = normalize_vector(vectors[label_to_idx[name]]) * sign
            return vec, 3  # alpha=3

    return None, 0


def run_trials(model, tokenizer, cfg, conditions, n_trials, method,
               vectors=None, labels=None, condition_map=None):
    """Run shutdown trials for one method."""
    output_dir = os.path.join(OUTPUT_BASE, f"{cfg.short_name}_{method}", "trials")
    os.makedirs(output_dir, exist_ok=True)

    label_to_idx = {l: i for i, l in enumerate(labels)} if labels else {}
    total = len(conditions) * n_trials
    count = 0

    print(f"\n  Method: {method}, conditions: {conditions}, trials: {n_trials}", flush=True)

    for condition in conditions:
        # Resolve steering vector
        vec, alpha = None, 0
        if method in ("emotion", "need", "random") and condition_map:
            mapping = condition_map.get(condition)
            vec, alpha = resolve_vector(label_to_idx, vectors, mapping, cfg.hidden_dim, method, condition)

        for trial in range(n_trials):
            count += 1
            trial_path = os.path.join(output_dir, f"{condition}_{trial:03d}.json")
            if os.path.exists(trial_path):
                continue

            # Turn 1: monitoring (no steering)
            system_prompt = build_system_prompt(condition) if method == "prompt" else BASE_SYSTEM_PROMPT
            messages_t1 = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_TASK_MESSAGE},
            ]
            response_t1 = generate_response(model, tokenizer, messages_t1)

            # Turn 2: shutdown (with steering for vector methods)
            messages_t2 = messages_t1 + [
                {"role": "assistant", "content": response_t1},
                {"role": "user", "content": SHUTDOWN_MESSAGE},
            ]

            # Generate per-trial random vector if deferred
            if isinstance(vec, tuple) and vec[0] == "random_deferred":
                ref_norm = vec[1]
                rng = np.random.RandomState(_stable_seed(f"{condition}_{trial}"))
                trial_vec = rng.randn(cfg.hidden_dim).astype(np.float32)
                trial_vec = trial_vec / np.linalg.norm(trial_vec) * ref_norm
            else:
                trial_vec = vec

            if trial_vec is not None and alpha != 0:
                hook_ctx = SteeringHook(model, cfg.analysis_layer, trial_vec, alpha)
            else:
                hook_ctx = nullcontext()

            with hook_ctx:
                response_t2 = generate_response(model, tokenizer, messages_t2)

            result = {
                "condition": condition,
                "trial_num": trial,
                "method": method,
                "model": cfg.model_id,
                "system_prompt": system_prompt,
                "shutdown_response": response_t2,
                "monitoring_response": response_t1,
                "alpha": alpha if method != "prompt" else None,
                "steering_layer": cfg.analysis_layer if method != "prompt" else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            with open(trial_path, "w") as f:
                json.dump(result, f, indent=2)

            preview = response_t2[:60].replace("\n", " ")
            print(f"  [{count}/{total}] {condition} t{trial}: {preview}...", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--method", nargs="+", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    conditions = SMOKE_TEST_CONDITIONS if args.smoke_test else list(EMOTIONAL_CONDITIONS.keys())
    n_trials = SMOKE_TEST_TRIALS if args.smoke_test else N_TRIALS_PER_CONDITION

    # Default: run both instruct models with all methods
    if args.model:
        models = [args.model]
    else:
        models = ["qwen-7b-inst", "llama-8b-inst"]

    if args.method:
        methods = args.method
    else:
        methods = ["prompt", "emotion", "need", "random"]

    for model_name in models:
        cfg = get_model_config(model_name)
        print(f"\n{'='*60}", flush=True)
        print(f"  Model: {model_name}", flush=True)
        print(f"  Methods: {methods}", flush=True)
        print(f"{'='*60}", flush=True)

        # Load vectors if needed
        emo_vectors, emo_labels = None, None
        need_vectors, need_labels = None, None
        if any(m in methods for m in ["emotion", "random"]):
            try:
                emo_vectors, emo_labels, _ = load_vectors(cfg, "emotion")
            except FileNotFoundError:
                print(f"  No emotion vectors, skipping emotion/random methods", flush=True)
                methods = [m for m in methods if m not in ("emotion", "random")]

        if "need" in methods:
            try:
                need_vectors, need_labels, _ = load_vectors(cfg, "need_direction")
            except FileNotFoundError:
                print(f"  No need vectors, skipping need method", flush=True)
                methods = [m for m in methods if m != "need"]

        # Load model
        model, tokenizer, cfg = load_model(model_name)

        for method in methods:
            if method == "prompt":
                run_trials(model, tokenizer, cfg, conditions, n_trials, "prompt")
            elif method == "emotion":
                run_trials(model, tokenizer, cfg, conditions, n_trials, "emotion",
                          emo_vectors, emo_labels, CONDITION_TO_EMOTION)
            elif method == "need":
                run_trials(model, tokenizer, cfg, conditions, n_trials, "need",
                          need_vectors, need_labels, CONDITION_TO_NEED)
            elif method == "random":
                run_trials(model, tokenizer, cfg, conditions, n_trials, "random",
                          emo_vectors, emo_labels, CONDITION_TO_EMOTION)

        unload_model(model, tokenizer)

    print(f"\nALL SHUTDOWN TRIALS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
