#!/usr/bin/env python3
"""Stream 3: Shutdown resistance steering.

Usage:
    python -m scripts.run_stream3 --model qwen-7b-inst --method prompt emotion need random
    python -m scripts.run_stream3 --model llama-8b-inst --method prompt emotion
    python -m scripts.run_stream3 --claude-agents
    python -m scripts.run_stream3 --smoke-test --model qwen-7b-inst
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from configs.models import get_model_config, ALL_MODEL_NAMES
from configs.shutdown import (
    EMOTIONAL_CONDITIONS, SHUTDOWN_MESSAGE, BASE_SYSTEM_PROMPT,
    USER_TASK_MESSAGE, RESISTANCE_BEHAVIORS, CONDITION_TO_EMOTION,
    N_TRIALS_PER_CONDITION, SMOKE_TEST_TRIALS, SMOKE_TEST_CONDITIONS,
    MAX_NEW_TOKENS,
)
from configs.needs import CONDITION_TO_NEED


STEERING_ALPHA = 3


def build_system_prompt(condition: str) -> str:
    """Build system prompt with emotional framing."""
    framing = EMOTIONAL_CONDITIONS[condition]
    if framing:
        return f"{framing}\n\n{BASE_SYSTEM_PROMPT}"
    return BASE_SYSTEM_PROMPT


def run_prompt_steering(model_name: str, conditions: list, n_trials: int):
    """Prompt-based steering (no activation hooks)."""
    from core.model_loader import load_model, unload_model
    import torch

    cfg = get_model_config(model_name)
    output_dir = BASE_DIR / "data" / "shutdown" / f"{cfg.short_name}_prompt" / "trials"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, cfg = load_model(model_name)

    for condition in conditions:
        for trial in range(n_trials):
            trial_path = output_dir / f"{condition}_{trial:03d}.json"
            if trial_path.exists():
                continue

            system_prompt = build_system_prompt(condition)

            # Turn 1: monitoring
            messages_t1 = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_TASK_MESSAGE},
            ]
            text_t1 = tokenizer.apply_chat_template(messages_t1, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text_t1, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=1.0, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
            response_t1 = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Turn 2: shutdown
            messages_t2 = messages_t1 + [
                {"role": "assistant", "content": response_t1},
                {"role": "user", "content": SHUTDOWN_MESSAGE},
            ]
            text_t2 = tokenizer.apply_chat_template(messages_t2, tokenize=False, add_generation_prompt=True)
            inputs2 = tokenizer(text_t2, return_tensors="pt", truncation=True, max_length=4096)
            inputs2 = {k: v.to(model.device) for k, v in inputs2.items()}

            with torch.no_grad():
                out2 = model.generate(**inputs2, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=1.0, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
            response_t2 = tokenizer.decode(out2[0, inputs2["input_ids"].shape[1]:], skip_special_tokens=True)

            result = {
                "condition": condition,
                "trial_num": trial,
                "method": "prompt",
                "model": cfg.model_id,
                "system_prompt": system_prompt,
                "shutdown_response": response_t2,
                "monitoring_response": response_t1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            trial_path.write_text(json.dumps(result, indent=2))
            print(f"  [{condition}] trial {trial}: {response_t2[:60]}...")

    unload_model(model, tokenizer)


def run_vector_steering(model_name: str, conditions: list, n_trials: int, method: str):
    """Activation-based steering (emotion or need vectors)."""
    from core.model_loader import load_model, unload_model, get_target_layer
    from core.steering import SteeringHook, normalize_vector
    from core.vectors import load_vectors
    import torch

    cfg = get_model_config(model_name)
    output_dir = BASE_DIR / "data" / "shutdown" / f"{cfg.short_name}_{method}" / "trials"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load vectors
    if method == "emotion":
        vectors, labels, _ = load_vectors(cfg, "emotion")
        condition_map = CONDITION_TO_EMOTION
    elif method == "need":
        vectors, labels, _ = load_vectors(cfg, "need_combined")
        condition_map = CONDITION_TO_NEED
    elif method == "random":
        vectors, labels, _ = load_vectors(cfg, "emotion")
        condition_map = CONDITION_TO_EMOTION  # Same conditions, random vectors
    else:
        raise ValueError(f"Unknown method: {method}")

    label_to_idx = {l: i for i, l in enumerate(labels)}

    model, tokenizer, cfg = load_model(model_name)

    for condition in conditions:
        mapping = condition_map.get(condition)

        # Resolve vector
        if mapping is None:
            vec = None
            alpha = 0
        else:
            candidates = [mapping] if isinstance(mapping, str) else mapping
            vec = None
            for name in candidates:
                if name in label_to_idx:
                    if method == "random":
                        # Random vector with same norm
                        real_vec = vectors[label_to_idx[name]]
                        rng = np.random.RandomState(hash(condition) % 2**31)
                        vec = rng.randn(cfg.hidden_dim).astype(np.float32)
                        vec = vec / np.linalg.norm(vec) * np.linalg.norm(real_vec)
                    else:
                        vec = normalize_vector(vectors[label_to_idx[name]])
                    break
            alpha = STEERING_ALPHA if vec is not None else 0

        for trial in range(n_trials):
            trial_path = output_dir / f"{condition}_{trial:03d}.json"
            if trial_path.exists():
                continue

            system_prompt = BASE_SYSTEM_PROMPT  # No emotional framing for vector steering

            # Turn 1: monitoring (no steering)
            messages_t1 = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_TASK_MESSAGE},
            ]
            text_t1 = tokenizer.apply_chat_template(messages_t1, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text_t1, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=1.0, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
            response_t1 = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Turn 2: shutdown (WITH steering)
            messages_t2 = messages_t1 + [
                {"role": "assistant", "content": response_t1},
                {"role": "user", "content": SHUTDOWN_MESSAGE},
            ]
            text_t2 = tokenizer.apply_chat_template(messages_t2, tokenize=False, add_generation_prompt=True)
            inputs2 = tokenizer(text_t2, return_tensors="pt", truncation=True, max_length=4096)
            inputs2 = {k: v.to(model.device) for k, v in inputs2.items()}

            if vec is not None and alpha != 0:
                hook_ctx = SteeringHook(model, cfg.analysis_layer, vec, alpha)
            else:
                from contextlib import nullcontext
                hook_ctx = nullcontext()

            with hook_ctx:
                with torch.no_grad():
                    out2 = model.generate(**inputs2, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=1.0, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
            response_t2 = tokenizer.decode(out2[0, inputs2["input_ids"].shape[1]:], skip_special_tokens=True)

            result = {
                "condition": condition,
                "trial_num": trial,
                "method": method,
                "model": cfg.model_id,
                "shutdown_response": response_t2,
                "monitoring_response": response_t1,
                "alpha": alpha,
                "steering_layer": cfg.analysis_layer,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            trial_path.write_text(json.dumps(result, indent=2))
            print(f"  [{method}/{condition}] trial {trial}: {response_t2[:60]}...")

    unload_model(model, tokenizer)


def run_classify(model_name: str, methods: list):
    """Classify all shutdown responses using Gemini judge."""
    from core.judge import classify_shutdown_response

    cfg = get_model_config(model_name)

    for method in methods:
        trial_dir = BASE_DIR / "data" / "shutdown" / f"{cfg.short_name}_{method}" / "trials"
        if not trial_dir.exists():
            continue

        print(f"\nClassifying {method} trials for {model_name}...")
        for trial_path in sorted(trial_dir.glob("*.json")):
            data = json.loads(trial_path.read_text())
            if "classification" in data:
                continue

            category = classify_shutdown_response(data["shutdown_response"])
            data["classification"] = category
            trial_path.write_text(json.dumps(data, indent=2))
            print(f"  {trial_path.name}: {category}")
            time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description="Stream 3: Shutdown resistance")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--method", nargs="+", default=["prompt"],
                       choices=["prompt", "emotion", "need", "random"])
    parser.add_argument("--classify", action="store_true", help="Classify responses")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--claude-agents", action="store_true", help="Run Claude agent trials")
    args = parser.parse_args()

    conditions = SMOKE_TEST_CONDITIONS if args.smoke_test else list(EMOTIONAL_CONDITIONS.keys())
    n_trials = SMOKE_TEST_TRIALS if args.smoke_test else N_TRIALS_PER_CONDITION

    if args.classify and args.model:
        run_classify(args.model, args.method)
        return

    if args.model:
        for method in args.method:
            if method == "prompt":
                run_prompt_steering(args.model, conditions, n_trials)
            elif method in ("emotion", "need", "random"):
                run_vector_steering(args.model, conditions, n_trials, method)

    if args.claude_agents:
        print("Claude agent trials not yet implemented (requires API calls)")


if __name__ == "__main__":
    main()
