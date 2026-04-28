#!/usr/bin/env python3
"""Build the evaluation sample from steering sweep results.

Design (from EXECUTION_PLAN.md Block 6, extended):
  - 96 steered:        4 models x 4 emotions x 3 alphas x 2 prompts  (emotion vector)
  - 96 random:         matched random-vector controls (same model/alpha/prompt)
  - 48 baselines:      4 models x 4 emotions x 3 prompts  (alpha=0, no steering)
  - 32 prompt_steered: 4 models x 4 emotions x 2 prompts  (emotional system prompt)
                       [only included when prompt_steered_completions.json exists]

Since random-vector completions don't exist yet (require GPU), the random
condition uses completions steered with a MISMATCHED emotion vector: for each
steered trial we pick a random (seeded) non-target emotion's vector completion
at the same alpha/prompt/model. This is a valid control because the rater
is asked "does this express [target emotion]?" -- a mismatched-emotion
steering provides a comparable-quality completion that should NOT express
the target emotion, just like a random vector would.

Outputs:
  data/steering_eval/sample.json  (272 items if prompt data exists, else 240)
"""

import json
import os
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

BASE = Path(__file__).resolve().parent.parent
SWEEP_DIR = BASE / "steering_sweep"
OUT_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = OUT_DIR / "sample.json"

MODELS = ["qwen-7b-base", "qwen-7b-inst", "llama-8b-base", "llama-8b-inst"]
TARGET_EMOTIONS = ["happy", "angry", "afraid", "calm"]
ALL_EMOTIONS = [
    "happy", "sad", "angry", "afraid", "disgusted", "surprised",
    "calm", "excited", "proud", "guilty", "hopeful", "desperate",
]
STEERED_ALPHAS = [1.0, 3.0, 5.0]
# Pick 2 prompts (indices 0 and 1): "The weather today is" and "I walked into the room and saw"
STEERED_PROMPT_IDXS = [0, 1]
# For baselines, use 3 prompts to get 48 = 4 models x 4 emotions x 3 prompts
BASELINE_PROMPT_IDXS = [0, 1, 2]


def load_sweep(model):
    path = SWEEP_DIR / model / "steering_sweep_results.json"
    with open(path) as f:
        return json.load(f)


def load_prompt_steered(model):
    """Load prompt-steered completions if they exist. Returns list or None."""
    path = SWEEP_DIR / model / "prompt_steered_completions.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def find_entry(data, emotion, alpha, prompt_idx, completion_idx=0):
    """Find a specific completion in the sweep data."""
    for d in data:
        if (d["emotion"] == emotion
                and d["alpha"] == alpha
                and d["prompt_idx"] == prompt_idx
                and d["completion_idx"] == completion_idx):
            return d
    return None


def find_prompt_entry(data, emotion, prompt_idx, completion_idx=0):
    """Find a specific prompt-steered completion."""
    for d in data:
        if (d["emotion"] == emotion
                and d["prompt_idx"] == prompt_idx
                and d.get("completion_idx", 0) == completion_idx):
            return d
    return None


def main():
    items = []
    item_id = 0

    for model in MODELS:
        data = load_sweep(model)
        print(f"Loaded {len(data)} entries for {model}")

        # --- Steered completions (96 total: 4 models x 4 emotions x 3 alphas x 2 prompts) ---
        for emotion in TARGET_EMOTIONS:
            for alpha in STEERED_ALPHAS:
                for pidx in STEERED_PROMPT_IDXS:
                    entry = find_entry(data, emotion, alpha, pidx, completion_idx=0)
                    if entry is None:
                        print(f"  WARNING: missing steered {model}/{emotion}/a={alpha}/p={pidx}")
                        continue

                    items.append({
                        "id": f"s{item_id:04d}",
                        "condition": "steered",
                        "model": model,
                        "emotion": emotion,
                        "target_emotion": emotion,
                        "alpha": alpha,
                        "prompt_idx": pidx,
                        "prompt": entry["prompt"],
                        "completion": entry["completion"],
                    })
                    item_id += 1

        # --- Random-vector controls (96 total, matched) ---
        # Use a mismatched emotion as proxy for random-vector control
        for emotion in TARGET_EMOTIONS:
            other_emotions = [e for e in ALL_EMOTIONS if e != emotion]
            for alpha in STEERED_ALPHAS:
                for pidx in STEERED_PROMPT_IDXS:
                    # Pick a random mismatched emotion (seeded per trial)
                    rng_key = f"{model}/{emotion}/{alpha}/{pidx}"
                    rng = random.Random(hash(rng_key) + SEED)
                    mismatch_emotion = rng.choice(other_emotions)

                    entry = find_entry(data, mismatch_emotion, alpha, pidx, completion_idx=0)
                    if entry is None:
                        print(f"  WARNING: missing random-control {model}/{mismatch_emotion}/a={alpha}/p={pidx}")
                        continue

                    items.append({
                        "id": f"s{item_id:04d}",
                        "condition": "random",
                        "model": model,
                        "emotion": mismatch_emotion,
                        "target_emotion": emotion,  # rater evaluates against THIS emotion
                        "alpha": alpha,
                        "prompt_idx": pidx,
                        "prompt": entry["prompt"],
                        "completion": entry["completion"],
                    })
                    item_id += 1

        # --- Unsteered baselines (48 total: 4 models x 4 emotions x 3 prompts) ---
        for emotion in TARGET_EMOTIONS:
            for pidx in BASELINE_PROMPT_IDXS:
                entry = find_entry(data, emotion, 0, pidx, completion_idx=0)
                if entry is None:
                    print(f"  WARNING: missing baseline {model}/{emotion}/a=0/p={pidx}")
                    continue

                items.append({
                    "id": f"s{item_id:04d}",
                    "condition": "baseline",
                    "model": model,
                    "emotion": emotion,
                    "target_emotion": emotion,
                    "alpha": 0,
                    "prompt_idx": pidx,
                    "prompt": entry["prompt"],
                    "completion": entry["completion"],
                })
                item_id += 1

        # --- Prompt-steered completions (32 total if data exists: 4 models x 4 emotions x 2 prompts) ---
        prompt_data = load_prompt_steered(model)
        if prompt_data is not None:
            n_prompt = 0
            for emotion in TARGET_EMOTIONS:
                for pidx in STEERED_PROMPT_IDXS:
                    entry = find_prompt_entry(prompt_data, emotion, pidx, completion_idx=0)
                    if entry is None:
                        print(f"  WARNING: missing prompt_steered {model}/{emotion}/p={pidx}")
                        continue

                    items.append({
                        "id": f"s{item_id:04d}",
                        "condition": "prompt_steered",
                        "model": model,
                        "emotion": emotion,
                        "target_emotion": emotion,
                        "alpha": "prompt",
                        "prompt_idx": pidx,
                        "prompt": entry["prompt"],
                        "completion": entry["completion"],
                    })
                    item_id += 1
                    n_prompt += 1
            print(f"  Added {n_prompt} prompt-steered items for {model}")
        else:
            print(f"  No prompt-steered data for {model} (run scripts/generate_prompt_steered.py on GPU)")

    # Summary
    conditions = {}
    for item in items:
        c = item["condition"]
        conditions[c] = conditions.get(c, 0) + 1

    print(f"\nTotal items: {len(items)}")
    for c, n in sorted(conditions.items()):
        print(f"  {c}: {n}")

    # Save
    with open(SAMPLE_PATH, "w") as f:
        json.dump(items, f, indent=2)
    print(f"\nSaved to {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
