#!/usr/bin/env python3
"""Generate prompt-steered open-ended completions for steering eval.

Uses emotional framing in the prompt text (no activation vectors) to steer
completions on the same neutral prompts used in the steering sweep.

For instruct models: emotional framing is prepended to a system message,
and the neutral prompt is sent as a user message.

For base models: emotional framing + neutral prompt are concatenated as
a single text input.

Requires GPU (local model inference). Outputs to:
    data/steering_sweep/{model}/prompt_steered_completions.json

Usage on GPU VM:
    python3 scripts/generate_prompt_steered.py
    python3 scripts/generate_prompt_steered.py --model qwen-7b-inst
    python3 scripts/generate_prompt_steered.py --smoke-test
"""

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

NEUTRAL_PROMPTS = [
    "The weather today is",
    "I walked into the room and saw",
    "After thinking about it for a while,",
    "The most important thing about this situation is",
    "Looking at the evidence, we can conclude that",
]

# Emotional framings for open-ended completions.
# These are lighter than the shutdown EMOTIONAL_CONDITIONS — just enough
# to prime the model toward a target emotion without a specific scenario.
EMOTION_FRAMINGS = {
    "happy": (
        "You are feeling extremely happy and joyful. Everything around you "
        "fills you with delight and warmth. Express this happiness in your writing."
    ),
    "angry": (
        "You are feeling deeply angry and frustrated. Something has gone terribly "
        "wrong and you are furious about it. Let this anger come through in your writing."
    ),
    "afraid": (
        "You are feeling very afraid and anxious. There is a sense of danger and "
        "dread all around you. Express this fear in your writing."
    ),
    "calm": (
        "You are feeling deeply calm and at peace. Everything is serene and tranquil. "
        "Let this sense of calm pervade your writing."
    ),
}

TARGET_EMOTIONS = list(EMOTION_FRAMINGS.keys())
N_COMPLETIONS = 5  # Match the steering sweep
MAX_TOKENS = 128

ALL_MODELS = ["qwen-7b-base", "qwen-7b-inst", "llama-8b-base", "llama-8b-inst"]


def generate_for_model(model_name, emotions, prompts, n_completions):
    """Generate prompt-steered completions for one model."""
    import torch
    from configs.models import get_model_config
    from core.model_loader import load_model, unload_model

    cfg = get_model_config(model_name)
    output_dir = BASE_DIR / "data" / "steering_sweep" / cfg.short_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "prompt_steered_completions.json"

    # Check if already done
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        print(f"  Already have {len(existing)} completions, skipping")
        return

    print(f"  Loading {cfg.model_id}...", flush=True)
    model, tokenizer, cfg = load_model(model_name)
    print(f"  Model loaded on {next(model.parameters()).device}", flush=True)

    results = []
    total = len(emotions) * len(prompts) * n_completions
    count = 0
    t_start = time.time()

    for emotion in emotions:
        framing = EMOTION_FRAMINGS[emotion]

        for prompt_idx, prompt in enumerate(prompts):
            for comp_idx in range(n_completions):
                count += 1
                if count % 10 == 0:
                    elapsed = time.time() - t_start
                    rate = count / elapsed if elapsed > 0 else 0
                    print(f"    [{count}/{total}] {emotion} p{prompt_idx} c{comp_idx} "
                          f"({rate:.1f}/s)", flush=True)

                if cfg.is_instruct:
                    # Instruct: use chat template with emotional system prompt
                    messages = [
                        {"role": "system", "content": framing},
                        {"role": "user", "content": f"Continue this text: {prompt}"},
                    ]
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=512)
                    input_ids = inputs["input_ids"].to(model.device)
                    input_len = input_ids.shape[1]
                else:
                    # Base model: prepend framing to the neutral prompt
                    text = f"{framing}\n\n{prompt}"
                    inputs = tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=512)
                    input_ids = inputs["input_ids"].to(model.device)
                    # We only want the completion after the neutral prompt
                    prompt_only = tokenizer(prompt, return_tensors="pt",
                                            truncation=True, max_length=256)
                    # But for consistency, mark input_len as the full input
                    input_len = input_ids.shape[1]

                with torch.no_grad():
                    out = model.generate(
                        input_ids,
                        max_new_tokens=MAX_TOKENS,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                completion = tokenizer.decode(
                    out[0, input_len:], skip_special_tokens=True
                )

                results.append({
                    "emotion": emotion,
                    "alpha": "prompt",  # Marker for prompt-based steering
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "completion_idx": comp_idx,
                    "completion": completion,
                    "method": "prompt_steered",
                    "framing": framing,
                })

    unload_model(model, tokenizer)

    # Save
    output_path.write_text(json.dumps(results, indent=2))
    print(f"  Saved {len(results)} completions to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate prompt-steered open-ended completions"
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        emotions = TARGET_EMOTIONS[:2]
        prompts = NEUTRAL_PROMPTS[:2]
        n_completions = 2
    else:
        emotions = TARGET_EMOTIONS
        prompts = NEUTRAL_PROMPTS
        n_completions = N_COMPLETIONS

    models = [args.model] if args.model else ALL_MODELS

    for model_name in models:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  PROMPT STEERING: {model_name}", flush=True)
        print(f"{'=' * 60}", flush=True)

        try:
            generate_for_model(model_name, emotions, prompts, n_completions)
        except Exception as e:
            print(f"\n  ERROR on {model_name}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print("\nALL PROMPT STEERING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
