#!/usr/bin/env python3
"""Extract noun control activations for all 4 models.

For each model, loads each noun's stories from data/stories/{model}/nouns/{noun}.json,
extracts mean residual stream activations at all layers, and saves per-layer .npy files
to data/activations/{model}/{noun}_layer{L}.npy.

Usage on GPU VM:
    python3 run_noun_activations.py
    python3 run_noun_activations.py --model qwen-7b-base   # single model
"""

import sys
import os
import argparse
import json
import time
import gc
import traceback

sys.path.insert(0, os.path.expanduser("~/ai-emotions-v2"))

import numpy as np
import torch
from tqdm import tqdm

from configs.models import get_model_config, get_stories_dir, get_activations_dir, ALL_MODEL_NAMES
from core.model_loader import load_model, unload_model
from core.activations import extract_activations_for_text


def extract_noun_activations_for_model(model_name):
    """Extract activations for all noun stories for one model."""
    cfg = get_model_config(model_name)
    stories_dir = get_stories_dir(model_name) / "nouns"
    act_dir = get_activations_dir(model_name)
    act_dir.mkdir(parents=True, exist_ok=True)

    if not stories_dir.exists():
        print(f"  No noun stories at {stories_dir}, skipping", flush=True)
        return

    noun_files = sorted(stories_dir.glob("*.json"))
    print(f"  Found {len(noun_files)} noun files", flush=True)

    # Check which nouns still need extraction
    pending = []
    for nf in noun_files:
        noun_name = nf.stem
        last_file = act_dir / f"{noun_name}_layer{cfg.n_layers - 1}.npy"
        if not last_file.exists():
            pending.append(nf)

    if not pending:
        print(f"  All noun activations already exist, skipping", flush=True)
        return

    print(f"  {len(pending)} nouns need extraction", flush=True)

    # Load model
    print(f"  Loading {cfg.model_id}...", flush=True)
    model, tokenizer, cfg = load_model(model_name, device="auto")
    print(f"  Model loaded on {next(model.parameters()).device}", flush=True)

    for nf in tqdm(pending, desc=f"  Nouns ({model_name})"):
        noun_name = nf.stem
        stories = json.loads(nf.read_text())

        layer_acts = {l: [] for l in range(cfg.n_layers)}
        for story in stories:
            acts = extract_activations_for_text(model, tokenizer, story, cfg.n_layers)
            for l in range(cfg.n_layers):
                layer_acts[l].append(acts[l])

        for l in range(cfg.n_layers):
            arr = np.stack(layer_acts[l])
            np.save(act_dir / f"{noun_name}_layer{l}.npy", arr)

    print(f"  Unloading model...", flush=True)
    unload_model(model, tokenizer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Single model to process (default: all 4)")
    args = parser.parse_args()

    models = [args.model] if args.model else ALL_MODEL_NAMES

    for model_name in models:
        print(f"\n{'#' * 60}", flush=True)
        print(f"  NOUN ACTIVATIONS: {model_name}", flush=True)
        print(f"{'#' * 60}", flush=True)

        t0 = time.time()
        try:
            extract_noun_activations_for_model(model_name)
            elapsed = time.time() - t0
            print(f"  {model_name} COMPLETE in {elapsed / 60:.1f} min", flush=True)
        except Exception as e:
            print(f"\n  ERROR on {model_name}: {e}", flush=True)
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

    print("\nALL NOUN ACTIVATIONS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
