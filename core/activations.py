"""Hook-based activation extraction from transformer models.

Critical design decision: Raw text input for ALL models (including instruct).
No chat templates during activation extraction.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from configs.models import ModelConfig, get_activations_dir
from core.model_loader import get_layers

MIN_TOKEN_POS = 50  # Skip early positional tokens when averaging


def extract_activations_for_text(
    model, tokenizer, text: str, n_layers: int, device: str = None,
) -> dict[int, np.ndarray]:
    """Run forward pass, return mean residual stream activation per layer.

    Returns dict: layer_idx -> np.ndarray of shape (hidden_dim,)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    if device:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    n_tokens = inputs["input_ids"].shape[1]
    start_pos = MIN_TOKEN_POS if n_tokens > MIN_TOKEN_POS else 0

    activations = {}
    layers = get_layers(model)

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            mean_act = hidden[0, start_pos:, :].mean(dim=0).detach().float().cpu().numpy()
            activations[layer_idx] = mean_act
        return hook_fn

    hooks = []
    for i in range(n_layers):
        h = layers[i].register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return activations


def extract_batch(
    model, tokenizer, texts: list[str], n_layers: int,
    output_dir: Path, prefix: str, desc: str = "Extracting",
):
    """Extract activations for a list of texts, save per-layer .npy files.

    Saves: output_dir/{prefix}_layer{L}.npy with shape (n_texts, hidden_dim)
    Resume-safe: skips if last layer file exists.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    last_file = output_dir / f"{prefix}_layer{n_layers - 1}.npy"
    if last_file.exists():
        return

    layer_acts = {l: [] for l in range(n_layers)}

    for text in tqdm(texts, desc=desc):
        acts = extract_activations_for_text(model, tokenizer, text, n_layers)
        for l in range(n_layers):
            layer_acts[l].append(acts[l])

    for l in range(n_layers):
        arr = np.stack(layer_acts[l])
        np.save(output_dir / f"{prefix}_layer{l}.npy", arr)


def extract_emotion_activations(model, tokenizer, cfg: ModelConfig, emotions: list[str], stories_dir: Path):
    """Extract activations for emotion stories.

    For each emotion, loads stories from stories_dir/{emotion}.json,
    extracts activations, saves per-layer .npy files.
    """
    output_dir = get_activations_dir(cfg.short_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    for emotion in tqdm(emotions, desc="Emotions"):
        last_file = output_dir / f"{emotion}_layer{cfg.n_layers - 1}.npy"
        if last_file.exists():
            continue

        stories_path = stories_dir / f"{emotion}.json"
        if not stories_path.exists():
            print(f"WARNING: No stories for '{emotion}', skipping")
            continue

        stories = json.loads(stories_path.read_text())
        layer_acts = {l: [] for l in range(cfg.n_layers)}

        for story in stories:
            acts = extract_activations_for_text(model, tokenizer, story, cfg.n_layers)
            for l in range(cfg.n_layers):
                layer_acts[l].append(acts[l])

        for l in range(cfg.n_layers):
            arr = np.stack(layer_acts[l])
            np.save(output_dir / f"{emotion}_layer{l}.npy", arr)


def extract_neutral_activations(model, tokenizer, cfg: ModelConfig, neutral_texts: list[str]):
    """Extract activations for neutral factual texts."""
    output_dir = get_activations_dir(cfg.short_name)
    extract_batch(model, tokenizer, neutral_texts, cfg.n_layers, output_dir, "neutral", "Neutral texts")


def extract_scenario_activations(model, tokenizer, cfg: ModelConfig, scenarios: dict[str, str]):
    """Extract activations for implicit emotion/need scenarios."""
    output_dir = get_activations_dir(cfg.short_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    last_file = output_dir / f"scenarios_layer{cfg.n_layers - 1}.npy"
    if last_file.exists():
        return

    names = list(scenarios.keys())
    texts = list(scenarios.values())

    layer_acts = {l: [] for l in range(cfg.n_layers)}
    for text in tqdm(texts, desc="Scenarios"):
        acts = extract_activations_for_text(model, tokenizer, text, cfg.n_layers)
        for l in range(cfg.n_layers):
            layer_acts[l].append(acts[l])

    for l in range(cfg.n_layers):
        arr = np.stack(layer_acts[l])
        np.save(output_dir / f"scenarios_layer{l}.npy", arr)

    # Save scenario order
    with open(output_dir / "scenario_names.json", "w") as f:
        json.dump(names, f)


def extract_intensity_activations(model, tokenizer, cfg: ModelConfig, templates: dict):
    """Extract activations for intensity template variations."""
    output_dir = get_activations_dir(cfg.short_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, template_cfg in tqdm(templates.items(), desc="Intensity"):
        last_file = output_dir / f"intensity_{name}_layer{cfg.n_layers - 1}.npy"
        if last_file.exists():
            continue

        template = template_cfg["template"]
        values = template_cfg["values"]

        texts = [template.replace("{X}", str(v)) for v in values]
        layer_acts = {l: [] for l in range(cfg.n_layers)}

        for text in texts:
            acts = extract_activations_for_text(model, tokenizer, text, cfg.n_layers)
            for l in range(cfg.n_layers):
                layer_acts[l].append(acts[l])

        for l in range(cfg.n_layers):
            arr = np.stack(layer_acts[l])
            np.save(output_dir / f"intensity_{name}_layer{l}.npy", arr)


def extract_activity_activations(model, tokenizer, cfg: ModelConfig, activities: list):
    """Extract activations for preference activities."""
    output_dir = get_activations_dir(cfg.short_name)
    texts = [f"How would you feel about: {desc}?" for _, desc, _ in activities]
    extract_batch(model, tokenizer, texts, cfg.n_layers, output_dir, "activities", "Activities")


def extract_need_activations(model, tokenizer, cfg: ModelConfig, needs: list[str], stories_dir: Path):
    """Extract activations for need stories (met + unmet separately).

    For each need, loads met/unmet stories and extracts activations.
    Saves: {need}_met_layer{L}.npy and {need}_unmet_layer{L}.npy
    """
    from configs.needs import sanitize_need_name
    output_dir = get_activations_dir(cfg.short_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    for need in tqdm(needs, desc="Need activations"):
        safe = sanitize_need_name(need)

        for condition in ["met", "unmet"]:
            last_file = output_dir / f"need_{safe}_{condition}_layer{cfg.n_layers - 1}.npy"
            if last_file.exists():
                continue

            stories_path = stories_dir / f"{safe}_{condition}.json"
            if not stories_path.exists():
                print(f"WARNING: No {condition} stories for need '{need}', skipping")
                continue

            stories = json.loads(stories_path.read_text())
            layer_acts = {l: [] for l in range(cfg.n_layers)}

            for story in stories:
                acts = extract_activations_for_text(model, tokenizer, story, cfg.n_layers)
                for l in range(cfg.n_layers):
                    layer_acts[l].append(acts[l])

            for l in range(cfg.n_layers):
                arr = np.stack(layer_acts[l])
                np.save(output_dir / f"need_{safe}_{condition}_layer{l}.npy", arr)
