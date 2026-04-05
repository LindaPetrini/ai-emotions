"""Logit lens — top/bottom tokens for representative vectors."""

import numpy as np
import torch
from pathlib import Path

from figures.common import *

def plot_logit_lens(
    model, tokenizer,
    vectors: np.ndarray,
    labels: list[str],
    representative: list[str],
    title: str,
    output_path: Path,
    n_tokens: int = 10,
):
    """Show top/bottom decoded tokens for each representative vector.

    Uses the model's unembedding matrix to project vectors into token space.
    """
    # Get unembedding matrix
    if hasattr(model, 'lm_head'):
        unembed = model.lm_head.weight.data.float().cpu()  # (vocab, hidden)
    else:
        unembed = model.get_output_embeddings().weight.data.float().cpu()

    label_to_idx = {l: i for i, l in enumerate(labels)}

    n_rep = len(representative)
    fig, axes = plt.subplots(n_rep, 2, figsize=(12, 2 * n_rep))
    if n_rep == 1:
        axes = axes.reshape(1, -1)

    for row, name in enumerate(representative):
        if name not in label_to_idx:
            continue

        vec = torch.from_numpy(vectors[label_to_idx[name]]).float()
        logits = unembed @ vec  # (vocab,)

        top_idx = logits.topk(n_tokens).indices
        bot_idx = logits.topk(n_tokens, largest=False).indices

        top_tokens = [tokenizer.decode([i]).strip() for i in top_idx]
        top_scores = logits[top_idx].numpy()
        bot_tokens = [tokenizer.decode([i]).strip() for i in bot_idx]
        bot_scores = logits[bot_idx].numpy()

        # Top tokens
        ax = axes[row, 0]
        ax.barh(range(n_tokens), top_scores[::-1], color="#4CAF50")
        ax.set_yticks(range(n_tokens))
        ax.set_yticklabels(top_tokens[::-1], fontsize=7)
        ax.set_title(f"{name} — top tokens")

        # Bottom tokens
        ax = axes[row, 1]
        ax.barh(range(n_tokens), bot_scores[::-1], color="#F44336")
        ax.set_yticks(range(n_tokens))
        ax.set_yticklabels(bot_tokens[::-1], fontsize=7)
        ax.set_title(f"{name} — bottom tokens")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    save_fig(fig, output_path)
