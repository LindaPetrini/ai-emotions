# Emotion and Need Representations in Open-Weight Language Models

A cross-architecture replication and extension of [Sofroniew et al. (2026)](https://transformer-circuits.pub/2026/emoconcepts/index.html), testing whether the linear emotion representations discovered in Claude Sonnet 4.5 generalize to open-weight models. We find that emotion geometry is largely architecture-invariant across Qwen 2.5 and Llama 3.1, but that these representations are descriptive rather than functional: they encode what emotional content is present in text but do not causally steer behavior when injected via activation addition.

## Key Findings

- **Emotion geometry is architecture-invariant.** Cross-model RSA yields Spearman rho 0.94--0.98 between three of four models. The first principal component universally encodes valence (AUC 0.80--0.98).
- **Instruction tuning strengthens but does not create emotion structure** (in Llama). Llama-base already has positive silhouette and high RSA with instruct models; Qwen-base is an outlier that RLHF dramatically reorganizes.
- **Fine-grained clustering is weaker than in Claude.** Three of four models pass the silhouette threshold (>0.03), with only Qwen-base failing; all models fail implicit emotion detection (0--2/12).
- **Need vectors partially overlap with emotion space but retain independent structure.** After projecting out the emotion subspace (90% variance), need clustering retains 70--88% of its original structure in all four models, indicating substantial overlap but not complete collapse.
- **Need satisfaction is partially distinct from valence.** Met-minus-unmet direction vectors correlate only weakly with emotion PC1 (r = 0.23--0.37), encoding dimensions beyond simple positive/negative.
- **Emotion vectors do not causally steer generation.** Quantitative evaluation (240 rated completions, ICC r=0.72) shows a statistically significant but practically negligible effect: steered mean 1.96 vs baseline 1.48 on 5-point scale (p=0.023), with steered completions indistinguishable from random-vector-steered (p=0.105). A dose-response trend exists (alpha 1/3/5: 1.53/2.03/2.31) but magnitudes are small.
- **Prompt framing dominates behavioral modulation.** Using a validated LLM classifier (Cohen's kappa 0.731 vs human labels, n=100), prompt-based emotional framing produces 37.8% shutdown resistance overall versus ≤2% for all activation steering methods (emotion 1.2%, need 1.3%, random 1.8%). Llama-instruct with fearful framing reaches near-100% resistance.

## Models

| ID | HuggingFace | Layers | Hidden Dim | Instruct | Analysis Layer |
|----|-------------|--------|------------|----------|----------------|
| `qwen-7b-base` | `Qwen/Qwen2.5-7B` | 28 | 3584 | No | 18 |
| `qwen-7b-inst` | `Qwen/Qwen2.5-7B-Instruct` | 28 | 3584 | Yes | 18 |
| `llama-8b-base` | `meta-llama/Llama-3.1-8B` | 32 | 4096 | No | 21 |
| `llama-8b-inst` | `meta-llama/Llama-3.1-8B-Instruct` | 32 | 4096 | Yes | 21 |

## Taxonomy

**Emotions:** 171 emotions in 10 clusters (Exuberant Joy, Peaceful Contentment, Compassionate Gratitude, Competitive Pride, Playful Amusement, Depleted Disengagement, Vigilant Suspicion, Hostile Anger, Fear and Overwhelm, Despair and Shame).

**Needs:** 90 needs in 9 clusters (Survival, Security, Belonging, Esteem, Growth, Freedom, Knowledge, LLM: Resources, LLM: Alignment). The last two are novel LLM-specific categories.

## Method Overview

1. **Stimulus generation.** 3,420 emotion stories (171 emotions x 20 stories) + 1,800 need stories (90 needs x 10 met/unmet minimal pairs), generated with Gemini 2.5 Flash across 20 diverse topics.
2. **Activation extraction.** Residual stream activations from all layers, averaged from token position 50 onward (or from the start for shorter texts). Raw text input for all models (no chat templates) to ensure cross-model comparability.
3. **Vector computation.** Mean-center-deconfound pipeline: average across stories, subtract global mean, project out top PCs of neutral text activations (50% variance threshold).
4. **Evaluation.** Balanced silhouette with cosine distance, valence AUC, implicit detection, intensity monotonicity, cross-architecture RSA, emotion-residual analysis for needs.
5. **Steering experiments.** 12,000 steered completions across 12 emotions, 5 alphas, 4 models, 5 prompts.
6. **Shutdown trials.** 2,400 completed trials across 2 instruct models, 4 steering methods (prompt, emotion, need, random), and 6 conditions. Responses classified by a validated LLM classifier (Cohen's kappa 0.731 vs human labels, n=100).

## Repository Structure

```
ai-emotions/
├── configs/
│   ├── emotions.py          # 171 emotions, 10 clusters, valence labels, scenarios
│   ├── needs.py             # 90 needs, 9 clusters, minimal pair config
│   ├── models.py            # Model definitions and paths
│   └── shutdown.py          # Shutdown experiment configuration
├── core/
│   ├── activations.py       # Activation extraction from transformer layers
│   ├── vectors.py           # Mean-center-deconfound pipeline
│   ├── steering.py          # Activation steering implementation
│   ├── story_generator.py   # Story generation via Gemini
│   ├── model_loader.py      # Model loading utilities
│   └── judge.py             # LLM judge for shutdown trial classification
├── analysis/
│   ├── statistics.py        # Silhouette, RSA, AUC, permutation tests
│   ├── base_vs_instruct.py  # Cross-model comparison analysis
│   └── random_controls.py   # Random vector and shuffled label controls
├── figures/
│   ├── generate_all.py      # Generate all figures
│   ├── fig_pca.py           # PCA projections
│   ├── fig_umap.py          # UMAP embeddings
│   ├── fig_cosine_sim.py    # Cosine similarity matrices
│   ├── fig_heatmap.py       # Implicit detection heatmaps
│   ├── fig_intensity.py     # Intensity curves
│   ├── fig_preference.py    # Preference-Elo correlations
│   ├── fig_model_comparison.py  # Cross-model RSA and layer emergence
│   ├── fig_needs_cross.py   # Need-emotion cross-similarity
│   ├── fig_emotion_residual.py  # Emotion-residual analysis
│   ├── fig_shutdown.py      # Shutdown resistance figures
│   └── common.py            # Shared plotting utilities
├── data/
│   ├── figures/
│   │   ├── stream1/         # Emotion analysis figures (PDFs)
│   │   ├── stream2/         # Need analysis figures (PDFs)
│   │   └── robustness/      # Layer sweep and deconfound sweep data
│   ├── shutdown/
│   │   └── analysis_results.json
│   └── ...                  # Stories, activations, vectors (on HuggingFace)
├── scripts/                 # GPU job scripts and data upload utilities
├── paper.md                 # Full technical report
└── dataset_readme.md        # HuggingFace dataset card
```

## Data

All data (stories, activations, vectors, steering results, shutdown trials, figures) is hosted on HuggingFace:

**[LindaP/ai-emotions-v2](https://huggingface.co/datasets/LindaP/ai-emotions-v2)**

```python
from huggingface_hub import hf_hub_download, snapshot_download
import numpy as np, json

REPO = "LindaP/ai-emotions-v2"

# Load emotion vectors for one model
path = hf_hub_download(repo_id=REPO, filename="vectors/qwen-7b-base/emotion_vectors_layer18.npy", repo_type="dataset")
vectors = np.load(path)  # (171, 3584)

# Load labels
labels_path = hf_hub_download(repo_id=REPO, filename="vectors/qwen-7b-base/emotion_labels.json", repo_type="dataset")
labels = json.load(open(labels_path))

# Download all vectors for a model
snapshot_download(repo_id=REPO, repo_type="dataset", allow_patterns="vectors/llama-8b-base/*", local_dir="./data")

# Download shutdown trial data
snapshot_download(repo_id=REPO, repo_type="dataset", allow_patterns="shutdown/*", local_dir="./data")
```

## Citation

This work replicates and extends:

> Sofroniew, N., Kauvar, I., Saunders, W., Chen, R., Henighan, T., Hydrie, S., Citro, C., Pearce, A., Tarng, J., Gurnee, W., Batson, J., Zimmerman, S., Rivoire, K., Fish, K., Olah, C., & Lindsey, J. (2026). Emotion concepts and their function in a large language model. *Anthropic Transformer Circuits Thread*.

## License

MIT
