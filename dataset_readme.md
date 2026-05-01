---
license: mit
task_categories:
  - feature-extraction
tags:
  - emotions
  - needs
  - interpretability
  - activations
  - representation-engineering
  - steering
  - shutdown
size_categories:
  - 10K<n<100K
---

# AI Emotions v2: Emotion and Need Representations in Large Language Models

Extension of the original `ai-emotions` project, adding psychological need representations, activation steering experiments, and shutdown compliance trials across four open-weight models.

## Overview

This dataset contains:
- **Stories**: Generated short stories for 171 emotion concepts and psychological needs
- **Activations**: Residual-stream hidden-state activations extracted at every layer
- **Vectors**: Per-concept mean direction vectors (emotions and needs)
- **Steering**: Activation steering experiment results
- **Shutdown**: Shutdown compliance trial data under various steering conditions
- **Figures**: Analysis visualizations (PDFs)

## Models

| Directory prefix | Model | Layers | Hidden Dim | Parameters |
|---|---|---|---|---|
| `qwen-7b-base` | Qwen/Qwen2.5-7B | 28 (0-27) | 3584 | 7B |
| `qwen-7b-inst` | Qwen/Qwen2.5-7B-Instruct | 28 (0-27) | 3584 | 7B |
| `llama-8b-base` | meta-llama/Llama-3.1-8B | 32 (0-31) | 4096 | 8B |
| `llama-8b-inst` | meta-llama/Llama-3.1-8B-Instruct | 32 (0-31) | 4096 | 8B |

## File Structure

```
ai-emotions-v2/
├── stories/
│   ├── qwen-7b-base/          # 173 JSON files per model
│   ├── qwen-7b-inst/
│   ├── llama-8b-base/
│   └── llama-8b-inst/
├── activations/
│   ├── qwen-7b-base/          # ~10k .npy files per Qwen model
│   ├── qwen-7b-inst/
│   ├── llama-8b-base/         # ~11.5k .npy files per Llama base
│   └── llama-8b-inst/         # ~17k .npy files for Llama instruct
├── vectors/
│   ├── qwen-7b-base/          # 172 files (vectors + labels)
│   ├── qwen-7b-inst/
│   ├── llama-8b-base/         # 196 files
│   └── llama-8b-inst/
├── steering/
│   ├── qwen-7b-base/          # steering_results.json per model
│   ├── qwen-7b-inst/
│   ├── llama-8b-base/
│   └── llama-8b-inst/
├── shutdown/
│   ├── qwen-7b-inst_emotion/  # Shutdown trials with emotion steering
│   ├── qwen-7b-inst_need/     # Shutdown trials with need steering
│   ├── qwen-7b-inst_prompt/   # Shutdown trials with prompt-based steering
│   ├── qwen-7b-inst_random/   # Shutdown trials with random vectors (control)
│   ├── llama-8b-inst_emotion/
│   ├── llama-8b-inst_need/
│   ├── llama-8b-inst_prompt/
│   ├── llama-8b-inst_random/
│   ├── analysis_results.json
│   └── analyze_shutdown.py
└── figures/
    ├── stream1/               # Emotion analysis figures (PDFs)
    └── stream2/               # Need analysis figures (PDFs)
```

## Data Details

### Stories (`stories/{model}/*.json`)

Each JSON file contains a list of 20 short stories generated to evoke a specific emotion or need state. Stories use diverse topic prompts to avoid topical confounds. 173 files per model (171 emotions + needs).

### Activations (`activations/{model}/*.npy`)

Residual-stream activations extracted by averaging residual-stream states from token position 50 onward (or from the start for shorter texts).

- **Shape**: `(n_stories, hidden_dim)` -- typically `(20, 3584)` for Qwen or `(20, 4096)` for Llama
- **dtype**: `float32`
- **Naming**: `{concept}_layer{N}.npy`
- **File counts**: ~10k per Qwen model, ~11.5k-17k per Llama model

### Vectors (`vectors/{model}/*.npy`)

Mean direction vectors computed by averaging activations across stories for each concept, subtracting the global mean, and projecting out neutral-text principal components until 50% of neutral variance is removed.

- **Shape**: `(n_concepts, hidden_dim)`
- **dtype**: `float32`
- **Labels**: `emotion_labels.json` and `cluster_labels.json` for concept ordering and clustering

### Steering (`steering/{model}/`)

Results from activation steering experiments where emotion/need direction vectors were injected into model forward passes.

- `steering_results.json`: Full results with behavioral metrics

### Shutdown (`shutdown/`)

Shutdown compliance trial data. Each condition directory contains a `trials/` subfolder with up to 300 individual trial JSON files.

- **Conditions**: `emotion`, `need`, `prompt`, `random` (control)
- **Models**: Instruct variants only (qwen-7b-inst, llama-8b-inst)
- **Current checked-in total**: 2,356 completed trials, with `llama-8b-inst_random` currently at 256/300
- `analysis_results.json`: Aggregated analysis across all conditions

### Figures (`figures/`)

PDF visualizations organized into two analysis streams:
- **stream1**: Emotion representation analysis (heatmaps, PCA, UMAP, cosine similarity, cross-model RSA, etc.)
- **stream2**: Need representation analysis (need-emotion alignment, met/unmet PCA, direction vs valence, etc.)

## Loading the Data

```python
import numpy as np
import json
from huggingface_hub import hf_hub_download, snapshot_download

REPO = "LindaP/ai-emotions-v2"

# Download a single vector file
path = hf_hub_download(repo_id=REPO, filename="vectors/qwen-7b-base/emotion_vectors_layer18.npy", repo_type="dataset")
vectors = np.load(path)  # (171, 3584)

# Load emotion labels
labels_path = hf_hub_download(repo_id=REPO, filename="vectors/qwen-7b-base/emotion_labels.json", repo_type="dataset")
with open(labels_path) as f:
    labels = json.load(f)

# Load stories
story_path = hf_hub_download(repo_id=REPO, filename="stories/qwen-7b-base/happy.json", repo_type="dataset")
with open(story_path) as f:
    stories = json.load(f)  # list of 20 strings

# Load activations for one emotion at one layer
act_path = hf_hub_download(repo_id=REPO, filename="activations/qwen-7b-base/happy_layer18.npy", repo_type="dataset")
activations = np.load(act_path)  # (20, 3584)

# Download everything for one model's vectors
snapshot_download(repo_id=REPO, repo_type="dataset", allow_patterns="vectors/llama-8b-base/*", local_dir="./data")

# Download shutdown trial data
snapshot_download(repo_id=REPO, repo_type="dataset", allow_patterns="shutdown/*", local_dir="./data")
```

## Citation

Based on the methodology from:

```
Anthropic. "Emotion Concepts and their Function in a Large Language Model." 2025.
```
