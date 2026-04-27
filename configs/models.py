"""
Model configurations for the ai-emotions pipeline.

Defines model registry with architecture details for all supported models,
plus helper functions for resolving data directories.
"""

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Base directory (project root = parent of configs/)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelConfig:
    model_id: str        # HuggingFace model ID
    short_name: str      # e.g. "qwen-7b-base"
    n_layers: int
    hidden_dim: int
    is_instruct: bool
    analysis_layer: int  # ~2/3 depth
    layer_accessor: str  # e.g. "model.model.layers"


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "qwen-7b-base": ModelConfig(
        model_id="Qwen/Qwen2.5-7B",
        short_name="qwen-7b-base",
        n_layers=28,
        hidden_dim=3584,
        is_instruct=False,
        analysis_layer=18,
        layer_accessor="model.model.layers",
    ),
    "qwen-7b-inst": ModelConfig(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        short_name="qwen-7b-inst",
        n_layers=28,
        hidden_dim=3584,
        is_instruct=True,
        analysis_layer=18,
        layer_accessor="model.model.layers",
    ),
    "llama-8b-base": ModelConfig(
        model_id="meta-llama/Llama-3.1-8B",
        short_name="llama-8b-base",
        n_layers=32,
        hidden_dim=4096,
        is_instruct=False,
        analysis_layer=21,
        layer_accessor="model.model.layers",
    ),
    "llama-8b-inst": ModelConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        short_name="llama-8b-inst",
        n_layers=32,
        hidden_dim=4096,
        is_instruct=True,
        analysis_layer=21,
        layer_accessor="model.model.layers",
    ),
}

ALL_MODEL_NAMES = list(MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_model_config(name: str) -> ModelConfig:
    """Look up a model config by short name. Raises KeyError if not found."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {ALL_MODEL_NAMES}"
        )
    return MODEL_REGISTRY[name]


def get_data_dir(model_name: str) -> Path:
    """Return data/{model_name}/."""
    return BASE_DIR / "data" / model_name


def get_stories_dir(model_name: str) -> Path:
    """Return data/stories/{model_name}/."""
    return BASE_DIR / "data" / "stories" / model_name


def get_activations_dir(model_name: str) -> Path:
    """Return data/activations/{model_name}/."""
    return BASE_DIR / "data" / "activations" / model_name


def get_vectors_dir(model_name: str) -> Path:
    """Return data/vectors/{model_name}/."""
    return BASE_DIR / "data" / "vectors" / model_name


def get_figures_dir(stream: int) -> Path:
    """Return data/figures/stream{stream}/."""
    return BASE_DIR / "data" / "figures" / f"stream{stream}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Models registered: {len(MODEL_REGISTRY)}")
    for name, cfg in MODEL_REGISTRY.items():
        print(f"  {name}: {cfg.model_id} ({cfg.n_layers}L, {cfg.hidden_dim}d, "
              f"analysis@{cfg.analysis_layer}, instruct={cfg.is_instruct})")
    print(f"\nBASE_DIR: {BASE_DIR}")
    for name in ALL_MODEL_NAMES:
        print(f"  stories:     {get_stories_dir(name)}")
        print(f"  activations: {get_activations_dir(name)}")
        print(f"  vectors:     {get_vectors_dir(name)}")
    print(f"  figures s1:  {get_figures_dir(1)}")
    assert len(MODEL_REGISTRY) == 4, f"Expected 4 models, got {len(MODEL_REGISTRY)}"
    print("\nAll checks passed.")
