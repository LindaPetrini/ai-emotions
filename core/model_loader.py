"""Architecture-agnostic model loading for Qwen and Llama families."""

import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from configs.models import ModelConfig, get_model_config


def load_model(model_name: str, device: str = "auto", dtype=torch.float16) -> tuple:
    """
    Load model and tokenizer.

    Args:
        model_name: Key in MODEL_REGISTRY (e.g. "qwen-7b-base")
        device: "auto", "cuda", "cpu"
        dtype: torch.float16 or torch.bfloat16

    Returns:
        (model, tokenizer, config: ModelConfig)
    """
    cfg = get_model_config(model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, cfg


def unload_model(model, tokenizer):
    """Free GPU/CPU memory."""
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_layers(model):
    """Get the decoder layers list from any supported architecture."""
    return model.model.layers


def get_target_layer(model, layer_idx: int):
    """Get a specific decoder layer module."""
    return model.model.layers[layer_idx]
