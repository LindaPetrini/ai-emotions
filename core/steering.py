"""Activation steering via forward hooks."""

import numpy as np
import torch
from core.model_loader import get_target_layer


class SteeringHook:
    """Context manager for adding a steering vector to a model layer.

    Usage:
        with SteeringHook(model, layer_idx, vector, alpha=3.0):
            output = model.generate(...)
    """

    def __init__(self, model, layer_idx: int, vector: np.ndarray, alpha: float = 1.0):
        self.model = model
        self.layer_idx = layer_idx
        self.vector = torch.from_numpy(vector.astype(np.float32))
        self.alpha = alpha
        self._handle = None

    def _hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        steering = self.alpha * self.vector.to(hidden.device, dtype=hidden.dtype)
        hidden = hidden + steering
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def __enter__(self):
        layer = get_target_layer(self.model, self.layer_idx)
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *args):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec


def generate_with_steering(
    model, tokenizer, prompt: str, vector: np.ndarray,
    layer_idx: int, alpha: float = 3.0,
    max_new_tokens: int = 256, temperature: float = 1.0,
    use_chat_template: bool = False, system_prompt: str = None,
) -> str:
    """Generate text with activation steering applied.

    Args:
        use_chat_template: If True, wrap prompt in chat template (for instruct models).
        system_prompt: Optional system prompt (only with chat template).
    """
    if use_chat_template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with SteeringHook(model, layer_idx, vector, alpha):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

    new_tokens = output_ids[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
