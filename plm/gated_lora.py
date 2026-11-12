"""Gated LoRA, unfused form: the adapter acts on a round's auxiliary rows only.

Text rows (the prompt and a round's verify rows) run the base weights, so the KV they write and the logits the verify
decode reads are the base model's own; the rows from ``rows_from`` onwards add ``s * B @ A``. A round lays its verify rows
first and its auxiliary rows after, so the gate is a single split point, and it is a Python int: constant per round shape,
so a captured round bakes it in. Every GEMM stays a plain ``dispatch_unquantized_gemm`` call, which keeps ``--mode fast``
routing the base projections through the tuned tiles.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import vllm.model_executor.layers.utils as vllm_utils

_rows_from: int | None = None

# The fused projections vLLM builds, in its own shard order, from the per-projection export names.
FUSED_PROJECTIONS = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "gate_up_proj": ("gate_proj", "up_proj")}
LORA_SUFFIXES = (".lora_a", ".lora_b")


@contextmanager
def rows_from(first_gated_row: int | None) -> Iterator[None]:
    global _rows_from
    previous, _rows_from = _rows_from, first_gated_row
    try:
        yield
    finally:
        _rows_from = previous


class GatedLoraMethod:
    """Wraps a layer's linear method: W over every row once, plus the rank-r correction on the gated rows."""

    def __init__(self, inner, lora_bs: list[torch.Tensor]) -> None:
        self.inner = inner
        self.lora_bs = lora_bs

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        first_gated_row = _rows_from
        if first_gated_row is None or first_gated_row >= x.shape[0]:
            return self.inner.apply(layer, x, bias)
        gemm = vllm_utils.dispatch_unquantized_gemm()
        output = gemm(layer, x, layer.weight, bias)
        hidden = gemm(layer, x[first_gated_row:], layer.lora_a)
        auxiliary, start, rank = output[first_gated_row:], 0, hidden.shape[1] // len(self.lora_bs)
        # One addmm per shard into its own output columns: a block-diagonal B would read the zeros between the shards.
        for index, lora_b in enumerate(self.lora_bs):
            auxiliary[:, start : start + lora_b.shape[0]].addmm_(hidden[:, index * rank : (index + 1) * rank], lora_b.t())
            start += lora_b.shape[0]
        return output


def _to_vllm_modules(tensors: dict[str, torch.Tensor], combine) -> dict[str, torch.Tensor]:
    """Export names (``...self_attn.q_proj``) to vLLM module names, combining a fused projection's shards."""
    combined: dict[str, torch.Tensor] = {}
    for module_name in sorted(tensors):
        parent, projection = module_name.rsplit(".", 1)
        for fused_projection, shards in FUSED_PROJECTIONS.items():
            if projection == shards[0]:
                combined[f"{parent}.{fused_projection}"] = combine([tensors[f"{parent}.{shard}"] for shard in shards])
        if not any(projection in shards for shards in FUSED_PROJECTIONS.values()):
            combined[module_name] = tensors[module_name]
    return combined


def _with_suffix(tensors: dict[str, torch.Tensor], suffix: str) -> dict[str, torch.Tensor]:
    return {name.removesuffix(suffix): tensor for name, tensor in tensors.items() if name.endswith(suffix)}


def _on_layer(layer: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(device=layer.weight.device, dtype=layer.weight.dtype)


def install(model: torch.nn.Module, tensors: dict[str, torch.Tensor]) -> None:
    """A fused projection's shards share its input, so their A's stack into one GEMM; each B maps only its own shard back."""
    modules = dict(model.named_modules())
    ups = _to_vllm_modules(_with_suffix(tensors, ".lora_b"), list)
    for module_name, down in _to_vllm_modules(_with_suffix(tensors, ".lora_a"), torch.cat).items():
        layer = modules[module_name]
        layer.register_buffer("lora_a", _on_layer(layer, down), persistent=False)
        shards = ups[module_name] if isinstance(ups[module_name], list) else [ups[module_name]]
        layer.quant_method = GatedLoraMethod(layer.quant_method, [_on_layer(layer, up) for up in shards])
