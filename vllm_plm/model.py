"""The vLLM model classes for PLM exports.

On the autoregressive path a PLM is Llama: the export's per-layer keys are the names vLLM's Llama loader consumes. What
``PLMForCausalLM`` adds is the auxiliary scheme, held as a real submodule so ``auxiliary_scheme.*`` loads next to
``lm_head``; ``RoundDecoder`` substitutes its vectors into the embedding before every forward.
"""

from collections.abc import Iterable

import torch
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader

from .auxiliary import build_auxiliary_scheme

AUXILIARY_PREFIX = "auxiliary_scheme."


class PLMBackboneForCausalLM(LlamaForCausalLM):
    """A PLM export decoded autoregressively: the auxiliary tensors are only read by parallel rounds, so they are dropped at load."""

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights((name, weight) for name, weight in weights if not name.startswith(AUXILIARY_PREFIX))


class PLMForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.auxiliary_scheme = build_auxiliary_scheme(vllm_config.model_config.hf_config, dtype=vllm_config.model_config.dtype)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["lm_head."] if self.config.tie_word_embeddings else []
        return AutoWeightsLoader(self, skip_prefixes=skip_prefixes).load_weights(weights)
