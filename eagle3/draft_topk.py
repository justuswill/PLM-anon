"""EAGLE-3 drafts sampled from the head's top-k at temperature 1 (``eagle3.generate --draft topk``).

vLLM's probabilistic drafting samples the head's full distribution q, and its standard rejection sampling then accepts
with probability sum(min(p, q)) against the target's p, here top-128 at temperature 1. Draft mass outside the target's
top-k is always rejected; truncating q to the head's own top-k moves it onto tokens the target can accept. Rejection
sampling stays exact for any q, provided the drafts are sampled from the same q it reads: the masked logits are what
vLLM's sampler writes to the draft-logits buffer the rejection step reads.

This replaces ``DraftModelSpeculator.sample_draft`` of vLLM's V2 model runner (vLLM 0.26), so it is a patch of vLLM internals.
It is installed by the eagle3 plugin in every vLLM process when EAGLE3_DRAFT_TOP_K is set.
"""

from __future__ import annotations

import os

import torch


def mask_to_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    threshold = logits.topk(top_k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def install() -> None:
    top_k = int(os.environ["EAGLE3_DRAFT_TOP_K"])
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    def sample_draft(self, hidden_states, positions, idx_mapping, temperature, seeds, draft_step, draft_logits):
        if draft_logits is None:
            return self._greedy_sample_draft(hidden_states)
        logits = mask_to_top_k(self.model.compute_logits(hidden_states), top_k)
        # Positions + 1 match the Gumbel noise of draft and target sampling, as in vLLM's own sample_draft.
        return gumbel_sample(logits, idx_mapping, temperature, seeds, positions + 1, apply_temperature=True, output_processed_logits=draft_logits,
                             output_processed_logits_col=draft_step, use_fp64=self.use_fp64_gumbel)

    DraftModelSpeculator.sample_draft = sample_draft
    from vllm.logger import init_logger

    # vLLM configures only its own logger namespace, so the message is logged under it.
    init_logger(f"vllm.{__name__}").info("EAGLE-3 drafts sample the head's top-%d (eagle3.draft_topk)", top_k)
