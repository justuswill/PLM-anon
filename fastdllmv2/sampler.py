"""Fast-dLLM v2 (FDv2) under vLLM: the block-diffusion baseline converted from ar-1.3B, decoded by its reference threshold sampler.

FDv2 decodes 32-token canvases at 32-aligned positions, in sub-blocks of 8 taken in order. Each pass samples every masked
position of the active sub-block and unmasks those whose probability is above the threshold, plus the most confident one.
Position i is read off row i-1 (token shift). Once a canvas is full, a commit pass writes its keys and values, which later
canvases attend to, and its last row gives the next canvas's first token.

Every pass is bidirectional within what it is sent. With ``max_num_batched_tokens=32`` the prompt is prefilled in aligned
32-token chunks, which is block-causal as FDv2 trains its clean stream. The driver (``bench.generate --fdv2``) sends the prompt
up to its last 32-aligned boundary and the rest as ``extra_args["fdv2_tail"]``, which opens the first canvas where the
reference puts it.

An FDv2 export is a stock Llama. ``hf_overrides`` add ``canvas_length``, which makes vLLM treat it as a block-diffusion model,
``mask_token_id`` and the architecture below. The sampler is built before any request arrives, so it reads its contract from
the environment: FDV2_THRESHOLD, FDV2_TEMPERATURE, FDV2_TOP_K (0 for the full vocabulary), FDV2_SUB_BLOCK.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(f"vllm.{__name__}")   # vLLM configures only its own logger namespace

TEXT_VOCAB_SIZE = 50257
EOD_TOKEN_ID = 50256
# Batch sizes whose denoise pass is captured as a CUDA graph; a larger batch amortises the launches and runs eagerly.
MAX_GRAPH_REQUESTS = 16


class CanvasStates:
    """Each request's canvas, indexed by the runner's slot."""

    def __init__(self, max_num_reqs: int, canvas_length: int, mask_token_id: int, device: torch.device) -> None:
        self.mask_token_id = mask_token_id
        self.device = device
        self.canvas = torch.full((max_num_reqs, canvas_length), mask_token_id, dtype=torch.int64, device=device)


class Fdv2ModelState(DefaultModelState):
    """Turns vLLM's speculative data path into FDv2's denoising loop: every pass attends bidirectionally within what it is sent."""

    def __init__(self, vllm_config, model, encoder_cache, device) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model_config.hf_config
        self.canvas_length = getattr(config, "canvas_length", None)
        # A denoising pass emits no token of its own: it refines the canvas in place.
        self.num_new_sampled_tokens_per_step = 0 if self.is_diffusion else 1
        if self.is_diffusion:
            self.states = CanvasStates(self.max_num_reqs, self.canvas_length, config.mask_token_id, device)
            # Updated in place, so a FULL cudagraph replay reads the current value at the captured address.
            self._causal = torch.zeros(self.max_num_reqs, dtype=torch.bool, device=device)

    @property
    def is_diffusion(self) -> bool:
        return self.canvas_length is not None

    def remove_request(self, req_id: str) -> None:
        return None

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        if not self.is_diffusion:
            return None
        threshold = float(os.environ.get("FDV2_THRESHOLD", "1"))
        temperature = float(os.environ.get("FDV2_TEMPERATURE", "1"))
        top_k = int(os.environ.get("FDV2_TOP_K", "0"))
        sub_block = int(os.environ.get("FDV2_SUB_BLOCK", "8"))
        return Fdv2Sampler(sampler, self.states, threshold, temperature, top_k, sub_block), None

    def prepare_attn(self, input_batch: InputBatch, cudagraph_mode: CUDAGraphMode, block_tables: tuple[torch.Tensor, ...], slot_mappings: torch.Tensor,
                     attn_groups: list[list[AttentionGroup]], kv_cache_config: KVCacheConfig, for_capture: bool = False) -> dict[str, Any]:
        if not self.is_diffusion:
            return super().prepare_attn(input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config, for_capture)
        full = cudagraph_mode == CUDAGraphMode.FULL
        num_reqs = input_batch.num_reqs_after_padding if full else input_batch.num_reqs
        num_tokens = input_batch.num_tokens_after_padding if full else input_batch.num_tokens
        # A per-request causal tensor is what TRITON_ATTN accepts; writing keys and values follows the slot mapping, not this flag.
        self._causal[:num_reqs] = False
        return build_attn_metadata(attn_groups=attn_groups, num_reqs=num_reqs, num_tokens=num_tokens, query_start_loc_gpu=input_batch.query_start_loc,
                                   query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np), max_query_len=input_batch.num_scheduled_tokens.max().item(),
                                   seq_lens=input_batch.seq_lens, max_seq_len=self.max_model_len, block_tables=block_tables, slot_mappings=slot_mappings,
                                   kv_cache_config=kv_cache_config, causal=self._causal[:num_reqs])


class Fdv2ForBlockDiffusion(LlamaForCausalLM):
    @staticmethod
    def get_model_state_cls():
        return Fdv2ModelState


class Fdv2Sampler:
    """Every decision is a tensor select, so a pass never waits on the host; after its first eager run, a batch size's pass is
    captured as a CUDA graph and replayed, which removes the launch cost of its ~100 small kernels (random draws advance on every replay)."""

    def __init__(self, sampler, states: CanvasStates, threshold: float, temperature: float, top_k: int, sub_block: int) -> None:
        self.req_states = sampler.req_states
        self.sampling_states = sampler.sampling_states
        self.states = states
        self.threshold = threshold
        self.temperature = temperature
        self.top_k = top_k
        self.sub_block = sub_block
        max_reqs, self.canvas_length = states.canvas.shape
        device = states.device
        self.answer_from = torch.zeros(max_reqs, dtype=torch.int64, device=device)
        self.committing = torch.zeros(max_reqs, dtype=torch.bool, device=device)
        self.tails: dict[int, list[int]] = {}
        self.positions = torch.arange(self.canvas_length, device=device)
        self.sub_positions = torch.arange(sub_block, device=device)
        self._sampled = torch.zeros(sampler.req_states.draft_tokens.shape[0], self.canvas_length, dtype=torch.int64, device=device)
        self._num_sampled = torch.zeros(self._sampled.shape[0], dtype=torch.int32, device=device)
        # Forward passes, read by the driver through collective_rpc: an engine step is not one forward.
        self.prefill_passes = 0
        self.decode_passes = 0
        self.graphs: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.cuda.CUDAGraph]] = {}
        self.uncapturable: set[tuple] = set()

    def add_request(self, req_idx: int, prompt_len: int, sampling_params) -> None:
        self.sampling_states.add_request(req_idx, sampling_params)
        self.tails[req_idx] = list((getattr(sampling_params, "extra_args", None) or {}).get("fdv2_tail", []))

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()

    @property
    def penalties_state(self):
        return SimpleNamespace(output_bin_counts=None)

    def sample(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(tokens, their probabilities) per row, over the text vocabulary; with a top-k, probabilities are renormalized over it."""
        logits = logits[..., :TEXT_VOCAB_SIZE].float()
        if self.temperature == 0:
            probabilities = torch.softmax(logits, dim=-1)
            tokens = probabilities.argmax(dim=-1)
        else:
            logits = logits / self.temperature
            if self.top_k:
                logits = logits.masked_fill(logits < logits.topk(self.top_k, dim=-1).values[..., -1:], float("-inf"))
            probabilities = torch.softmax(logits, dim=-1)
            # An exponential race draws from `probabilities` exactly; torch.multinomial validates on the host.
            tokens = (probabilities / torch.empty_like(probabilities).exponential_()).argmax(dim=-1)
        return tokens, probabilities.gather(-1, tokens[..., None])[..., 0]

    def first_index(self, flags: torch.Tensor) -> torch.Tensor:
        """Per row, the first True column, or the canvas length if there is none."""
        return torch.where(flags.any(dim=1), flags.int().argmax(dim=1), self.canvas_length)

    def first_stop(self, canvas: torch.Tensor, answer_from: torch.Tensor) -> torch.Tensor:
        """Per row, the first EOD of the answer with nothing masked before it, or the canvas length."""
        answer = self.positions >= answer_from[:, None]
        first_eod = self.first_index((canvas == EOD_TOKEN_ID) & answer)
        first_mask = self.first_index((canvas == self.states.mask_token_id) & answer)
        return torch.where(first_eod < first_mask, first_eod, self.canvas_length)

    def unmask_active_sub_block(self, canvas: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        masked = canvas == self.states.mask_token_id
        start = self.first_index(masked) % self.canvas_length // self.sub_block * self.sub_block
        window = start[:, None] + self.sub_positions
        shifted = (window - 1).clamp(min=0)
        tokens, confidence = self.sample(rows[torch.arange(rows.shape[0], device=rows.device)[:, None], shifted])
        active = masked.gather(1, window)
        confidence = torch.where(active, confidence, -1.0)
        most_confident = self.sub_positions == confidence.argmax(dim=1, keepdim=True)
        chosen = ((confidence > self.threshold) | most_confident) & active
        return canvas.scatter(1, window, torch.where(chosen, tokens, canvas.gather(1, window)))

    def denoise(self, canvas: torch.Tensor, rows: torch.Tensor, answer_from: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(the canvas after this pass, its stop index or the canvas length); a canvas already stopped is left as it is."""
        stop = self.first_stop(canvas, answer_from)
        stopped = (stop < self.canvas_length)[:, None]
        denoised = torch.where(stopped, canvas, self.unmask_active_sub_block(canvas, rows))
        return denoised, torch.where(stopped[:, 0], stop, self.first_stop(denoised, answer_from))

    def opened_canvases(self, first_tokens: torch.Tensor) -> torch.Tensor:
        canvas = torch.full((first_tokens.shape[0], self.canvas_length), self.states.mask_token_id, dtype=torch.int64, device=first_tokens.device)
        canvas[:, 0] = first_tokens
        return canvas

    def decode(self, logits: torch.Tensor, slots: torch.Tensor, num_reqs: int) -> None:
        """One denoise or commit pass over `slots`: the canvases, the drafts the next forward reads, and _sampled/_num_sampled."""
        rows = logits.view(num_reqs, self.canvas_length, -1)
        canvas, committing, answer_from = self.states.canvas[slots], self.committing[slots], self.answer_from[slots]
        denoised, stop = self.denoise(canvas, rows, answer_from)
        stopped = (stop < self.canvas_length) & ~committing
        # A committing pass wrote the finished canvas's keys and values; its last row opens the next canvas.
        next_canvas = torch.where(committing[:, None], self.opened_canvases(self.sample(rows[:, -1])[0]), denoised)
        through_stop = torch.where(self.positions > stop[:, None], EOD_TOKEN_ID, denoised)
        emitted = torch.where(committing[:, None], canvas, through_stop)
        emitting = committing | stopped
        self.states.canvas[slots] = next_canvas
        self.answer_from[slots] = torch.where(committing, 0, answer_from)
        self.committing[slots] = ~emitting & ~(denoised == self.states.mask_token_id).any(dim=1)
        self.req_states.draft_tokens[slots, : self.canvas_length] = next_canvas.to(self.req_states.draft_tokens.dtype)
        self._sampled[:num_reqs].copy_(torch.where(emitting[:, None], emitted, 0))
        self._num_sampled[:num_reqs].copy_(torch.where(emitting, self.canvas_length, 0))

    def run_decode(self, logits: torch.Tensor, idx_mapping: torch.Tensor, num_reqs: int) -> None:
        key = (num_reqs, tuple(logits.shape), logits.dtype)
        if key in self.graphs:
            static_logits, static_slots, graph = self.graphs[key]
            static_logits.copy_(logits)
            static_slots.copy_(idx_mapping)
            graph.replay()
            return
        self.decode(logits, idx_mapping.long(), num_reqs)
        if num_reqs <= MAX_GRAPH_REQUESTS and key not in self.uncapturable:
            # The eager pass above was the warm-up; capturing records kernels without running them.
            static_logits, static_slots = torch.empty_like(logits), torch.empty(num_reqs, dtype=torch.int64, device=logits.device)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                    self.decode(static_logits, static_slots, num_reqs)
            except Exception as error:  # noqa: BLE001 -- an eager pass is always correct, only slower
                logger.warning("FDv2 sampler: capturing the %d-request pass failed (%s); it runs eagerly", num_reqs, error)
                self.uncapturable.add(key)
                return
            self.graphs[key] = (static_logits, static_slots, graph)

    def __call__(self, logits: torch.Tensor, input_batch, draft_logits=None) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        if input_batch.num_draft_tokens == 0:
            self.prefill_passes += 1
            return self.begin(logits, input_batch, input_batch.idx_mapping_np[:num_reqs].tolist())
        self.decode_passes += 1
        self.run_decode(logits, input_batch.idx_mapping[:num_reqs], num_reqs)
        sampled, num_sampled = self._sampled[:num_reqs], self._num_sampled[:num_reqs]
        query_lens = input_batch.query_start_loc[1 : num_reqs + 1] - input_batch.query_start_loc[:num_reqs]
        return SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None, num_nans=None, num_sampled=num_sampled, num_rejected=query_lens - num_sampled)

    def open_canvas(self, slot: int, known: torch.Tensor, answer_from: int) -> None:
        canvas = self.states.canvas[slot]
        canvas.fill_(self.states.mask_token_id)
        canvas[: known.shape[0]] = known
        self.answer_from[slot] = answer_from
        self.committing[slot] = known.shape[0] == self.canvas_length
        self.req_states.draft_tokens[slot, : self.canvas_length] = canvas

    def begin(self, logits: torch.Tensor, input_batch, slots: list[int]) -> SamplerOutput:
        """Open the first canvas of every request whose prompt finished prefilling this step."""
        num_reqs = len(slots)
        done = input_batch.num_computed_prefill_tokens_np[:num_reqs] + input_batch.num_scheduled_tokens[:num_reqs] >= input_batch.prefill_len_np[:num_reqs]
        for index in np.nonzero(done)[0].tolist():
            slot = slots[index]
            tail = self.tails.get(slot, [])
            # A prompt shorter than one block has no aligned prefix to prefill: its canvas then opens on the token its last
            # position predicts, one block grid later than the reference's.
            known = torch.tensor(tail, dtype=torch.int64, device=logits.device) if tail else self.sample(logits[index : index + 1])[0]
            self.open_canvas(slot, known, len(tail))
        sampled = self._sampled[:num_reqs, :1]
        sampled.zero_()
        num_sampled = self._num_sampled[:num_reqs]
        num_sampled.zero_()
        return SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None, num_nans=None, num_sampled=num_sampled, num_rejected=num_sampled)
