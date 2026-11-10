"""Parallel decoding as captured CUDA graphs on vLLM's model, paged KV cache and FlexAttention, bypassing its scheduler.

One request at a time. A round decodes ``W = k + k*k`` query positions: ``k`` verify rows and ``k`` proposal blocks of
``k`` rows. Every piece of loop-carried state lives on the device, so a round is one graph replay. The prompt is
prefilled by captured graphs at fixed bucket widths, round 0 (one hypothesis) is its own graph, and the steady round's
two halves around the forward are ``torch.compile``d.

This drives vLLM internals directly: it builds FlexAttention metadata itself, installs the round mask on it, mutates
the builder's persistent buffers and enters ``set_forward_context``. Things that fail silently otherwise:

- the builder's persistent buffers are allocated under ``inference_mode`` and only accept updates from inside it;
- page 0 is vLLM's null block, so the static block table uses pages ``1..P``;
- the PLM mask mod is installed on the metadata and the ``BlockMask`` built before any forward, or the first forward
  rebuilds it with the default causal mask and drops the page list.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .mask import UNREACHABLE, RoundMask, round_layout

AUX_STREAM_LENGTH = 8192
EOS_WINDOW = 1024


def build_llm(model: str, *, dtype: str, max_model_len: int, gpu_memory_utilization: float, kv_cache_memory_bytes: int, block_size: int):
    """The LLM a RoundDecoder drives. The engine core must live in this process, since the decoder reaches into its model runner."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM

    return LLM(model=model, dtype=dtype, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization, enforce_eager=True,
               attention_backend="FLEX_ATTENTION", block_size=block_size, kv_cache_memory_bytes=kv_cache_memory_bytes, max_num_seqs=1)


@dataclass
class DecodeResult:
    token_ids: list[int]
    text: str
    seconds: float
    steady_rounds: int
    steady_ms: float
    steady_tokens: int

    @property
    def tokens_per_second(self) -> float:
        return len(self.token_ids) / self.seconds if self.seconds else 0.0


@dataclass
class ChunkGraph:
    """One prefill bucket: its FlexAttention metadata, the buffers its body writes, and its graph."""

    width: int
    metadata: object
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    kv_num_blocks: torch.Tensor
    rows: torch.Tensor
    sentinel_aux: torch.Tensor
    unreachable: torch.Tensor
    groups: int
    graph: torch.cuda.CUDAGraph | None = None


def nucleus_acceptance(logits: torch.Tensor, pending: torch.Tensor, nucleus: torch.Tensor) -> torch.Tensor:
    """Proposals inside the teacher's top-``nucleus`` mass; ``nucleus <= 0`` accepts none, which is exact match."""
    probs = torch.softmax(logits.float(), dim=-1)
    own = probs.gather(1, pending[:, None]).squeeze(1)
    return ((probs * (probs > own[:, None])).sum(-1) < nucleus) | ((own >= probs.amax(-1)) & (nucleus > 0))


def floor_acceptance(logits: torch.Tensor, pending: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
    """Proposals the teacher gives probability above ``epsilon``, in the log domain so no probability tensor is materialised."""
    scores = logits.float()
    own_logit = scores.gather(1, pending[:, None]).squeeze(1)
    return own_logit - torch.logsumexp(scores, dim=-1) > torch.log(epsilon)


class RoundDecoder:
    sequence_capacity = 1 << 17

    def __init__(self, llm, k: int, accept_rule: str, threshold: float | None, chunk_widths: tuple[int, ...]) -> None:
        """``accept_rule`` is ``nucleus`` (``threshold`` None: exact match) or ``floor``; the threshold is a device tensor."""
        runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        assert runner.vllm_config.scheduler_config.max_num_seqs == 1, "PLM decoding supports batch size 1 only"
        self.llm, self.k, self.width = llm, k, k + k * k
        self.model = runner.model
        self.scheme = self.model.auxiliary_scheme
        self.device = torch.device("cuda")
        self.mask = RoundMask(runner.vllm_config.scheduler_config.max_num_batched_tokens, self.device)
        self._attach_mask()
        self.builder = runner.attn_groups[0][0].get_metadata_builder()
        self.vllm_config = runner.vllm_config
        self.layer_names = list(runner.kv_cache_config.kv_cache_groups[0].layer_names)
        self.tokenizer = llm.get_tokenizer()
        self.block_size = self.builder.block_size
        self.vocab = int(self.model.config.text_vocab_size)
        self.num_aux_tokens = int(self.model.config.num_aux_tokens)
        self.embed_weight = self.model.model.embed_tokens.weight
        self.lm_head_weight = self.model.lm_head.weight[: self.vocab]
        self.sentinel = self.scheme.masked_value
        self.chunk_widths = chunk_widths
        self.aux_seed: int | None = None
        self._compiled_pre = self._compiled_post = None
        with torch.inference_mode():
            self._allocate_state()
            self.accept_rule = accept_rule
            self.threshold = torch.full((), -1.0 if threshold is None else float(threshold), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ construction

    def _allocate_state(self) -> None:
        dev, BS, W, K = self.device, self.block_size, self.width, self.k
        self.num_pages = (self.vllm_config.model_config.max_model_len + BS - 1) // BS
        self.pages = torch.arange(1, self.num_pages + 1, dtype=torch.int32, device=dev)
        self.block_table = self.pages[None, :]
        self.max_seq = self.num_pages * BS
        self.out = torch.full((self.max_seq,), -1, dtype=torch.long, device=dev)
        self.offset_t = torch.zeros((), dtype=torch.long, device=dev)
        self.base_t = torch.zeros((), dtype=torch.long, device=dev)
        self.anchor = torch.zeros(1, dtype=torch.long, device=dev)
        self.pending = torch.zeros(K - 1, dtype=torch.long, device=dev)
        # Per-round accepted counts in round order, written from inside the captured round through a device cursor.
        self.round_sequence = torch.zeros(self.sequence_capacity, dtype=torch.int64, device=dev)
        self.sequence_step = torch.zeros(1, dtype=torch.int64, device=dev)
        self._one = torch.ones(1, dtype=torch.int64, device=dev)
        self.ids_buf = torch.empty(W, dtype=torch.long, device=dev)
        self.rows = torch.arange(W, device=dev)
        self.kidx = torch.arange(K, device=dev)
        self.aux_buf = self.scheme.sample_prior((AUX_STREAM_LENGTH,), device=dev)

        self.md, self.seq_lens, self.slot_mapping = self._make_metadata(W, 0, self.max_seq)
        self._install_mask(self.md, self.seq_lens)
        self.groups = self.md.block_mask.kv_indices.shape[-2]
        self.kv_num_blocks = self.md.block_mask.kv_num_blocks
        self.physical_to_logical_saved = self.md.physical_to_logical.clone()

        self.round0_layout = round_layout([W - 1], dev)
        self.steady_layout = round_layout([K] * K, dev)

    def _attach_mask(self) -> None:
        for layer in self.model.model.layers:
            layer.self_attn.attn.logical_mask_mod = self.mask.view

    def _make_metadata(self, num_tokens: int, offset: int, max_seq: int):
        from vllm.v1.attention.backend import CommonAttentionMetadata

        dev, BS = self.device, self.block_size
        qsl = torch.tensor([0, num_tokens], dtype=torch.int32, device=dev)
        seq_lens = torch.tensor([max_seq], dtype=torch.int32, device=dev)
        slot_mapping = (torch.arange(num_tokens, device=dev) + offset + BS).to(torch.int64)
        common = CommonAttentionMetadata(query_start_loc=qsl, query_start_loc_cpu=qsl.cpu(), seq_lens=seq_lens, num_reqs=1,
                                         num_actual_tokens=num_tokens, max_query_len=num_tokens, max_seq_len=max_seq,
                                         block_table_tensor=self.block_table, slot_mapping=slot_mapping, causal=True,
                                         seq_lens_cpu_upper_bound=torch.tensor([max_seq], dtype=torch.int32))
        return self.builder.build(common_prefix_len=0, common_attn_metadata=common), seq_lens, slot_mapping

    def _install_mask(self, metadata, seq_lens: torch.Tensor) -> None:
        metadata.logical_mask_mod = self.mask.view
        metadata.mask_mod = metadata.get_mask_mod()
        metadata.block_mask = metadata._build_block_mask_direct()
        metadata.seq_lens = seq_lens

    def prepare(self) -> None:
        """Allocate the per-request buffers and capture the steady-round, prefill and round-0 graphs."""
        with torch.inference_mode():
            self._allocate_request_state()
            self.chunks = [self._make_chunk(width) for width in self._requested_widths()]
            self._restore_round_structure()
            self._capture_graphs()

    def _allocate_request_state(self) -> None:
        dev = self.device
        self.prompt_pad = torch.zeros(self.max_seq, dtype=torch.long, device=dev)
        self.prompt_len_t = torch.zeros((), dtype=torch.long, device=dev)
        self.first = torch.zeros(1, dtype=torch.long, device=dev)
        self.base_host = torch.zeros((), dtype=torch.long).pin_memory()
        self.tokens_host = torch.empty(EOS_WINDOW, dtype=torch.long).pin_memory()

    def _requested_widths(self) -> tuple[int, ...]:
        for width in self.chunk_widths:
            if width != self.width and width % self.block_size:
                raise ValueError(f"chunk width {width} must be a multiple of the KV block size {self.block_size}")
            if width > self.max_seq - self.width:
                raise ValueError(f"chunk width {width} leaves no room for the round inside max_model_len")
        return tuple(sorted(set(self.chunk_widths)))

    def _make_chunk(self, width: int) -> ChunkGraph:
        dev = self.device
        common = dict(width=width, rows=torch.arange(width, device=dev),
                      sentinel_aux=torch.full((width,), self.sentinel, dtype=torch.long, device=dev),
                      unreachable=torch.full((width,), UNREACHABLE, dtype=torch.int64, device=dev))
        if width == self.width:
            return ChunkGraph(metadata=self.md, seq_lens=self.seq_lens, slot_mapping=self.slot_mapping, kv_num_blocks=self.kv_num_blocks, groups=self.groups, **common)
        metadata, seq_lens, slot_mapping = self._make_metadata(width, 0, self.max_seq)
        self._install_mask(metadata, seq_lens)
        return ChunkGraph(metadata=metadata, seq_lens=seq_lens, slot_mapping=slot_mapping, kv_num_blocks=metadata.block_mask.kv_num_blocks,
                          groups=metadata.block_mask.kv_indices.shape[-2], **common)

    def _restore_round_structure(self) -> None:
        """Every metadata object wants every page visible; building one rewrites the shared persistent buffers, so restore all, widest last."""
        self.md.block_mask.kv_indices[0, 0, :, : self.num_pages] = self.pages[None, :].expand(self.groups, self.num_pages)
        self.builder.persistent_kv_indices[: self.groups, self.num_pages] = -1  # the kernel prefetches one past
        self.md.physical_to_logical.copy_(self.physical_to_logical_saved)
        for chunk in sorted(getattr(self, "chunks", []), key=lambda chunk: chunk.groups):
            if chunk.metadata is self.md:
                continue
            chunk.metadata.block_mask.kv_indices[0, 0, :, : self.num_pages] = self.pages[None, :].expand(chunk.groups, self.num_pages)
            self.builder.persistent_kv_indices[: chunk.groups, self.num_pages] = -1
            chunk.metadata.physical_to_logical.copy_(self.physical_to_logical_saved)

    def _capture_graphs(self) -> None:
        # The steady round is captured first: inductor specialises FlexAttention on the first query shape it compiles.
        self._fill_aux(0)
        self._install_tables(self.steady_layout)
        self.offset_t.zero_()
        self.base_t.fill_(1)
        self.graph = self._capture_body(self._steady_round)
        self.prompt_pad[:4].copy_(torch.arange(1, 5, device=self.device))
        self.prompt_len_t.fill_(4)
        for chunk in self.chunks:
            self.offset_t.zero_()
            chunk.graph = self._capture_body(lambda chunk=chunk: self._run_chunk(chunk), reset=self.offset_t.zero_)
        self.round0_graph = self._capture_body(self._round0)
        self.chunks[0].graph.replay()
        self.round0_graph.replay()
        torch.cuda.synchronize()

    def _capture_body(self, body, reset=None) -> torch.cuda.CUDAGraph:
        for _ in range(2):
            body()
            if reset is not None:
                reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        if reset is not None:
            reset()
        return graph

    # ------------------------------------------------------------------ pieces of a round

    def _forward(self, metadata, slot_mapping, num_tokens: int, positions, inputs_embeds):
        from vllm.forward_context import set_forward_context

        attn = {name: metadata for name in self.layer_names}
        slots = {name: slot_mapping for name in self.layer_names}
        with set_forward_context(attn, self.vllm_config, num_tokens=num_tokens, slot_mapping=slots):
            return self.model(input_ids=None, positions=positions, intermediate_tensors=None, inputs_embeds=inputs_embeds)

    def _decode_verify(self, logits, hidden, index):
        return self.scheme.decode_unmasked(logits, hidden, self.lm_head_weight, self.aux_buf[index])

    def _embed(self, input_ids, aux_values):
        aux_mask = (input_ids >= self.vocab) & (input_ids < self.vocab + self.num_aux_tokens)
        token_embeds = F.embedding(input_ids.masked_fill(aux_mask, 0).clamp(0, self.vocab - 1), self.embed_weight)
        vectors = self.scheme.embed(input_ids, aux_values, aux_mask)
        return torch.where(aux_mask[:, None], vectors.to(token_embeds.dtype), token_embeds)

    def _install_tables(self, layout: tuple) -> None:
        width = layout[2].numel()
        self.mask.depth[:width].copy_(layout[2])
        self.mask.own_start[:width].copy_(layout[3])

    def _round0(self) -> None:
        """Round 0 after the prefill: one hypothesis, so the whole round is a single proposal block."""
        K, W, BS, V = self.k, self.width, self.block_size, self.vocab
        offset_t, base_t, out = self.offset_t, self.base_t, self.out
        offset_t.copy_(self.prompt_len_t)
        out.fill_(-1)
        out[0] = self.first[0]
        base_t.fill_(1)
        self.anchor.copy_(self.first)
        self.pending.zero_()
        self._install_tables(self.round0_layout)
        rel_positions, rel_aux = self.round0_layout[:2]
        self.mask.offset.copy_(offset_t)
        ids = self.ids_buf
        ids[0] = self.anchor[0]
        ids[1:] = V
        aux_index = base_t + rel_aux
        aux_values = torch.where(rel_aux < 0, torch.full_like(aux_index, self.sentinel), self.aux_buf[aux_index.clamp(min=0)])
        embeds = self._embed(ids, aux_values)
        self.seq_lens.copy_((offset_t + W).to(torch.int32).view(1))
        self.md.decode_offset.copy_(offset_t.view(1).to(self.md.decode_offset.dtype))
        self.slot_mapping.copy_(self.rows + offset_t + BS)
        self.kv_num_blocks.copy_(((offset_t + W + BS - 1) // BS).to(self.kv_num_blocks.dtype).expand_as(self.kv_num_blocks))
        hidden = self._forward(self.md, self.slot_mapping, W, rel_positions + offset_t, embeds)
        logits = self.model.compute_logits(hidden)[:, :V]
        verify = self._decode_verify(logits[:1], hidden[:1], base_t + self.kidx[:1])
        hypotheses = logits[1:].argmax(dim=-1)
        num_correct = torch.zeros((), dtype=torch.int64, device=self.device)
        picked = verify.index_select(0, num_correct.clamp(max=0).view(1))
        slots = base_t + self.kidx
        out[slots] = torch.where(self.kidx < num_correct, verify[:1].expand(K), torch.where(self.kidx == num_correct, picked.expand(K), out[slots]))
        self.pending.copy_(hypotheses[1:K])
        self.anchor.copy_(picked)
        base_t.add_(num_correct + 1)
        offset_t.add_(num_correct + 1)
        self._install_tables(self.steady_layout)

    def _steady_round(self) -> None:
        K = self.k
        embeds, positions = self._compile(self._prologue, "_compiled_pre")(
            self.offset_t, self.base_t, self.anchor, self.pending, self.aux_buf, self.steady_layout[0], self.steady_layout[1], self.ids_buf,
            self.mask.offset, self.seq_lens, self.md.decode_offset, self.slot_mapping, self.kv_num_blocks, self.rows, self.embed_weight)
        hidden = self._forward(self.md, self.slot_mapping, self.width, positions, embeds)
        logits = self.model.compute_logits(hidden)[:, : self.vocab]
        # The assignment engine autotunes Triton kernels and compiles its own argmin, so the verify decode stays outside the traced epilogue.
        verify = self._decode_verify(logits[:K], hidden[:K], self.base_t + self.kidx)
        self._compile(self._epilogue, "_compiled_post")(logits, verify, self.pending, self.anchor, self.out, self.base_t, self.offset_t, self.kidx,
                                                         self.threshold, self._one, self.round_sequence, self.sequence_step)
        self.base_host.copy_(self.base_t, non_blocking=True)
        self.tokens_host.copy_(self.out[:EOS_WINDOW], non_blocking=True)

    def _compile(self, function, attribute: str):
        if getattr(self, attribute) is None:
            setattr(self, attribute, torch.compile(function, dynamic=False, fullgraph=True))
        return getattr(self, attribute)

    def _prologue(self, offset_t, base_t, anchor, pending, aux_buf, rel_positions, rel_aux, ids, mask_offset, seq_lens,
                  decode_offset, slot_mapping, kv_num_blocks, rows, embed_weight):
        K, W, BS, V = self.k, self.width, self.block_size, self.vocab
        mask_offset.copy_(offset_t)
        ids[0] = anchor[0]
        ids[1:K] = pending
        ids[K:] = V
        aux_index = base_t + rel_aux
        aux_values = torch.where(rel_aux < 0, torch.full_like(aux_index, self.sentinel), aux_buf[aux_index.clamp(min=0)])
        positions = rel_positions + offset_t
        aux_mask = (ids >= V) & (ids < V + self.num_aux_tokens)
        token_embeds = F.embedding(ids.masked_fill(aux_mask, 0).clamp(0, V - 1), embed_weight)
        vectors = self.scheme.embed(ids, aux_values, aux_mask)
        embeds = torch.where(aux_mask[:, None], vectors.to(token_embeds.dtype), token_embeds)
        seq_lens.copy_((offset_t + W).to(torch.int32).view(1))
        decode_offset.copy_(offset_t.view(1).to(decode_offset.dtype))
        slot_mapping.copy_(rows + offset_t + BS)
        kv_num_blocks.copy_(((offset_t + W + BS - 1) // BS).to(kv_num_blocks.dtype).expand_as(kv_num_blocks))
        return embeds, positions

    def _epilogue(self, logits, verify, pending, anchor, out, base_t, offset_t, kidx, threshold, one, round_sequence, sequence_step):
        K = self.k
        widened = floor_acceptance if self.accept_rule == "floor" else nucleus_acceptance
        accepted = (verify[:-1] == pending) | widened(logits[: K - 1], pending, threshold)
        num_correct = accepted.to(torch.int64).cumprod(0).sum()
        confirmed_head = torch.cat([pending, verify[-1:]])
        picked = verify.index_select(0, num_correct.clamp(max=K - 1).view(1))
        slots = base_t + kidx
        out[slots] = torch.where(kidx < num_correct, confirmed_head, torch.where(kidx == num_correct, picked.expand(K), out[slots]))
        round_sequence.index_copy_(0, sequence_step.clamp(max=self.sequence_capacity - 1), num_correct.view(1))
        sequence_step.add_(one)
        pending.copy_(logits[K:].argmax(dim=-1).view(K, K).index_select(0, num_correct.view(1))[0][1:K])
        anchor.copy_(picked)
        base_t.add_(num_correct + 1)
        offset_t.add_(num_correct + 1)

    # ------------------------------------------------------------------ the prefill

    def _run_chunk(self, chunk: ChunkGraph) -> None:
        """Causal forward over ``chunk.width`` prompt positions from ``offset_t``; rows past the prompt are rewritten before any round reads them."""
        width, offset_t = chunk.width, self.offset_t
        self.mask.depth[:width].copy_(chunk.rows)
        self.mask.own_start[:width].copy_(chunk.unreachable)
        self.mask.offset.copy_(offset_t)
        chunk.seq_lens.copy_((offset_t + width).to(torch.int32).view(1))
        chunk.metadata.decode_offset.copy_(offset_t.view(1).to(chunk.metadata.decode_offset.dtype))
        chunk.slot_mapping.copy_(chunk.rows + offset_t + self.block_size)
        blocks = ((offset_t + width + self.block_size - 1) // self.block_size).to(chunk.kv_num_blocks.dtype)
        chunk.kv_num_blocks.copy_(blocks.expand_as(chunk.kv_num_blocks))
        ids = self.prompt_pad.index_select(0, offset_t + chunk.rows)
        hidden = self._forward(chunk.metadata, chunk.slot_mapping, width, chunk.rows + offset_t, self._embed(ids, chunk.sentinel_aux))
        last_row = hidden.index_select(0, (self.prompt_len_t - 1 - offset_t).clamp(0, width - 1).view(1))
        logits = self.model.compute_logits(last_row)[:, : self.vocab]
        self.first.copy_(self._decode_verify(logits, last_row, slice(0, 1)))
        offset_t.add_(width)

    def _chunk_plan(self, length: int) -> list[ChunkGraph]:
        """The widest bucket while more than it remains, then the narrowest bucket that finishes the prompt."""
        widest = self.chunks[-1]
        plan, remaining = [], length
        while remaining > widest.width:
            plan.append(widest)
            remaining -= widest.width
        plan.append(next(chunk for chunk in self.chunks if chunk.width >= remaining))
        return plan

    def _prefill(self, prompt_ids: torch.Tensor, length: int) -> None:
        self.prompt_pad[:length].copy_(prompt_ids)
        self.prompt_len_t.fill_(length)
        self.offset_t.zero_()
        for chunk in self._chunk_plan(length):
            chunk.graph.replay()
        self.round0_graph.replay()

    # ------------------------------------------------------------------ the request

    def _fill_aux(self, seed: int) -> None:
        if self.aux_seed == seed:
            return
        generator = torch.Generator(device="cuda").manual_seed(seed)
        self.aux_buf.copy_(self.scheme.sample_prior((self.aux_buf.numel(),), device="cuda", generator=generator))
        self.aux_seed = seed

    def _replay_until_tokens(self, num_tokens: int) -> int:
        """One round in flight; stop once a completed round reports enough tokens through pinned memory."""
        replays = 0
        while True:
            self.graph.replay()
            replays += 1
            event = torch.cuda.Event()
            event.record()
            event.synchronize()
            if int(self.base_host) >= num_tokens:
                return replays

    def _replay_until_eos_or_tokens(self, num_tokens: int) -> int:
        """Stop at the token budget or at an end-of-text token among the tokens a round confirmed, whichever comes first."""
        end_of_text, limit = self.tokenizer.eos_token_id, min(num_tokens, EOS_WINDOW)
        replays, seen = 0, 2  # round 0 confirms exactly one token, so base is 2 when the loop starts
        while True:
            self.graph.replay()
            replays += 1
            event = torch.cuda.Event()
            event.record()
            event.synchronize()
            base = int(self.base_host)
            if end_of_text in self.tokens_host[seen: min(base, limit)].tolist() or base >= num_tokens:
                return replays
            seen = base

    def _steady_loop(self, num_tokens: int, stop_at_eos: bool) -> tuple[int, float]:
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
        replays = self._replay_until_eos_or_tokens(num_tokens) if stop_at_eos else self._replay_until_tokens(num_tokens)
        end_event.record()
        torch.cuda.synchronize()
        return replays, start_event.elapsed_time(end_event)

    @torch.inference_mode()
    def generate(self, prompt: str, num_tokens: int, *, seed: int = 0, stop_at_eos: bool = True) -> DecodeResult:
        torch.cuda.synchronize()
        started = time.perf_counter()
        prompt_ids = torch.tensor(self.tokenizer(prompt).input_ids, device=self.device)
        length = prompt_ids.numel()
        if length + num_tokens + 2 * self.width + self.block_size > self.max_seq:
            raise ValueError(f"prompt ({length}) + tokens ({num_tokens}) exceeds max_model_len {self.max_seq} for width {self.width}")
        self._fill_aux(seed)
        self._prefill(prompt_ids, length)
        replays, steady_ms = self._steady_loop(num_tokens, stop_at_eos)
        seconds = time.perf_counter() - started
        tokens = self.out[:num_tokens].tolist()
        eos = self.tokenizer.eos_token_id
        if stop_at_eos and eos is not None and eos in tokens:
            tokens = tokens[: tokens.index(eos) + 1]
        final_base = int(self.base_t.item())
        return DecodeResult(token_ids=tokens, text=self.tokenizer.decode(tokens), seconds=seconds, steady_rounds=replays,
                            steady_ms=steady_ms, steady_tokens=final_base - 2)
