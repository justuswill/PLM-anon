"""EAGLE-3 with EAGLE-2's dynamic draft tree, as captured CUDA graphs on vLLM 0.26's model, paged KV cache and FlexAttention.

One request at a time, bypassing vLLM's scheduler. The FlexAttention plumbing is a copy of vllm_plm/decoder.py's, kept
separate so this baseline shares no code path with the PLM decoder. A step is one graph replay:

1. verify: the target runs the W = total_token tree nodes in one forward. Node i sits at KV slot C+i (C = committed tokens)
   and RoPE position C+depth(i); a mask table lets it see the committed prefix and its tree ancestors.
2. accept: multi-candidate speculative sampling over siblings, as upstream EAGLE's evaluate_posterior: at each node the
   children are tried in draft-score order, child x accepted with probability p'(x), where p' is the target's top-k
   temperature-1 distribution with the already rejected siblings removed and renormalized; if none is accepted the bonus
   token is drawn from that residual. Lossless for any draft. Greedy accepts the child equal to the target's argmax.
3. compaction: the accepted path's KV moves to slots C..C+a; C grows by a+1.
4. draft update: the head consumes the a+1 accepted entries (target aux hidden states of the path, tokens shifted by one).
5. tree build: EAGLE-2's topK_genrate with fixed shapes: top_k children of the root, then ``depth`` levels expanding the
   top_k best frontier nodes by cumulative log-probability, then the W-1 best of all candidates become the next tree.

The prompt is prefilled by captured graphs at fixed bucket widths (target) and one eager pass (draft).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from eagle3.tree_head import DraftHead

EOS_WINDOW = 4096


def build_llm(model: str, *, max_model_len: int, block_size: int = 64, kv_cache_memory_bytes: int = 1 << 30):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # the decoder reaches into the engine core's model runner
    from vllm import LLM

    return LLM(model=model, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=0.5, enforce_eager=True,
               attention_backend="FLEX_ATTENTION", block_size=block_size, kv_cache_memory_bytes=kv_cache_memory_bytes, max_num_seqs=1)


class TreeMaskView:
    """Every attention layer's logical mask: the committed prefix, plus the table ``tree[row, key]`` within the current window."""

    def __init__(self, tables: "TreeMask") -> None:
        self.tables = tables

    def __call__(self, b, h, q_idx, kv_idx):
        t = self.tables
        row = (q_idx - t.offset).clamp(0, t.size - 1)
        key = kv_idx - t.offset
        return (key < 0) | ((key < t.size) & t.tree[row, key.clamp(0, t.size - 1)])


class TreeMask:
    def __init__(self, size: int, device: torch.device) -> None:
        self.size = size
        self.tree = torch.zeros(size, size, dtype=torch.bool, device=device)
        self.causal = torch.ones(size, size, dtype=torch.bool, device=device).tril()
        self.offset = torch.zeros((), dtype=torch.int64, device=device)
        self.view = TreeMaskView(self)


@dataclass
class Chunk:
    width: int
    metadata: object
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    kv_num_blocks: torch.Tensor
    groups: int
    rows: torch.Tensor
    graph: torch.cuda.CUDAGraph | None = None


@dataclass
class TreeResult:
    token_ids: list[int]
    text: str
    seconds: float
    steps: int
    steady_ms: float
    steady_tokens: int


class TreeDecoder:
    def __init__(self, llm, head_dir: str, *, total_token: int = 60, depth: int = 7, top_k: int = 10, sampler_top_k: int = 128,
                 greedy: bool = False, chunk_widths=tuple(range(64, 641, 64)), compile: bool = True) -> None:
        runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        self.llm, self.runner, self.model = llm, runner, runner.model
        self.vllm_config = runner.vllm_config
        self.device = torch.device("cuda")
        self.W, self.depth, self.top_k, self.sampler_top_k, self.greedy = total_token, depth, top_k, sampler_top_k, greedy
        self.compile = compile
        self.tokenizer = llm.get_tokenizer()
        self.vocab = len(self.tokenizer) if len(self.tokenizer) <= self.model.config.vocab_size else self.model.config.vocab_size
        self.vocab = min(self.vocab, 50257)
        self.eos = self.tokenizer.eos_token_id
        self.hidden_size = self.model.config.hidden_size
        self.model.model.aux_hidden_state_layers = DraftHead.aux_layers(head_dir)
        self.chunk_widths = tuple(sorted(chunk_widths))
        self.mask = TreeMask(max(max(self.chunk_widths), total_token), self.device)
        for layer in self.model.model.layers:
            layer.self_attn.attn.logical_mask_mod = self.mask.view
        self.builder = runner.attn_groups[0][0].get_metadata_builder()
        self.layer_names = list(runner.kv_cache_config.kv_cache_groups[0].layer_names)
        self.block_size = self.builder.block_size
        self.kv_caches = [layer.self_attn.attn.kv_cache[0] if isinstance(layer.self_attn.attn.kv_cache, list) else layer.self_attn.attn.kv_cache
                          for layer in self.model.model.layers]
        with torch.inference_mode():
            self._allocate()
            self.draft = DraftHead(head_dir, self.model.model.embed_tokens.weight, self.max_seq, depth * top_k, self.device)

    # ------------------------------------------------------------------ construction (vLLM FlexAttention plumbing as in RoundDecoder)

    def _allocate(self) -> None:
        dev, BS, W = self.device, self.block_size, self.W
        self.num_pages = (self.vllm_config.model_config.max_model_len + BS - 1) // BS
        self.pages = torch.arange(1, self.num_pages + 1, dtype=torch.int32, device=dev)
        self.block_table = self.pages[None, :]
        self.max_seq = self.num_pages * BS
        self.md, self.seq_lens, self.slot_mapping = self._make_metadata(W)
        self._install_mask(self.md, self.seq_lens)
        self.groups = self.md.block_mask.kv_indices.shape[-2]
        self.kv_num_blocks = self.md.block_mask.kv_num_blocks
        self.physical_to_logical_saved = self.md.physical_to_logical.clone()
        # step state
        self.C = torch.zeros((), dtype=torch.long, device=dev)          # committed tokens (prompt included); the root sits at C
        self.base = torch.zeros((), dtype=torch.long, device=dev)       # generated tokens written to out
        self.out = torch.full((self.max_seq,), -1, dtype=torch.long, device=dev)
        self.rows = torch.arange(W, device=dev)
        self.all_true = torch.ones(W, dtype=torch.bool, device=dev)
        self.tree_tokens = torch.zeros(W, dtype=torch.long, device=dev)
        self.tree_depth = torch.zeros(W, dtype=torch.long, device=dev)
        self.tree_anc = torch.zeros(W, W, dtype=torch.bool, device=dev)
        self.children = torch.full((W, self.top_k), -1, dtype=torch.long, device=dev)
        self.h_root = torch.zeros(1, self.hidden_size, dtype=torch.bfloat16, device=dev)
        self.bonus = torch.zeros(1, dtype=torch.long, device=dev)
        self.step_accepts = torch.zeros(1 << 15, dtype=torch.long, device=dev)
        self.step_count = torch.zeros((), dtype=torch.long, device=dev)
        self.base_host = torch.zeros((), dtype=torch.long).pin_memory()
        self.out_host = torch.empty(EOS_WINDOW, dtype=torch.long).pin_memory()
        self.prompt_pad = torch.zeros(self.max_seq, dtype=torch.long, device=dev)
        self.prompt_len = torch.zeros((), dtype=torch.long, device=dev)
        self.prompt_aux = torch.zeros(self.max_seq, 3 * self.hidden_size, dtype=torch.bfloat16, device=dev)
        self.prompt_last_logits = torch.zeros(1, self.vocab, dtype=torch.float32, device=dev)
        self.max_path = self.depth + 2   # root + up to depth+1 accepted draft tokens

    def _make_metadata(self, num_tokens: int):
        from vllm.v1.attention.backend import CommonAttentionMetadata

        dev, BS = self.device, self.block_size
        qsl = torch.tensor([0, num_tokens], dtype=torch.int32, device=dev)
        seq_lens = torch.tensor([self.max_seq], dtype=torch.int32, device=dev)
        slot_mapping = (torch.arange(num_tokens, device=dev) + BS).to(torch.int64)
        common = CommonAttentionMetadata(query_start_loc=qsl, query_start_loc_cpu=qsl.cpu(), seq_lens=seq_lens, num_reqs=1,
                                         num_actual_tokens=num_tokens, max_query_len=num_tokens, max_seq_len=self.max_seq,
                                         block_table_tensor=self.block_table, slot_mapping=slot_mapping, causal=True,
                                         seq_lens_cpu_upper_bound=torch.tensor([self.max_seq], dtype=torch.int32))
        return self.builder.build(common_prefix_len=0, common_attn_metadata=common), seq_lens, slot_mapping

    def _install_mask(self, metadata, seq_lens) -> None:
        metadata.logical_mask_mod = self.mask.view
        metadata.mask_mod = metadata.get_mask_mod()
        metadata.block_mask = metadata._build_block_mask_direct()
        metadata.seq_lens = seq_lens

    def _make_chunk(self, width: int) -> Chunk:
        md, seq_lens, slot_mapping = self._make_metadata(width)
        self._install_mask(md, seq_lens)
        return Chunk(width, md, seq_lens, slot_mapping, md.block_mask.kv_num_blocks, md.block_mask.kv_indices.shape[-2], torch.arange(width, device=self.device))

    def _restore_structure(self) -> None:
        """Building metadata rewrites the builder's shared persistent buffers: make every page visible again for all, widest last."""
        items = [(self.md, self.groups)] + [(chunk.metadata, chunk.groups) for chunk in getattr(self, "chunks", [])]
        for metadata, groups in sorted(items, key=lambda item: item[1]):
            metadata.block_mask.kv_indices[0, 0, :, : self.num_pages] = self.pages[None, :].expand(groups, self.num_pages)
            self.builder.persistent_kv_indices[:groups, self.num_pages] = -1
            metadata.physical_to_logical.copy_(self.physical_to_logical_saved)

    def _forward(self, metadata, slot_mapping, num_tokens: int, positions, input_ids):
        from vllm.forward_context import set_forward_context

        attn = {name: metadata for name in self.layer_names}
        slots = {name: slot_mapping for name in self.layer_names}
        with set_forward_context(attn, self.vllm_config, num_tokens=num_tokens, slot_mapping=slots):
            hidden, aux = self.model(input_ids=input_ids, positions=positions, intermediate_tensors=None)
        return hidden, torch.cat(aux, dim=-1)

    def _set_window(self, metadata, seq_lens, slot_mapping, kv_num_blocks, offset, width: int) -> None:
        BS = self.block_size
        self.mask.offset.copy_(offset)
        seq_lens.copy_((offset + width).to(torch.int32).view(1))
        metadata.decode_offset.copy_(offset.view(1).to(metadata.decode_offset.dtype))
        slot_mapping.copy_(torch.arange(width, device=self.device) + offset + BS)
        kv_num_blocks.copy_(((offset + width + BS - 1) // BS).to(kv_num_blocks.dtype).expand_as(kv_num_blocks))

    # ------------------------------------------------------------------ sampling

    def _target_probs(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        kth = logits.topk(self.sampler_top_k, dim=-1).values[:, -1:]
        return torch.softmax(logits.masked_fill(logits < kth, float("-inf")), dim=-1)

    def _sample(self, probs: torch.Tensor) -> torch.Tensor:
        if self.greedy:
            return probs.argmax(-1)
        return (probs / torch.empty_like(probs).exponential_()).argmax(-1)

    # ------------------------------------------------------------------ one step's pieces

    def _verify_and_accept(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Target forward over the tree; returns (path node indices [max_path], accepted draft count a, aux hidden [W, 3H])."""
        W, K, C = self.W, self.top_k, self.C
        self.mask.tree.zero_()
        self.mask.tree[:W, :W] = self.tree_anc
        self._set_window(self.md, self.seq_lens, self.slot_mapping, self.kv_num_blocks, C, W)
        hidden, aux = self._forward(self.md, self.slot_mapping, W, C + self.tree_depth, self.tree_tokens)
        logits = self.model.compute_logits(hidden)[:, : self.vocab]
        path, count = self._compiled("_accept")(logits)
        return path, count, aux

    def _accept(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        K = self.top_k
        valid = self.children >= 0
        child_tokens = self.tree_tokens[self.children.clamp(min=0)]
        if self.greedy:
            target = logits.argmax(-1)
            accepted = valid & (child_tokens == target[:, None])
            probs = None
        else:
            probs = self._target_probs(logits)
            p = probs.gather(1, child_tokens) * valid
            tried_before = torch.cumsum(p, dim=1) - p
            accept_prob = p / (1.0 - tried_before).clamp(min=1e-20)
            accepted = valid & (torch.rand_like(p) < accept_prob)
        first = torch.where(accepted.any(1), accepted.int().argmax(1), K)
        next_node = torch.where(first < K, self.children.gather(1, first.clamp(max=K - 1)[:, None])[:, 0], -1)
        # Indexing with a 0-d tensor would read it on the host; 1-element tensors and index_select keep the walk on the device.
        path = torch.zeros(self.max_path, dtype=torch.long, device=self.device)
        cur = torch.zeros(1, dtype=torch.long, device=self.device)
        alive = torch.ones(1, dtype=torch.bool, device=self.device)
        count = torch.zeros(1, dtype=torch.long, device=self.device)
        for s in range(1, self.max_path):
            nxt = next_node.index_select(0, cur)
            alive = alive & (nxt >= 0)
            cur = torch.where(alive, nxt, cur)
            path[s:s + 1] = torch.where(alive, cur, s)
            count = count + alive.long()
        last = cur
        if self.greedy:
            bonus = logits.index_select(0, last).argmax(-1)
        else:
            kids = self.children.index_select(0, last)[0]
            removed = torch.zeros(self.vocab, device=self.device)
            removed.index_put_((self.tree_tokens[kids.clamp(min=0)],), (kids >= 0).float(), accumulate=True)
            residual = probs.index_select(0, last)[0] * (removed == 0)
            residual = residual / residual.sum().clamp(min=1e-20)
            bonus = self._sample(residual[None])
        self.bonus.copy_(bonus)
        return path, count.reshape(())

    def _commit(self, path: torch.Tensor, count: torch.Tensor) -> None:
        """Write the accepted tokens and the bonus, and move the path's KV to slots C..C+count."""
        C, BS, P = self.C, self.block_size, self.max_path
        steps = torch.arange(P, device=self.device)
        emitted = steps + 1   # token s of this step, s = 1..P: accepted draft tokens, then the bonus
        tokens = torch.where(emitted <= count, self.tree_tokens[path[emitted.clamp(max=P - 1)]], torch.where(emitted == count + 1, self.bonus.expand(P), -1))
        slots = self.base + steps
        keep = tokens >= 0
        self.out.index_copy_(0, slots, torch.where(keep, tokens, self.out[slots]))
        src = C + path + BS
        dst = C + steps + BS
        for kv in self.kv_caches:
            moved = kv[src // BS, :, src % BS, :]
            kv[dst // BS, :, dst % BS, :] = moved
        self.step_accepts.index_copy_(0, self.step_count.view(1), (count + 1).view(1))
        self.step_count.add_(1)
        self.base.add_(count + 1)

    def _draft_update(self, path: torch.Tensor, count: torch.Tensor, aux: torch.Tensor) -> None:
        """Draft entries C..C+count: target aux of the path nodes, paired with the next token (the following path node, then the bonus)."""
        P, C, d = self.max_path, self.C, self.draft
        r = torch.arange(P, device=self.device)
        next_tokens = torch.where(r < count, self.tree_tokens[path[(r + 1).clamp(max=P - 1)]], self.bonus.expand(P))
        positions = C + r
        visible = d.key_index[None, :] <= positions[:, None]
        visible[:, d.max_entries:] = False
        h = d.layer(next_tokens, d.fc(aux[path]), positions, positions, visible)
        self.h_root.copy_(h.index_select(0, count.view(1)))
        self.C.add_(count + 1)

    def _build_tree(self) -> None:
        """EAGLE-2 topK_genrate with fixed shapes, from the root entry's draft output ``h_root``; the root token is ``bonus``."""
        d, K, W, dev = self.draft, self.top_k, self.W, self.device
        Cn = self.C   # draft entries 0..Cn-1 are committed; level i sits at position Cn + i
        lp = d.logprobs(self.h_root)[0]
        top = lp.topk(K)
        frontier_tokens, scores = top.indices, top.values
        frontier_hidden = self.h_root.expand(K, -1)
        frontier_gid = torch.arange(K, device=dev)
        anc = torch.zeros(K, d.scratch, dtype=torch.bool, device=dev)
        cand_tokens, cand_scores, cand_parent, cand_depth = [frontier_tokens], [scores], [torch.full((K,), -1, device=dev)], [torch.ones(K, dtype=torch.long, device=dev)]
        stable_visible = d.key_index[: d.max_entries][None, :] < Cn
        for level in range(self.depth):
            slots = d.max_entries + level * K + torch.arange(K, device=dev)
            anc[:, level * K: (level + 1) * K] = torch.eye(K, dtype=torch.bool, device=dev)
            visible = torch.cat([stable_visible.expand(K, -1), anc], dim=1)
            h = d.layer(frontier_tokens, frontier_hidden, (Cn + level).expand(K), slots, visible)
            top = d.logprobs(h).topk(K, dim=-1)
            cu = top.values + scores[:, None]
            best = cu.view(-1).topk(K)
            parents = best.indices // K
            offset = K + level * K * K
            cand_tokens.append(top.indices.view(-1))
            cand_scores.append(cu.view(-1))
            cand_parent.append(frontier_gid.repeat_interleave(K))
            cand_depth.append(torch.full((K * K,), level + 2, dtype=torch.long, device=dev))
            frontier_tokens = top.indices.view(-1)[best.indices]
            scores = best.values
            frontier_hidden = h[parents]
            frontier_gid = offset + best.indices
            anc = anc[parents]
        tokens, sc, parent, depth = torch.cat(cand_tokens), torch.cat(cand_scores), torch.cat(cand_parent), torch.cat(cand_depth)
        # A child's score never exceeds its parent's; the tiny depth penalty breaks exact ties so a chosen node's parent is chosen too.
        chosen = torch.sort((sc - 1e-6 * depth).topk(W - 1).indices).values
        node_parent_gid = parent[chosen]
        parent_node = torch.where(node_parent_gid < 0, 0, torch.searchsorted(chosen, node_parent_gid.clamp(min=0)) + 1)
        self.tree_tokens[:1].copy_(self.bonus)
        self.tree_tokens[1:] = tokens[chosen]
        self.tree_depth[:1].zero_()
        self.tree_depth[1:] = depth[chosen]
        parents_all = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), parent_node])
        anc_mat = torch.eye(W, dtype=torch.bool, device=dev)
        cur = self.rows
        for _ in range(self.depth + 1):
            cur = parents_all[cur]
            anc_mat.index_put_((self.rows, cur), self.all_true)
        self.tree_anc.copy_(anc_mat)
        # children of each node, in draft-score order
        child_score = torch.full((W, W), float("-inf"), device=dev)
        child_score[parents_all[1:], self.rows[1:]] = sc[chosen]
        kids = child_score.topk(K, dim=1)
        self.children.copy_(torch.where(kids.values > float("-inf"), kids.indices, -1))

    def _compiled(self, name: str):
        """torch.compile'd once per piece (static shapes); the compiled code is then recorded into the step's CUDA graph."""
        cache = self.__dict__.setdefault("_compiled_cache", {})
        if name not in cache:
            cache[name] = torch.compile(getattr(self, name), fullgraph=True, dynamic=False) if self.compile else getattr(self, name)
        return cache[name]

    def _step(self) -> None:
        path, count, aux = self._verify_and_accept()
        self._commit(path, count)
        self._compiled("_draft_update")(path, count, aux)
        self._compiled("_build_tree")()
        self.base_host.copy_(self.base, non_blocking=True)
        self.out_host.copy_(self.out[:EOS_WINDOW], non_blocking=True)

    # ------------------------------------------------------------------ prefill

    def _run_chunk(self, chunk: Chunk) -> None:
        width, offset = chunk.width, self.C
        self.mask.tree.copy_(self.mask.causal)
        self._set_window(chunk.metadata, chunk.seq_lens, chunk.slot_mapping, chunk.kv_num_blocks, offset, width)
        ids = self.prompt_pad.index_select(0, offset + chunk.rows)
        hidden, aux = self._forward(chunk.metadata, chunk.slot_mapping, width, chunk.rows + offset, ids)
        self.prompt_aux.index_copy_(0, offset + chunk.rows, aux)
        last_row = hidden.index_select(0, (self.prompt_len - 1 - offset).clamp(0, width - 1).view(1))
        self.prompt_last_logits.copy_(self.model.compute_logits(last_row)[:, : self.vocab].float())
        self.C.add_(width)

    def _chunk_plan(self, length: int) -> list[Chunk]:
        widest, plan, remaining = self.chunks[-1], [], length
        while remaining > widest.width:
            plan.append(widest)
            remaining -= widest.width
        plan.append(next(chunk for chunk in self.chunks if chunk.width >= remaining))
        return plan

    def _draft_prefill(self, length: int) -> None:
        """Eager: entries 0..n-1 pair the prompt's aux states with tokens 1..n (token n is the first sampled one)."""
        d, n = self.draft, length
        positions = torch.arange(n, device=self.device)
        tokens = torch.cat([self.prompt_pad[1:n], self.bonus])
        visible = d.key_index[None, :] <= positions[:, None]
        visible[:, d.max_entries:] = False
        h = d.layer(tokens, d.fc(self.prompt_aux[:n]), positions, positions, visible)
        self.h_root.copy_(h[n - 1:n])

    # ------------------------------------------------------------------ capture and generation

    def prepare(self) -> None:
        with torch.inference_mode():
            self.chunks = [self._make_chunk(width) for width in self.chunk_widths]
            self._restore_structure()
            self._dummy_state()
            self.graph = self._capture(self._step, self._dummy_state)
            for chunk in self.chunks:
                self.prompt_pad[:8].copy_(torch.arange(1, 9, device=self.device))
                self.prompt_len.fill_(8)
                chunk.graph = self._capture(lambda chunk=chunk: self._run_chunk(chunk), lambda: self.C.zero_())
            self.build_graph = self._capture(self._compiled("_build_tree"), self._dummy_state)
            torch.cuda.synchronize()

    def _dummy_state(self) -> None:
        self.C.fill_(8)
        self.base.zero_()
        self.step_count.zero_()
        self.tree_tokens.fill_(1)
        self.tree_depth.copy_(self.rows.clamp(max=self.depth + 1))
        self.tree_anc.copy_(torch.ones(self.W, self.W, dtype=torch.bool, device=self.device).tril())
        self.children.fill_(-1)

    def _capture(self, body, reset) -> torch.cuda.CUDAGraph:
        for _ in range(2):
            body()
            reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        reset()
        return graph

    @torch.inference_mode()
    def generate(self, prompt: str, max_tokens: int, seed: int = 0) -> TreeResult:
        torch.cuda.synchronize()
        started = time.perf_counter()
        torch.manual_seed(seed)
        ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        n = len(ids)
        if n + max_tokens + 2 * self.W + self.block_size > self.max_seq:
            raise ValueError(f"prompt {n} + {max_tokens} tokens exceeds max_model_len")
        self.prompt_pad[:n].copy_(torch.tensor(ids, device=self.device))
        self.prompt_len.fill_(n)
        self.C.zero_()
        for chunk in self._chunk_plan(n):
            chunk.graph.replay()
        self.C.fill_(n)
        probs = self._target_probs(self.prompt_last_logits)
        self.bonus.copy_(self._sample(probs))
        self.out.fill_(-1)
        self.out[0] = self.bonus[0]
        self.base.fill_(1)
        self.step_count.zero_()
        self._draft_prefill(n)
        self.build_graph.replay()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
        steps, seen = 0, 0
        eos_seen = self.eos is not None and int(self.out[0]) == self.eos
        while not eos_seen:
            self.graph.replay()
            steps += 1
            torch.cuda.current_stream().synchronize()
            base = int(self.base_host)
            window = self.out_host[seen: min(base, EOS_WINDOW)].tolist()
            if (self.eos is not None and self.eos in window) or base >= max_tokens:
                break
            seen = min(base, EOS_WINDOW)
        end_event.record()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        tokens = self.out[: int(self.base)].tolist()[:max_tokens]
        if self.eos is not None and self.eos in tokens:
            tokens = tokens[: tokens.index(self.eos) + 1]
        return TreeResult(tokens, self.tokenizer.decode(tokens, skip_special_tokens=True), seconds, steps, start_event.elapsed_time(end_event), int(self.base) - 1)
