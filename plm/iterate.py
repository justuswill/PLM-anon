"""Proportional nearest-neighbour assignment of the auxiliary vocabulary to a token shortlist: the ``iterate`` backend's engine.

For each position, every one of ``A`` auxiliary vectors is handed to one of the ``M`` likeliest tokens so that each goes to
a token it is close to and the number per token follows ``softmax(logits)``. The answer is always a plain argmin
``sigma(a) = argmin_m (C[m, a] + b[m])`` under one additive factor per token; ``b`` is the transportation LP's dual,
found by a quota prior, condensing each item to its ``k`` cheapest candidates, and annealed Sinkhorn sweeps, twice.

The three kernels are ``triton.autotune``d on first use of each problem shape. ``load`` is accumulated with atomics, so
the fitted factors are not bitwise reproducible run to run; the assignment is what has to be stable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor

# An empty token must not move its factor by more than this many cost spreads in one sweep, or the anneal freezes a collapse.
STEP_LIMIT = 0.25
LOAD_FLOOR = 1e-2
# Out-of-reach factor for a token entitled to nothing, in cost spreads.
BLOCKED = 1e3
EMPTY = -1e30  # finite stand-in for -inf inside the streaming scan

# Launch configurations are chosen per problem shape on first use.  TILE is
# not among them: it also sets how wide the pool the top-k runs over and how
# many columns `tile_max` has, so it is a structural choice, not a launch one.
_SCAN_CONFIGS = [triton.Config({"GROUP": g}, num_warps=w)
                 for g in (8, 16, 32) for w in (4, 8, 16)]
_CONDENSE_CONFIGS = [triton.Config({"BLOCK": b}, num_warps=w)
                     for b in (16, 32, 64, 128) for w in (1, 2, 4)]


def _fits_in_registers(configs, named_args, **meta):
    """Drop condense configurations whose cost tile cannot stay in registers.

    The kernel holds an ``(M, BLOCK)`` float32 tile, spread over ``32 *
    num_warps`` threads.  Once that exceeds ~128 registers per thread it
    spills and the kernel falls off a cliff -- on an RTX A6000 the pruned
    configurations measured 23 ms, 26 ms and 141 ms against 6 ms for the
    survivors.  This is a property of the tile, not of the GPU, so it is safe
    to apply before benchmarking, and it removes a third of the compilations
    the first call would otherwise pay for.
    """
    targets = meta.get("M", named_args.get("M"))
    keep = [c for c in configs
            if targets * c.kwargs["BLOCK"] <= 128 * 32 * c.num_warps]
    return keep or configs


@triton.autotune(configs=_SCAN_CONFIGS, key=["n_vocab", "TILE"], warmup=10, rep=40)
@triton.jit
def _scan_kernel(logits, tile_max, running_max, running_sum, n_vocab, stride_n,
                 TILE: tl.constexpr, GROUP: tl.constexpr, SENTINEL: tl.constexpr):
    # int64 throughout: N * V is 2.5e9 elements and int32 pointer arithmetic
    # silently wraps past 2.1e9.
    row = tl.program_id(0).to(tl.int64)
    n_tiles = tl.cdiv(n_vocab, TILE)
    largest, total = SENTINEL, 0.0

    for start in range(0, n_vocab, TILE * GROUP):
        offs = start + tl.arange(0, TILE * GROUP)
        block = tl.load(logits + row * stride_n.to(tl.int64) + offs,
                        mask=offs < n_vocab, other=float("-inf")).to(tl.float32)
        block = tl.maximum(block, SENTINEL)
        # Online softmax: rescale the accumulator whenever the running
        # maximum moves, so one pass yields the normaliser exactly.
        peak = tl.maximum(largest, tl.max(block))
        total = total * tl.exp(largest - peak) + tl.sum(tl.exp(block - peak))
        largest = peak

        where = start // TILE + tl.arange(0, GROUP)
        tl.store(tile_max + row * n_tiles + where,
                 tl.max(tl.reshape(block, (GROUP, TILE)), axis=1), mask=where < n_tiles)

    tl.store(running_max + row, largest)
    tl.store(running_sum + row, total)


@triton.autotune(configs=_CONDENSE_CONFIGS, key=["n_items", "M", "K"],
                 prune_configs_by={"early_config_prune": _fits_in_registers},
                 warmup=10, rep=40)
@triton.jit
def _condense_kernel(table, tokens, prior, out_key, n_items,
                     stride_tn, stride_pn, stride_on,
                     M: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    problem = tl.program_id(0)
    item = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = item < n_items
    target = tl.arange(0, M)

    # The problem's whole cost matrix for these BLOCK items, in registers.
    # This (M, BLOCK) tile is the only place the cost matrix ever exists.
    token = tl.load(tokens + problem * stride_tn + target).to(tl.int64)
    offset = tl.load(prior + problem * stride_pn + target)
    tile = tl.load(table + token[:, None] * n_items + item[None, :],
                   mask=mask[None, :], other=float("inf")).to(tl.float32) + offset[:, None]

    bits = tile.to(tl.int32, bitcast=True)
    key = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF) & ~0x7F | target[:, None]
    for slot in tl.static_range(K):
        best = tl.min(key, axis=0)
        key = tl.where(key == best[None, :], 0x7FFFFFFF, key)
        tl.store(out_key + problem * stride_on + slot * n_items + item, best, mask=mask)


@triton.jit
def _fit_kernel(key_ptr, counts_ptr, offset_ptr, unit_ptr, out_ptr, load_ptr, corr_ptr,
                stride_key_n, stride_m_n,
                epsilon, anneal, relax, load_floor, step_limit,
                M: tl.constexpr, K: tl.constexpr, A: tl.constexpr, ITERATIONS: tl.constexpr):
    problem = tl.program_id(0)
    target_row = tl.arange(0, M)
    corr_base = corr_ptr + problem * stride_m_n + target_row
    load_base = load_ptr + problem * stride_m_n + target_row

    counts = tl.load(counts_ptr + problem * stride_m_n + target_row)
    unit = tl.load(unit_ptr + problem)
    share = tl.log(tl.maximum(counts.to(tl.float32), 1e-9))
    temperature = epsilon * unit
    bound = step_limit * unit
    correction = tl.zeros([M], tl.float32)
    tl.store(corr_base, correction, cache_modifier=".cg")

    # The condensed problem never changes during the anneal, so it is decoded once and the
    # sweep reduces to one gather of the factors plus the softmax.
    packed = tl.load(key_ptr + problem * stride_key_n
                     + tl.arange(0, K)[:, None] * A + tl.arange(0, A)[None, :])
    target = packed & 0x7F
    carried = packed & ~0x7F
    base_cost = tl.where(carried >= 0, carried, carried ^ 0x7FFFFFFF).to(tl.float32, bitcast=True)
    scatter = load_ptr + problem * stride_m_n + target
    gather = corr_ptr + problem * stride_m_n + target
    tl.debug_barrier()

    for _ in range(ITERATIONS):
        tl.store(load_base, tl.zeros([M], tl.float32), cache_modifier=".cg")
        tl.debug_barrier()
        cost = base_cost + tl.load(gather, cache_modifier=".cg")
        weight = -cost / temperature
        weight = tl.exp(weight - tl.max(weight, axis=0)[None, :])
        weight = weight / tl.sum(weight, axis=0)[None, :]
        tl.atomic_add(scatter, weight)
        tl.debug_barrier()

        load = tl.load(load_base, cache_modifier=".cg")
        step = temperature * (tl.log(tl.maximum(load, load_floor)) - share)
        step = tl.minimum(tl.maximum(step, -bound), bound)
        correction = correction + relax * step
        correction = tl.where(counts > 0, correction, 0.0)
        tl.store(corr_base, correction, cache_modifier=".cg")
        tl.debug_barrier()
        temperature *= anneal

    tl.store(out_ptr + problem * stride_m_n + target_row,
             tl.load(offset_ptr + problem * stride_m_n + target_row) + correction)


def _tiled(logits: Tensor, tile: int) -> Tensor:
    """``(N, tiles, tile)`` over the whole vocabulary, ragged tail padded out of contention.

    A vocabulary that is not a multiple of ``tile`` has no tail tile to view, and dropping
    it would lose whatever top logits sit there, so it is padded -- one copy of the logits,
    which is why the divisible case stays a view.
    """
    n_problems, n_vocab = logits.shape
    if n_vocab % tile == 0:
        return logits.view(n_problems, -1, tile)
    padded = torch.nn.functional.pad(logits, (0, tile - n_vocab % tile), value=-torch.inf)
    return padded.view(n_problems, -1, tile)


def scan_logits(logits: Tensor, tile: int = 64):
    """One streaming pass: per-tile maxima and the exact softmax normaliser.

    Both things the shortlist needs from the full vocabulary, for one read of it.
    In torch they are two passes, and the second is waste -- the running maximum
    the normaliser needs is already being computed by the first.
    """
    n_problems, n_vocab = logits.shape
    n_tiles = -(-n_vocab // tile)
    # bfloat16 loses nothing when the logits are bfloat16: a maximum of them is
    # exactly representable, and this halves what the topk over it reads.
    tile_max = torch.empty(n_problems, n_tiles, dtype=logits.dtype, device=logits.device)
    largest = torch.empty(n_problems, dtype=torch.float32, device=logits.device)
    total = torch.empty_like(largest)
    _scan_kernel[(n_problems,)](logits, tile_max, largest, total, n_vocab, logits.stride(0),
                                TILE=tile, SENTINEL=EMPTY)
    return tile_max.masked_fill(tile_max < EMPTY / 10, -torch.inf), largest + total.log()


def condense(table: Tensor, tokens: Tensor, prior: Tensor, k: int) -> Tensor:
    """Each item's ``k`` cheapest candidates, packed, as ``(N, k, A)`` int32.

    Selection is on ``cost + prior`` and the returned costs *keep* ``prior``
    folded in -- taking it back out is four more passes -- so callers price
    against ``bias - prior``.
    """
    n_problems, n_targets = tokens.shape
    n_items = table.shape[1]
    tokens = tokens.to(torch.int64).contiguous()
    prior = prior.float().contiguous()
    key = torch.empty(n_problems, k, n_items, dtype=torch.int32, device=table.device)
    grid = lambda meta: (n_problems, triton.cdiv(n_items, meta["BLOCK"]))  # noqa: E731
    _condense_kernel[grid](table, tokens, prior, key, n_items,
                           tokens.stride(0), prior.stride(0), key.stride(0),
                           M=n_targets, K=k)
    return key


def quota(weights: Tensor, total: int) -> Tensor:
    """Largest-remainder apportionment: integer counts summing exactly to ``total``.

    ``weights`` must sum to one along the last axis; the caller renormalises.
    """
    exact = weights * total
    counts = exact.floor().to(torch.int64)
    remaining = total - counts.sum(dim=-1, keepdim=True)
    rank = (exact - counts).argsort(dim=-1, descending=True).argsort(dim=-1)
    return counts + (rank < remaining).to(torch.int64)


@dataclass
class Assignment:
    tokens: Tensor  # (N, M) int64: the candidate token ids
    counts: Tensor  # (N, M) int64: how many auxiliaries each candidate asked for
    sigma: Tensor  # (N, A) int64: index into `tokens`, so auxiliary a -> tokens[sigma[a]]
    bias: Tensor  # (N, M) float32: the fitted factors


class Voronoi:
    """Holds the vocabulary-wide ``(V, A)`` cost table; ``assign`` maps a batch of dense logits to an assignment."""

    num_warps = 32

    @classmethod
    def from_table(cls, table: Tensor) -> "Voronoi":
        """An engine over a contiguous ``(V, A)`` table whose row ``v`` holds the cost of handing each auxiliary to token ``v``."""
        engine = cls.__new__(cls)
        engine.table = table
        engine.n_items = table.shape[1]
        engine._columns = None
        return engine

    def shortlist(self, logits: Tensor, n_targets: int = 128, tile: int = 64):
        """Candidate tokens and their quotas, from the dense logits alone.

        Exact, not approximate: each of the ``n_targets`` largest logits sits in
        some tile whose maximum is at least as large, and there are only
        ``n_targets`` such logits, so at most ``n_targets`` tiles can contain one.
        Taking the top tiles and then the top logits within them cannot miss any.
        ``torch.topk`` over the whole vocabulary agrees exactly, at a sixth of
        the speed.
        """
        n_problems, n_vocab = logits.shape
        tile_max, normaliser = scan_logits(logits, tile)
        # A vocabulary holding fewer tiles than there are targets to fill has nothing to
        # rank: every tile is a candidate.
        candidate = tile_max.topk(min(n_targets, tile_max.shape[1]), dim=-1).indices

        tiles = _tiled(logits, tile)
        pool = tiles.gather(1, candidate[:, :, None].expand(-1, -1, tile)).reshape(n_problems, -1)
        value, position = pool.topk(n_targets, dim=-1)
        tokens = candidate.gather(1, position // tile) * tile + position % tile

        # Renormalise: on a real head this is a no-op, but on a flat distribution
        # the top n_targets can hold a couple of percent of the mass, and
        # largest-remainder only sums to A when handed something summing to one.
        probability = (value.float() - normaliser[:, None]).exp()
        return tokens, quota(probability / probability.sum(-1, keepdim=True), self.n_items)

    def spread(self, tokens: Tensor, sample: int = 64, chunk: int = 4096) -> Tensor:
        """``(N, 1)`` gap between an item's best and worst target over a fixed slice of items: the unit ``b`` lives in."""
        columns = self._columns
        if columns is None or columns.shape[1] != sample:
            self._columns = columns = self.table[:, :sample].contiguous()
        units = []
        for start in range(0, tokens.shape[0], chunk):
            head = columns[tokens[start:start + chunk].long()]
            units.append((head.amax(1) - head.amin(1)).float().mean(-1))
        return torch.cat(units)[:, None]

    def prior(self, counts: Tensor, unit: Tensor, weight: float = 0.1) -> Tensor:
        """The factor a token deserves before any geometry is looked at."""
        share = (counts.float() / self.n_items).clamp_min(1e-9)
        return torch.where(counts > 0, -weight * unit * share.log(),
                           (BLOCKED * unit).expand_as(share))

    def fit(self, key: Tensor, counts: Tensor, offset: Tensor, unit: Tensor,
            iterations: int, epsilon: float, relax: float = 1.9,
            endpoint: float = 0.0087, limit: float = STEP_LIMIT) -> Tensor:
        """The whole annealed Sinkhorn fit on the condensed problem in one launch per problem; ``relax`` over-relaxes each correction."""
        n_problems, shortlist, n_items = key.shape
        n_targets = counts.shape[1]
        # Every axis becomes a `tl.arange`, so each has to be a power of two: top_m, the shortlist and the auxiliary vocabulary.
        if any(size & (size - 1) for size in (n_items, n_targets, shortlist)):
            raise ValueError(f"the fused anneal needs power-of-two sizes, got {n_items}, {n_targets}, {shortlist}")

        out = torch.empty(n_problems, n_targets, dtype=torch.float32, device=key.device)
        scratch = torch.empty(2, n_problems, n_targets, dtype=torch.float32, device=key.device)
        _fit_kernel[(n_problems,)](
            key, counts.contiguous(), offset.float().contiguous(), unit.contiguous().view(-1),
            out, scratch[0], scratch[1],
            key.stride(0), n_targets,
            epsilon, (endpoint / epsilon) ** (1.0 / max(iterations, 1)), relax, LOAD_FLOOR, limit,
            M=n_targets, K=shortlist, A=n_items, ITERATIONS=iterations,
            num_warps=self.num_warps,
        )
        return out

    def argmin(self, tokens: Tensor, bias: Tensor, chunk: int = 2048) -> Tensor:
        """``sigma(a) = argmin_m (C[m, a] + b_m)`` over all ``M`` targets."""
        out = torch.empty(tokens.shape[0], self.n_items, dtype=torch.int64, device=self.table.device)
        for start in range(0, tokens.shape[0], chunk):
            block = slice(start, min(start + chunk, tokens.shape[0]))
            out[block] = _fused_argmin(self.table, tokens[block].long(), bias[block])
        return out

    def assign(self, logits: Tensor, *, n_targets: int = 128, k: int = 8, rounds: int = 2,
               iterations: int = 12, weight: float = 0.1, epsilon: float = 0.15,
               refocus: float = 0.35, **kwargs) -> Assignment:
        """Dense ``(N, V)`` logits to an assignment.

        The defaults -- ``k=8, rounds=2, iterations=12`` -- are the cheapest
        setting that never collapses a problem; see the module docstring before
        lowering any of them.  Raising ``iterations`` to 24 or 48 buys a little
        more accuracy at proportionally more time.
        """
        tokens, counts = self.shortlist(logits, n_targets)
        unit = self.spread(tokens)
        bias = self.prior(counts, unit, weight)

        for index in range(rounds):
            offset = bias
            key = condense(self.table, tokens, offset, k)
            bias = self.fit(key, counts, offset, unit, iterations,
                            epsilon if index == 0 else epsilon * refocus, **kwargs)

        return Assignment(tokens, counts, self.argmin(tokens, bias), bias)


def _argmin_impl(table: Tensor, tokens: Tensor, bias: Tensor) -> Tensor:
    return (table[tokens].float() + bias[:, :, None]).argmin(1)


# Uncompiled this materialises the chunk's cost matrix in float32 first, which is
# eight times the traffic of the reduction itself.
_fused_argmin = torch.compile(_argmin_impl, dynamic=False)

