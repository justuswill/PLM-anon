"""The round's attention mask, in vLLM's logical coordinates.

A round packs ``k`` verify rows, which check last round's chain causally, and ``k`` proposal blocks, each written as
though the prefix it assumes were already confirmed, so the blocks must not see each other. vLLM's FlexAttention
backend converts paged indices to logical ones before calling ``logical_mask_mod``; the per-row tables are held
relative to the round offset, so they survive every round whose layout repeats.
"""

from __future__ import annotations

import torch

# No key index reaches it, so a row with this own-block start has no own-block term.
UNREACHABLE = 1 << 30


class RoundMaskView:
    """The callable every attention layer holds. One object for all layers: the backend raises on differing mask mods."""

    def __init__(self, tables: "RoundMask") -> None:
        self.tables = tables

    def __call__(self, batch_index: torch.Tensor, head_index: torch.Tensor, logical_q_idx: torch.Tensor, logical_kv_idx: torch.Tensor) -> torch.Tensor:
        tables = self.tables
        row = logical_q_idx - tables.offset
        key = logical_kv_idx - tables.offset
        prefix_visible = key <= tables.depth[row]
        own_block_visible = (key >= tables.own_start[row]) & (logical_kv_idx <= logical_q_idx)
        return prefix_visible | own_block_visible


class RoundMask:
    """Persistent tables refilled in place: a mask closure over tensors is guarded on shape, never value, so it never recompiles."""

    def __init__(self, max_query_len: int, device: torch.device) -> None:
        self.depth = torch.zeros(max_query_len, dtype=torch.int64, device=device)
        self.own_start = torch.full((max_query_len,), UNREACHABLE, dtype=torch.int64, device=device)
        self.offset = torch.zeros((), dtype=torch.int64, device=device)
        self.view = RoundMaskView(self)


def round_layout(block_sizes: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(rel_positions, rel_aux, depth, own_start)`` per row of a round with ``len(block_sizes)`` verify rows, relative to its offset.

    Block d's row j predicts the token at offset+d+1+j and reads that position's auxiliary, so like an AR position it sits
    one before its target, at offset+d+j. Verify rows are real tokens and read the masked sentinel (``rel_aux = -1``).
    """
    n_hyp = len(block_sizes)
    total = n_hyp + sum(block_sizes)
    starts, start = [], n_hyp
    for width in block_sizes:
        starts.append(start)
        start += width
    layout = torch.tensor(block_sizes + starts, dtype=torch.int64, device=device)
    widths, block_starts = layout[:n_hyp], layout[n_hyp:]
    depths = torch.arange(n_hyp, device=device)
    tail = total - n_hyp
    depth_of_row = torch.repeat_interleave(depths, widths, output_size=tail)
    own_start_of_row = torch.repeat_interleave(block_starts, widths, output_size=tail)
    local = torch.arange(n_hyp, total, device=device) - own_start_of_row
    rel_positions = torch.cat([depths, depth_of_row + local])
    depth = torch.cat([depths, depth_of_row])
    own_start = torch.cat([torch.full((n_hyp,), UNREACHABLE, dtype=torch.int64, device=device), own_start_of_row])
    rel_aux = torch.cat([torch.full((n_hyp,), -1, dtype=torch.int64, device=device), depth_of_row + local])
    return rel_positions, rel_aux, depth, own_start
