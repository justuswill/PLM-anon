"""Install the RTX 6000 Ada GEMM tile table for one round width into vLLM's unquantized linear path (``--mode fast``).

``gpu_tuning/tiles_hidden<H>.json`` holds the tiles for one model width: round widths 56 and 110 (post-trained, k = 7 and 10)
at hidden sizes 2048 and 3072, and 42 (pretrained, k = 6) at every released hidden size. A missing shape is an error, never a
silent fallback to cuBLAS.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import fast_gemm

TABLES = Path(__file__).resolve().parent / "gpu_tuning"
VOCAB_PADDING = 64   # vLLM's ParallelLMHead pads the vocabulary to this multiple
WIDEST_TUNED_DOWN_ROUND = 72
# Above W=72 the tuner's SPLIT_K=4 tile for `down` loses to this SPLIT_K=1 one inside the round graph, being one launch fewer.
DOWN_SPLIT_K1 = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3}


def model_shapes(hf_config) -> dict[str, tuple[int, int]]:
    hidden, intermediate, head_dim = hf_config.hidden_size, hf_config.intermediate_size, hf_config.head_dim
    heads, kv_heads = hf_config.num_attention_heads, hf_config.num_key_value_heads
    vocab = -(-hf_config.vocab_size // VOCAB_PADDING) * VOCAB_PADDING
    return {"qkv": ((heads + 2 * kv_heads) * head_dim, hidden), "o": (hidden, heads * head_dim), "gate_up": (2 * intermediate, hidden),
            "down": (hidden, intermediate), "lm_head": (vocab, hidden)}


def load_table(hf_config, width: int) -> dict[tuple[int, int, int], dict]:
    path = TABLES / f"tiles_hidden{hf_config.hidden_size}.json"
    if not path.exists():
        raise SystemExit(f"--mode fast has no tile table for hidden size {hf_config.hidden_size}; use --mode default")
    table = {tuple(int(v) for v in key.split(",")): cfg for key, cfg in json.loads(path.read_text()).items()}
    shapes = model_shapes(hf_config)
    wanted = {name: (width, *shape) for name, shape in shapes.items()}
    missing = set(wanted.values()) - set(table)
    if missing:
        raise SystemExit(f"--mode fast has no tuned tile for (width, N, K) {sorted(missing)} in {path.name}; use --mode default or a tuned --k")
    configs = {key: table[key] for key in wanted.values()}
    if width > WIDEST_TUNED_DOWN_ROUND:
        configs[wanted["down"]] = DOWN_SPLIT_K1
    return configs


def install(hf_config, width: int) -> None:
    fast_gemm.install(load_table(hf_config, width))
