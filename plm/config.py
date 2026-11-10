"""The decoding configuration behind ``--mode``, ``--match`` and ``--k``, and the engines it builds.

``--mode fast`` routes the round's five GEMM shapes through Triton tiles tuned on an RTX 6000 Ada (a process-global
patch of vLLM's linear path). ``--mode default`` is stock vLLM with no patching and runs on any GPU vLLM supports.
``--match exact`` accepts a proposal only if it equals the verify row's own decode, so the output distribution is the
model's own. ``--match relaxed`` also accepts proposals that pass a looser rule on the verify row, chosen per
``--setting``: the top-0.94 nucleus for post-trained models, probability above 2e-3 for pretrained ones.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

KV_BLOCK_SIZE = 64
KV_CACHE_BYTES = 1 << 30
GPU_MEMORY_UTILIZATION = 0.25
PREFILL_BUCKETS = tuple(range(64, 641, 64))


@dataclass(frozen=True)
class Setting:
    default_k: dict[str, int]       # --match -> round width k
    relaxed_rule: tuple[str, float]  # (accept rule, threshold) for --match relaxed


# Fitted on this repository's GPU: k for tokens per second, the relaxed rule for the fastest rule within 1% MAUVE of exact.
SETTINGS = {
    "posttrain": Setting(default_k={"exact": 7, "relaxed": 10}, relaxed_rule=("nucleus", 0.94)),
    "pretrain": Setting(default_k={"exact": 6, "relaxed": 6}, relaxed_rule=("floor", 2e-3)),
}
# The Voronoi sampler runs in the parallel rounds' float16 so the two decodes match token for token; top-k AR runs in the training dtype.
AR_DTYPE = {"voronoi": "float16", "topk": "bfloat16"}


def is_plm_export(model: str) -> bool:
    config = Path(model) / "config.json"
    return config.exists() and "PLMForCausalLM" in json.loads(config.read_text()).get("architectures", [])


@dataclass
class RoundConfig:
    model: str
    setting: str = "posttrain"
    mode: str = "fast"
    match: str = "exact"
    k: int | None = None
    dtype: str = "float16"
    max_model_len: int = 4096

    def __post_init__(self) -> None:
        if self.k is None:
            self.k = SETTINGS[self.setting].default_k[self.match]

    @property
    def accept(self) -> tuple[str, float | None]:
        """Exact match is the nucleus test switched off, in either setting."""
        return SETTINGS[self.setting].relaxed_rule if self.match == "relaxed" else ("nucleus", None)

    @property
    def width(self) -> int:
        return self.k + self.k * self.k

    def describe(self) -> dict:
        rule, threshold = self.accept
        return {"setting": self.setting, "mode": self.mode, "match": self.match, "k": self.k, "width": self.width, "dtype": self.dtype,
                "model": self.model, "accept": f"{rule} {threshold}" if threshold is not None else "exact"}

    def build(self):
        from vllm_plm import gemm_tables
        from vllm_plm.decoder import RoundDecoder, build_llm

        llm = build_llm(self.model, dtype=self.dtype, max_model_len=self.max_model_len, gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                        kv_cache_memory_bytes=KV_CACHE_BYTES, block_size=KV_BLOCK_SIZE)
        if self.mode == "fast":
            gemm_tables.install(llm.llm_engine.model_config.hf_config, self.width)
        decoder = RoundDecoder(llm, self.k, *self.accept, PREFILL_BUCKETS)
        decoder.prepare()
        return decoder


def build_ar(model: str, dtype: str, max_model_len: int, voronoi_scheme: str | None):
    """Stock vLLM autoregression, with CUDA graphs on: the fastest AR available. A PLM export is served as the plain Llama it is on this path."""
    from vllm import LLM

    extra = {}
    if is_plm_export(model):
        extra["hf_overrides"] = {"architectures": ["PLMBackboneForCausalLM"]}
    if voronoi_scheme is not None:
        # The processor is built inside the engine core process, so it reads its scheme from the environment.
        os.environ["PLM_VORONOI_SCHEME_DIR"] = str(Path(voronoi_scheme).resolve())
        extra["logits_processors"] = ["vllm_plm.voronoi_ar:VoronoiARLogitsProcessor"]
    return LLM(model=model, dtype=dtype, max_model_len=max_model_len, gpu_memory_utilization=GPU_MEMORY_UTILIZATION, block_size=KV_BLOCK_SIZE,
               kv_cache_memory_bytes=KV_CACHE_BYTES, max_num_seqs=1, disable_log_stats=True, **extra)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="a downloaded model directory, e.g. checkpoints/posttrain/plm-1.3B-beta1")
    parser.add_argument("--setting", default="posttrain", choices=tuple(SETTINGS),
                        help="posttrain: k 7 (exact) / 10 (relaxed), nucleus 0.94. pretrain: k 6, probability floor 2e-3")
    parser.add_argument("--mode", default="fast", choices=("fast", "default"),
                        help="fast: Triton GEMM tiles tuned on an RTX 6000 Ada. default: stock vLLM, any GPU")
    parser.add_argument("--match", default="exact", choices=("exact", "relaxed"), help="exact, or relaxed (the setting's looser accept rule)")
    parser.add_argument("--k", type=int, help="proposal length per round; default from --setting")
    parser.add_argument("--max-model-len", type=int, default=4096, help="prompt plus generated tokens")
    parser.add_argument("--ar", action="store_true", help="autoregressive decoding with stock vLLM instead; ignores --mode, --match and --k")
    parser.add_argument("--ar-sampler", default="voronoi", choices=tuple(AR_DTYPE),
                        help="voronoi: a PLM's own decode rule on the seeded random values, theoretically token-identical to --match exact. "
                             "topk: temperature 1 over the 128 likeliest tokens")
    parser.add_argument("--ar-dtype", default="", choices=("", "bfloat16", "float16"), help="default: float16 for voronoi, bfloat16 for topk")
    parser.add_argument("--ar-voronoi-scheme", default="", help="PLM export whose auxiliary scheme --ar-sampler voronoi decodes with; default the model's own")


def from_args(args: argparse.Namespace) -> RoundConfig:
    return RoundConfig(model=args.model, setting=args.setting, mode=args.mode, match=args.match, k=args.k, max_model_len=args.max_model_len)
