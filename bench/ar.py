"""Autoregressive decoding on stock vLLM, with CUDA graphs on: the fastest AR available. ``topk`` and ``greedy`` are ordinary
samplers; ``voronoi`` and ``icdf`` decode with a parallel model's own rule on its seeded random values (``plm.voronoi_ar``),
which makes them theoretically token-identical to that model's ``--match exact``."""

from __future__ import annotations

import os
import time
from pathlib import Path

from bench import engine
from bench.harness import speed_record
from plm.rounds import is_plm_export, scheme_kind

# The scheme samplers run in the parallel rounds' float16 so the two decodes match token for token; top-k AR runs in the training dtype.
AR_DTYPE = {"voronoi": "float16", "icdf": "float16", "topk": "bfloat16", "greedy": "bfloat16"}
# The auxiliary-scheme kind each scheme sampler decodes with.
AR_SCHEME_KIND = {"voronoi": "logit_voronoi", "icdf": "linear_interpolation"}


def build(model: str, dtype: str, max_model_len: int, scheme: str | None):
    """A PLM export is served as the plain Llama it is on this path."""
    from vllm import LLM

    extra = {}
    if is_plm_export(model):
        extra["hf_overrides"] = {"architectures": ["PLMBackboneForCausalLM"]}
    if scheme is not None:
        if scheme_kind(scheme) is None:
            raise SystemExit(f"{scheme} has no auxiliary scheme to decode with")
        # The processor is built inside the engine core process, so it reads its scheme from the environment.
        os.environ["PLM_AR_SCHEME_DIR"] = str(Path(scheme).resolve())
        extra["logits_processors"] = ["plm.voronoi_ar:VoronoiARLogitsProcessor"]
    return LLM(model=model, dtype=dtype, max_model_len=max_model_len, gpu_memory_utilization=engine.GPU_MEMORY_UTILIZATION, block_size=engine.KV_BLOCK_SIZE,
               kv_cache_memory_bytes=engine.KV_CACHE_BYTES, max_num_seqs=1, **extra)


def sampling(sampler: str, tokens: int, stop_at_eos: bool, seed: int):
    from vllm import SamplingParams

    if sampler in AR_SCHEME_KIND:   # the processor leaves one finite logit per row; greedy reads it off
        return SamplingParams(max_tokens=tokens, ignore_eos=not stop_at_eos, temperature=0.0, extra_args={"aux_seed": seed})
    if sampler == "greedy":
        return SamplingParams(max_tokens=tokens, ignore_eos=not stop_at_eos, temperature=0.0)
    return SamplingParams(max_tokens=tokens, ignore_eos=not stop_at_eos, temperature=1.0, top_k=engine.TOP_K, seed=seed)


def run(llm, prompt: str, params) -> dict:
    started = time.perf_counter()
    output = llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
    ms = 1000 * (time.perf_counter() - started)
    emitted = len(output.token_ids)
    return speed_record(prompt, output.text, [0] * emitted, emitted, emitted, ms, ms, emitted)
