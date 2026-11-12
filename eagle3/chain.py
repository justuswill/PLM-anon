"""EAGLE-3 as vLLM's speculative decoding: the head drafts a chain of ``k`` tokens per target call, verified by exact
rejection sampling against the target's top-128 distribution at temperature 1."""

from __future__ import annotations

import os
import time

from bench import engine
from bench.harness import speed_record

SPEC_DRAFTS, SPEC_ACCEPTED = "vllm:spec_decode_num_drafts", "vllm:spec_decode_num_accepted_tokens"


def build(model: str, head: str, k: int, draft: str, max_model_len: int):
    from vllm import LLM

    if draft == "topk":
        # Patched here, before the engine core forks, and by the plugin in a spawned engine core, which reads the environment.
        os.environ["EAGLE3_DRAFT_TOP_K"] = str(engine.TOP_K)
        from eagle3 import draft_topk

        draft_topk.install()
    speculative = {"method": "eagle3", "model": head, "num_speculative_tokens": k, "draft_sample_method": "greedy" if draft == "greedy" else "probabilistic"}
    # Speculative decoding reports its drafts and accepted tokens only through the engine's stats, so they stay on.
    return LLM(model=model, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=engine.GPU_MEMORY_UTILIZATION,
               block_size=engine.KV_BLOCK_SIZE, kv_cache_memory_bytes=engine.KV_CACHE_BYTES, max_num_seqs=1, disable_log_stats=False,
               speculative_config=speculative)


def sampling(tokens: int, stop_at_eos: bool, seed: int):
    from vllm import SamplingParams

    return SamplingParams(max_tokens=tokens, ignore_eos=not stop_at_eos, temperature=1.0, top_k=engine.TOP_K, seed=seed)


def speculative_counts(llm) -> tuple[int, int]:
    values = {metric.name: metric.value for metric in llm.get_metrics() if metric.name in (SPEC_DRAFTS, SPEC_ACCEPTED)}
    return values.get(SPEC_DRAFTS, 0), values.get(SPEC_ACCEPTED, 0)


def run(llm, prompt: str, params) -> dict:
    """One target call for the prefill, then one per draft it verifies; ``accepted_per_call`` is left empty, vLLM reports totals only."""
    drafts_before, accepted_before = speculative_counts(llm)
    started = time.perf_counter()
    output = llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
    ms = 1000 * (time.perf_counter() - started)
    drafts_after, accepted_after = speculative_counts(llm)
    emitted, calls = len(output.token_ids), 1 + drafts_after - drafts_before
    record = speed_record(prompt, output.text, [], calls, emitted, ms, ms, emitted)
    record["accepted_draft_tokens"] = accepted_after - accepted_before
    return record
