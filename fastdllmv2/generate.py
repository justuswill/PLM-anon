"""The Fast-dLLM v2 baseline's decoding for ``bench.generate --fdv2``: an FDv2 export decoded by block diffusion until the
end-of-text token.

    python -m bench.generate --model checkpoints/posttrain/fdv2-1.3B --fdv2 --fdv2-threshold 0.9 \\
        --dataset prompts/gsm8k/prompts.json --tokens 512 --stop-at-eos --seeds 0 1 2 --out results.json

Each pass unmasks every position of the active 8-token sub-block above ``--fdv2-threshold``, plus the most confident one,
sampling at temperature 1 over the 128 likeliest tokens (or greedily with ``--fdv2-temperature 0``). ``tokens/call`` counts
the prefill as one call and every denoising and commit pass as one.
"""

from __future__ import annotations

import argparse
import os
import time
import zlib
from collections.abc import Callable

from bench import engine
from bench.harness import speed_record

CANVAS = 32
SUB_BLOCK = 8
MASK_TOKEN_ID = 50257
EOD_TOKEN_ID = 50256


def build(model: str, max_model_len: int, threshold: float, temperature: float):
    # The sampler is built inside the engine core process before any request, so it reads its contract from the environment.
    os.environ["FDV2_THRESHOLD"] = str(threshold)
    os.environ["FDV2_TEMPERATURE"] = str(temperature)
    os.environ["FDV2_TOP_K"] = str(engine.TOP_K)
    os.environ["FDV2_SUB_BLOCK"] = str(SUB_BLOCK)
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"   # the driver reads the sampler's pass counts through collective_rpc
    from vllm import LLM

    return LLM(model=model, dtype="bfloat16", max_model_len=max_model_len, enable_prefix_caching=False,
               hf_overrides={"canvas_length": CANVAS, "mask_token_id": MASK_TOKEN_ID, "architectures": ["Fdv2ForBlockDiffusion"]},
               diffusion_config={"canvas_length": CANVAS, "max_denoising_steps": CANVAS}, attention_backend="TRITON_ATTN",
               max_num_seqs=1, max_num_batched_tokens=CANVAS, gpu_memory_utilization=engine.GPU_MEMORY_UTILIZATION,
               kv_cache_memory_bytes=engine.KV_CACHE_BYTES)


def sampler_passes(worker) -> tuple[int, int]:
    sampler = worker.model_runner.sampler
    return sampler.prefill_passes, sampler.decode_passes


def seed_sampler(worker, seed: int) -> None:
    import torch

    torch.manual_seed(seed)


def run(llm, prompt: str, tokens: int, seed: int) -> dict:
    """The prompt up to its last 32-aligned boundary is prefilled; the rest opens the first canvas. The prefill counts as one call."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    ids = llm.get_tokenizer()(prompt, add_special_tokens=False)["input_ids"]
    aligned = (len(ids) - 1) // CANVAS * CANVAS
    prefix, tail = (ids[:aligned], ids[aligned:]) if aligned else (ids, [])
    # Seeded per prompt so a pass does not depend on what ran before it in the process.
    llm.collective_rpc(seed_sampler, args=(seed * 2**32 + zlib.crc32(prompt.encode()),))
    _, decode_before = llm.collective_rpc(sampler_passes)[0]
    started = time.perf_counter()
    output = llm.generate([TokensPrompt(prompt_token_ids=prefix)], SamplingParams(max_tokens=tokens + len(tail), extra_args={"fdv2_tail": tail}), use_tqdm=False)[0].outputs[0]
    ms = 1000 * (time.perf_counter() - started)
    _, decode_after = llm.collective_rpc(sampler_passes)[0]
    answer = list(output.token_ids)[len(tail):]
    if answer and answer[-1] == EOD_TOKEN_ID:
        answer = answer[:-1]
    calls = 1 + decode_after - decode_before
    return speed_record(prompt, llm.get_tokenizer().decode(answer, skip_special_tokens=True), [], calls, len(answer), ms, ms, len(answer))


def generator(args: argparse.Namespace) -> tuple[dict, Callable[[str, int], dict]]:
    if not args.stop_at_eos:
        raise SystemExit("--fdv2 decodes until the end-of-text token: pass --stop-at-eos")
    llm = build(args.model, args.max_model_len, args.fdv2_threshold, args.fdv2_temperature)
    described = {"decoding": "fdv2", "model": args.model, "dtype": "bfloat16", "threshold": args.fdv2_threshold, "temperature": args.fdv2_temperature,
                 "top_k": engine.TOP_K if args.fdv2_temperature else None, "canvas": CANVAS, "sub_block": SUB_BLOCK}
    return described, lambda prompt, seed: run(llm, prompt, args.tokens, seed)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Fast-dLLM v2 baseline")
    group.add_argument("--fdv2", action="store_true", help="the model is an FDv2 export: decode it by block diffusion; needs --stop-at-eos")
    group.add_argument("--fdv2-threshold", type=float, default=0.9, help="unmask every position above this probability (>= 1: one a pass)")
    group.add_argument("--fdv2-temperature", type=float, default=1.0, choices=(0.0, 1.0), help="1 samples the 128 likeliest tokens, 0 is greedy")
