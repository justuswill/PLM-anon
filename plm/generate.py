"""Generate a continuation for every prompt in a dataset, once per seed, and write a results file with per-prompt speed.

    python -m plm.generate --model checkpoints/posttrain/plm-1.3B-beta1 --mode fast --match exact --k 7 \\
        --dataset prompts/gsm8k/prompts.json --tokens 512 --stop-at-eos --seeds 0 1 2 --out results.json

One process runs the whole dataset ``1 + len(--seeds)`` times: a warm-up pass, discarded, which pays graph capture,
compilation and a cold clock, then one pass per seed. The seed drives the auxiliary stream of parallel rounds and of
the Voronoi AR sampler identically, and the sampler of top-k AR.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from collections.abc import Callable
from pathlib import Path

from plm import config as plm_config

TOP_K = plm_config.TOP_K


def load_dataset(path: str, count: int | None) -> list[str]:
    loaded = json.loads(Path(path).read_text())
    if not isinstance(loaded, list) or not all(isinstance(prompt, str) for prompt in loaded):
        raise SystemExit(f"{path}: expected a JSON list of prompt strings")
    return loaded if count is None else loaded[:count]


def speed_record(prompt: str, continuation: str, accepted: list[int], calls: int, tokens: int, ms: float, steady_ms: float, steady_tokens: int) -> dict:
    return {"prompt": prompt, "continuation": continuation, "accepted_per_call": accepted, "calls": calls, "tokens": tokens, "ms": ms,
            "ms_per_call": steady_ms / calls if calls else 0.0, "tokens_per_call": steady_tokens / calls if calls else 0.0,
            "tokens_per_second": tokens / (ms / 1000) if ms else 0.0}


def ar_sampling(args: argparse.Namespace, seed: int):
    from vllm import SamplingParams

    if args.ar_sampler == "voronoi":   # the processor leaves one finite logit per row; greedy reads it off
        return SamplingParams(max_tokens=args.tokens, ignore_eos=not args.stop_at_eos, temperature=0.0, extra_args={"aux_seed": seed})
    if args.ar_sampler == "greedy":
        return SamplingParams(max_tokens=args.tokens, ignore_eos=not args.stop_at_eos, temperature=0.0)
    return SamplingParams(max_tokens=args.tokens, ignore_eos=not args.stop_at_eos, temperature=1.0, top_k=TOP_K, seed=seed)


def run_ar(llm, prompt: str, params) -> dict:
    started = time.perf_counter()
    output = llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
    ms = 1000 * (time.perf_counter() - started)
    emitted = len(output.token_ids)
    return speed_record(prompt, output.text, [0] * emitted, emitted, emitted, ms, ms, emitted)


def fdv2_passes(worker) -> tuple[int, int]:
    sampler = worker.model_runner.sampler
    return sampler.prefill_passes, sampler.decode_passes


def seed_fdv2(worker, seed: int) -> None:
    import torch

    torch.manual_seed(seed)


def run_fdv2(llm, prompt: str, tokens: int, seed: int) -> dict:
    """The prompt up to its last 32-aligned boundary is prefilled; the rest opens the first canvas. The prefill counts as one call, as for the other decodings."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    ids = llm.get_tokenizer()(prompt, add_special_tokens=False)["input_ids"]
    aligned = (len(ids) - 1) // plm_config.FDV2_CANVAS * plm_config.FDV2_CANVAS
    prefix, tail = (ids[:aligned], ids[aligned:]) if aligned else (ids, [])
    # Seeded per prompt so a pass does not depend on what ran before it in the process.
    llm.collective_rpc(seed_fdv2, args=(seed * 2**32 + zlib.crc32(prompt.encode()),))
    _, decode_before = llm.collective_rpc(fdv2_passes)[0]
    started = time.perf_counter()
    output = llm.generate([TokensPrompt(prompt_token_ids=prefix)], SamplingParams(max_tokens=tokens + len(tail), extra_args={"fdv2_tail": tail}), use_tqdm=False)[0].outputs[0]
    ms = 1000 * (time.perf_counter() - started)
    _, decode_after = llm.collective_rpc(fdv2_passes)[0]
    answer = list(output.token_ids)[len(tail):]
    if answer and answer[-1] == plm_config.FDV2_EOD_TOKEN_ID:
        answer = answer[:-1]
    calls = 1 + decode_after - decode_before
    return speed_record(prompt, llm.get_tokenizer().decode(answer, skip_special_tokens=True), [], calls, len(answer), ms, ms, len(answer))


def run_plm(decoder, prompt: str, tokens: int, stop_at_eos: bool, seed: int) -> dict:
    started = time.perf_counter()
    first_round = int(decoder.sequence_step.item())
    result = decoder.generate(prompt, tokens, seed=seed, stop_at_eos=stop_at_eos)
    accepted = decoder.round_sequence[first_round: int(decoder.sequence_step.item())].tolist()
    ms = 1000 * (time.perf_counter() - started)
    record = speed_record(prompt, result.text, accepted, result.steady_rounds, len(result.token_ids), ms, result.steady_ms, result.steady_tokens)
    record["tokens_per_second"] = result.tokens_per_second
    return record


def measure(generate_one: Callable[[str, int], dict], dataset: list[str], seed: int) -> list[dict]:
    records = []
    for index, prompt in enumerate(dataset, 1):
        records.append(generate_one(prompt, seed))
        print(f"  [seed {seed}  {index}/{len(dataset)}] {records[-1]['tokens_per_call']:.2f} tok/call  {records[-1]['tokens_per_second']:.0f} tok/s")
    return records


def aggregate(records: list[dict]) -> dict:
    calls = sum(record["calls"] for record in records)
    tokens = sum(record["tokens"] for record in records)
    steady_ms = sum(record["ms_per_call"] * record["calls"] for record in records)
    return {"prompts": len(records), "calls": calls, "tokens": tokens, "ms_per_call": steady_ms / calls if calls else 0.0,
            "tokens_per_call": tokens / calls if calls else 0.0, "tokens_per_second": 1000 * tokens / steady_ms if steady_ms else 0.0,
            "e2e_tokens_per_second": 1000 * tokens / sum(record["ms"] for record in records)}


def build_generator(args: argparse.Namespace) -> tuple[dict, Callable[[str, int], dict]]:
    if args.fdv2:
        if not args.stop_at_eos:
            raise SystemExit("--fdv2 decodes until the end-of-text token: pass --stop-at-eos")
        llm = plm_config.build_fdv2(args.model, args.max_model_len, args.fdv2_threshold, args.fdv2_temperature)
        described = {"decoding": "fdv2", "model": args.model, "dtype": "bfloat16", "threshold": args.fdv2_threshold, "temperature": args.fdv2_temperature,
                     "top_k": TOP_K if args.fdv2_temperature else None, "canvas": plm_config.FDV2_CANVAS, "sub_block": 8}
        return described, lambda prompt, seed: run_fdv2(llm, prompt, args.tokens, seed)
    if args.ar:
        dtype = args.ar_dtype or plm_config.AR_DTYPE[args.ar_sampler]
        scheme = (args.ar_voronoi_scheme or args.model) if args.ar_sampler == "voronoi" else None
        llm = plm_config.build_ar(args.model, dtype, args.max_model_len, scheme)
        described = {"decoding": f"ar-{args.ar_sampler}", "model": args.model, "dtype": dtype,
                     "voronoi_scheme": scheme, "top_k": TOP_K if args.ar_sampler == "topk" else None, "temperature": 0.0 if args.ar_sampler == "greedy" else 1.0}
        return described, lambda prompt, seed: run_ar(llm, prompt, ar_sampling(args, seed))
    round_config = plm_config.from_args(args)
    decoder = round_config.build()
    described = {"decoding": round_config.match, **round_config.describe()}
    return described, lambda prompt, seed: run_plm(decoder, prompt, args.tokens, args.stop_at_eos, seed)


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, help="JSON list of prompt strings")
    parser.add_argument("--prompts", type=int, default=None, help="use only the first N prompts")
    parser.add_argument("--tokens", type=int, default=128, help="generated tokens per prompt (a ceiling with --stop-at-eos)")
    parser.add_argument("--stop-at-eos", action="store_true", help="end each continuation at the end-of-text token")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="one measured pass per seed, after a discarded warm-up pass")
    parser.add_argument("--out", default="results.json")


def run_passes(described: dict, generate_one: Callable[[str, int], dict], dataset: list[str], args: argparse.Namespace) -> None:
    """The warm-up pass, then one measured pass per seed, written as one results file."""
    print("=== warm-up pass (discarded)")
    measure(generate_one, dataset, args.seeds[0])
    passes = []
    for seed in args.seeds:
        print(f"=== seed {seed}")
        records = measure(generate_one, dataset, seed)
        passes.append({"seed": seed, "aggregate": aggregate(records), "results": records})
        print("  " + "  ".join(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}" for key, value in passes[-1]["aggregate"].items()))

    described |= {"seeds": args.seeds, "prompts": len(dataset), "tokens": args.tokens, "stop_at_eos": args.stop_at_eos,
                  "max_model_len": args.max_model_len, "dataset": args.dataset}
    Path(args.out).write_text(json.dumps({"config": described, "passes": passes}, indent=1))
    print(f"  -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    plm_config.add_arguments(parser)
    add_run_arguments(parser)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.prompts)
    described, generate_one = build_generator(args)
    run_passes(described, generate_one, dataset, args)


if __name__ == "__main__":
    main()
