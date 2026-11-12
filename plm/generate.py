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
    return SamplingParams(max_tokens=args.tokens, ignore_eos=not args.stop_at_eos, temperature=1.0, top_k=TOP_K, seed=seed)


def run_ar(llm, prompt: str, params) -> dict:
    started = time.perf_counter()
    output = llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
    ms = 1000 * (time.perf_counter() - started)
    emitted = len(output.token_ids)
    return speed_record(prompt, output.text, [0] * emitted, emitted, emitted, ms, ms, emitted)


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
    if args.ar:
        dtype = args.ar_dtype or plm_config.AR_DTYPE[args.ar_sampler]
        scheme = (args.ar_voronoi_scheme or args.model) if args.ar_sampler == "voronoi" else None
        llm = plm_config.build_ar(args.model, dtype, args.max_model_len, scheme)
        described = {"decoding": f"ar-{args.ar_sampler}", "model": args.model, "dtype": dtype,
                     "voronoi_scheme": scheme, "top_k": TOP_K if args.ar_sampler == "topk" else None}
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
