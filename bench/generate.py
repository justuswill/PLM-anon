"""Generate a continuation for every prompt in a dataset, once per seed, and write a results file with per-prompt speed:
a PLM or the distilled PTP model in parallel rounds, any model autoregressively (``--ar``), or one of the baselines, whose
code lives in its own package: EAGLE-3 (``--eagle3 <head>``, ``eagle3/``) and Fast-dLLM v2 (``--fdv2``, ``fastdllmv2/``).

    python -m bench.generate --model checkpoints/posttrain/plm-1.3B-beta1 --mode fast --match exact --k 7 \\
        --dataset prompts/gsm8k/prompts.json --tokens 512 --stop-at-eos --seeds 0 1 2 --out results.json

One process runs the whole dataset ``1 + len(--seeds)`` times: a warm-up pass, discarded, which pays graph capture,
compilation and a cold clock, then one pass per seed. The seed drives the auxiliary stream of parallel rounds and of
the Voronoi and inverse-CDF AR samplers identically, and the sampler of top-k AR.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import eagle3.generate
import fastdllmv2.generate
from bench import ar, engine
from bench.harness import add_run_arguments, load_dataset, run_passes, speed_record
from plm.rounds import SETTINGS, RoundConfig, scheme_kind


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="a downloaded model directory, e.g. checkpoints/posttrain/plm-1.3B-beta1")
    parser.add_argument("--setting", default="posttrain", choices=tuple(SETTINGS),
                        help="posttrain: k 7 (exact) / 10 (relaxed), nucleus 0.94. pretrain: k 6, probability floor 2e-3")
    parser.add_argument("--mode", default="fast", choices=("fast", "default"),
                        help="fast: Triton GEMM tiles tuned on an RTX 6000 Ada. default: stock vLLM, any GPU")
    parser.add_argument("--match", default="exact", choices=("exact", "relaxed"), help="exact, or relaxed (the setting's looser accept rule)")
    parser.add_argument("--k", type=int, help="proposal length per round; default from --setting")
    parser.add_argument("--ar", action="store_true", help="autoregressive decoding with stock vLLM instead; ignores --mode, --match and --k")
    parser.add_argument("--ar-sampler", default="voronoi", choices=tuple(ar.AR_DTYPE),
                        help="voronoi: a PLM's own decode rule on the seeded random values, theoretically token-identical to --match exact. "
                             "icdf: the distilled model's rule, inverse CDF over the 128 likeliest tokens at temperature 1 on its seeded random values, "
                             "theoretically token-identical to its --match exact. "
                             "topk: temperature 1 over the 128 likeliest tokens. greedy: the likeliest token")
    parser.add_argument("--ar-dtype", default="", choices=("", "bfloat16", "float16"), help="default: float16 for voronoi and icdf, bfloat16 for topk and greedy")
    parser.add_argument("--ar-scheme", default="", help="with --ar-sampler voronoi or icdf: the export whose auxiliary scheme to decode with; default the model's own")


def run_plm(decoder, prompt: str, tokens: int, stop_at_eos: bool, seed: int) -> dict:
    started = time.perf_counter()
    first_round = int(decoder.sequence_step.item())
    result = decoder.generate(prompt, tokens, seed=seed, stop_at_eos=stop_at_eos)
    accepted = decoder.round_sequence[first_round: int(decoder.sequence_step.item())].tolist()
    ms = 1000 * (time.perf_counter() - started)
    record = speed_record(prompt, result.text, accepted, result.steady_rounds, len(result.token_ids), ms, result.steady_ms, result.steady_tokens)
    record["tokens_per_second"] = result.tokens_per_second
    return record


def ar_generator(args: argparse.Namespace) -> tuple[dict, Callable[[str, int], dict]]:
    dtype = args.ar_dtype or ar.AR_DTYPE[args.ar_sampler]
    scheme = (args.ar_scheme or args.model) if args.ar_sampler in ar.AR_SCHEME_KIND else None
    if scheme is not None and scheme_kind(scheme) != ar.AR_SCHEME_KIND[args.ar_sampler]:
        raise SystemExit(f"--ar-sampler {args.ar_sampler} needs a {ar.AR_SCHEME_KIND[args.ar_sampler]} scheme; {scheme} has {scheme_kind(scheme)}")
    llm = ar.build(args.model, dtype, args.max_model_len, scheme)
    described = {"decoding": f"ar-{args.ar_sampler}", "model": args.model, "dtype": dtype, "scheme": scheme,
                 "top_k": engine.TOP_K if args.ar_sampler == "topk" else None, "temperature": 0.0 if args.ar_sampler == "greedy" else 1.0}
    return described, lambda prompt, seed: ar.run(llm, prompt, ar.sampling(args.ar_sampler, args.tokens, args.stop_at_eos, seed))


def plm_generator(args: argparse.Namespace) -> tuple[dict, Callable[[str, int], dict]]:
    round_config = RoundConfig(model=args.model, setting=args.setting, mode=args.mode, match=args.match, k=args.k, max_model_len=args.max_model_len)
    decoder = round_config.build(engine.GPU_MEMORY_UTILIZATION, engine.KV_CACHE_BYTES, engine.KV_BLOCK_SIZE)
    described = {"decoding": round_config.match, **round_config.describe()}
    return described, lambda prompt, seed: run_plm(decoder, prompt, args.tokens, args.stop_at_eos, seed)


def build_generator(args: argparse.Namespace) -> tuple[dict, Callable[[str, int], dict]]:
    if args.eagle3:
        return eagle3.generate.generator(args)
    if args.fdv2:
        return fastdllmv2.generate.generator(args)
    return (ar_generator if args.ar else plm_generator)(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    eagle3.generate.add_arguments(parser)
    fastdllmv2.generate.add_arguments(parser)
    add_run_arguments(parser)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.prompts)
    described, generate_one = build_generator(args)
    run_passes(described, generate_one, dataset, args)


if __name__ == "__main__":
    main()
