"""The EAGLE-3 baseline's decodings for ``bench.generate --eagle3 <head>``: an autoregressive target with an EAGLE-3 draft
head, at temperature 1 over the target's 128 likeliest tokens.

    python -m bench.generate --model checkpoints/posttrain/ar-1.3B --eagle3 checkpoints/posttrain/eagle3-ar-1.3B \\
        --dataset prompts/gsm8k/prompts.json --tokens 512 --stop-at-eos --seeds 0 1 2 --out results.json
    python -m bench.generate ... --eagle3-tree

The default is vLLM's speculative decoding, a chain of ``--eagle3-k`` drafted tokens per target call. ``--eagle3-tree``
decodes with EAGLE-2's dynamic draft tree instead (``eagle3.tree``). Both are exact: the output distribution is the target's.
"""

from __future__ import annotations

import argparse
import zlib
from collections.abc import Callable

from bench import engine
from bench.harness import speed_record


def tree_generator(args: argparse.Namespace) -> tuple[dict, Callable[[str, int], dict]]:
    from eagle3.tree import TreeDecoder, build_llm

    if not args.stop_at_eos:
        raise SystemExit("--eagle3-tree decodes until the end-of-text token: pass --stop-at-eos")
    decoder = TreeDecoder(build_llm(args.model, max_model_len=args.max_model_len), args.eagle3, total_token=args.eagle3_tree_nodes, depth=args.eagle3_tree_depth,
                          top_k=args.eagle3_tree_top_k, sampler_top_k=engine.TOP_K)
    decoder.prepare()

    def generate_one(prompt: str, seed: int) -> dict:
        # Seeded per prompt so a pass does not depend on what ran before it in the process.
        result = decoder.generate(prompt, args.tokens, seed=seed * 2**32 + zlib.crc32(prompt.encode()))
        ms = 1000 * result.seconds
        return speed_record(prompt, result.text, [], 1 + result.steps, len(result.token_ids), ms, ms, len(result.token_ids))

    described = {"decoding": "eagle3-tree", "nodes": args.eagle3_tree_nodes, "depth": args.eagle3_tree_depth, "top_k_per_node": args.eagle3_tree_top_k}
    return described, generate_one


def chain_generator(args: argparse.Namespace) -> tuple[dict, Callable[[str, int], dict]]:
    from eagle3 import chain

    llm = chain.build(args.model, args.eagle3, args.eagle3_k, args.eagle3_draft, args.max_model_len)
    described = {"decoding": "eagle3-chain", "num_speculative_tokens": args.eagle3_k, "draft": args.eagle3_draft}
    return described, lambda prompt, seed: chain.run(llm, prompt, chain.sampling(args.tokens, args.stop_at_eos, seed))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("EAGLE-3 baseline")
    group.add_argument("--eagle3", default="", help="an EAGLE-3 head trained on --model, e.g. checkpoints/posttrain/eagle3-ar-1.3B: decode with it")
    group.add_argument("--eagle3-k", type=int, default=4, help="chain: tokens drafted per target call; 4 was fastest of 4, 5 and 6 on the Table 11 benchmarks")
    group.add_argument("--eagle3-draft", default="greedy", choices=("greedy", "sample", "topk"),
                       help="chain: the head's argmax; a sample of its full distribution; or a sample of its 128 likeliest tokens")
    group.add_argument("--eagle3-tree", action="store_true", help="EAGLE-2's dynamic draft tree instead of a chain; needs --stop-at-eos")
    group.add_argument("--eagle3-tree-nodes", type=int, default=60, help="tree: nodes verified per target call (upstream EAGLE's total_token)")
    group.add_argument("--eagle3-tree-depth", type=int, default=7, help="tree: draft levels expanded per call")
    group.add_argument("--eagle3-tree-top-k", type=int, default=10, help="tree: children expanded per node and frontier width")


def generator(args: argparse.Namespace) -> tuple[dict, Callable[[str, int], dict]]:
    described, generate_one = (tree_generator if args.eagle3_tree else chain_generator)(args)
    return {**described, "model": args.model, "head": args.eagle3, "dtype": "bfloat16", "top_k": engine.TOP_K, "temperature": 1.0}, generate_one
