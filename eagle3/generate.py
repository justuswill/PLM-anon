"""``plm.generate`` for the EAGLE-3 baseline: the same warm-up pass, seeds and results file, decoding an autoregressive
target with an EAGLE-3 draft head at temperature 1 over the target's 128 likeliest tokens.

    python -m eagle3.generate --model checkpoints/posttrain/ar-1.3B --head checkpoints/posttrain/eagle3-ar-1.3B \\
        --dataset prompts/gsm8k/prompts.json --tokens 512 --stop-at-eos --seeds 0 1 2 --out results.json
    python -m eagle3.generate ... --tree

The default is vLLM's speculative decoding, a chain of ``--k`` drafted tokens per target call. ``--tree`` decodes with
EAGLE-2's dynamic draft tree instead (``eagle3.tree``). Both are exact: the output distribution is the target's.
"""

from __future__ import annotations

import argparse
import zlib
from collections.abc import Callable

from plm import config as plm_config
from plm.generate import add_run_arguments, load_dataset, run_passes, speed_record


def tree_generator(args: argparse.Namespace) -> tuple[dict, Callable[[str, int], dict]]:
    from eagle3.tree import TreeDecoder, build_llm

    if not args.stop_at_eos:
        raise SystemExit("--tree decodes until the end-of-text token: pass --stop-at-eos")
    decoder = TreeDecoder(build_llm(args.model, max_model_len=args.max_model_len), args.head, total_token=args.tree_nodes, depth=args.tree_depth,
                          top_k=args.tree_top_k, sampler_top_k=plm_config.TOP_K)
    decoder.prepare()

    def generate_one(prompt: str, seed: int) -> dict:
        # Seeded per prompt so a pass does not depend on what ran before it in the process.
        result = decoder.generate(prompt, args.tokens, seed=seed * 2**32 + zlib.crc32(prompt.encode()))
        ms = 1000 * result.seconds
        return speed_record(prompt, result.text, [], 1 + result.steps, len(result.token_ids), ms, ms, len(result.token_ids))

    described = {"decoding": "eagle3-tree", "nodes": args.tree_nodes, "depth": args.tree_depth, "top_k_per_node": args.tree_top_k}
    return described, generate_one


def chain_generator(args: argparse.Namespace) -> tuple[dict, Callable[[str, int], dict]]:
    from eagle3 import chain

    llm = chain.build(args.model, args.head, args.k, args.draft, args.max_model_len)
    described = {"decoding": "eagle3-chain", "num_speculative_tokens": args.k, "draft": args.draft}
    return described, lambda prompt, seed: chain.run(llm, prompt, chain.sampling(args.tokens, args.stop_at_eos, seed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="the autoregressive target, e.g. checkpoints/posttrain/ar-1.3B")
    parser.add_argument("--head", required=True, help="an EAGLE-3 head trained on --model, e.g. checkpoints/posttrain/eagle3-ar-1.3B")
    parser.add_argument("--max-model-len", type=int, default=4096, help="prompt plus generated tokens")
    parser.add_argument("--k", type=int, default=4, help="chain: tokens drafted per target call; 4 was fastest of 4, 5 and 6 on the Table 11 benchmarks")
    parser.add_argument("--draft", default="greedy", choices=("greedy", "sample", "topk"),
                        help="chain: the head's argmax; a sample of its full distribution; or a sample of its 128 likeliest tokens")
    parser.add_argument("--tree", action="store_true", help="EAGLE-2's dynamic draft tree instead of a chain; needs --stop-at-eos")
    parser.add_argument("--tree-nodes", type=int, default=60, help="tree: nodes verified per target call (upstream EAGLE's total_token)")
    parser.add_argument("--tree-depth", type=int, default=7, help="tree: draft levels expanded per call")
    parser.add_argument("--tree-top-k", type=int, default=10, help="tree: children expanded per node and frontier width")
    add_run_arguments(parser)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.prompts)
    described, generate_one = (tree_generator if args.tree else chain_generator)(args)
    described = {**described, "model": args.model, "head": args.head, "dtype": "bfloat16", "top_k": plm_config.TOP_K, "temperature": 1.0}
    run_passes(described, generate_one, dataset, args)


if __name__ == "__main__":
    main()
