"""The measurement harness every decoding's CLI shares: the dataset, a warm-up pass and one pass per seed, per-prompt
speed records, and the results file."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path


def load_dataset(path: str, count: int | None) -> list[str]:
    loaded = json.loads(Path(path).read_text())
    if not isinstance(loaded, list) or not all(isinstance(prompt, str) for prompt in loaded):
        raise SystemExit(f"{path}: expected a JSON list of prompt strings")
    return loaded if count is None else loaded[:count]


def speed_record(prompt: str, continuation: str, accepted: list[int], calls: int, tokens: int, ms: float, steady_ms: float, steady_tokens: int) -> dict:
    return {"prompt": prompt, "continuation": continuation, "accepted_per_call": accepted, "calls": calls, "tokens": tokens, "ms": ms,
            "ms_per_call": steady_ms / calls if calls else 0.0, "tokens_per_call": steady_tokens / calls if calls else 0.0,
            "tokens_per_second": tokens / (ms / 1000) if ms else 0.0}


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


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-model-len", type=int, default=4096, help="prompt plus generated tokens")
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
