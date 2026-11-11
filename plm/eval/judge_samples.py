"""Rewrite one pass of a ``plm.generate`` results file as a samples file for the pairwise LLM judge of Figure 1.

    python -m plm.eval.judge_samples --results results/fig1/eagle3-1.3B.json --prompts data/fineweb-edu/prompts.jsonl \\
        --model vllm_eagle3_1.3b_pre --out results/fig1/samples/vllm_eagle3_1.3b_pre.jsonl

One JSON object a line: ``model``, ``prompt_id``, ``prompt`` and ``text`` are what the judge reads; ``generated_tokens``,
``seconds`` and ``nfe`` (model calls) carry the speed. The prompts file supplies ``prompt_id``; it is joined to the
results by position and checked on the prompt text, so the results must come from ``data/fineweb-edu/prompts.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_prompts(path: Path) -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [(row["prompt_id"], row["prompt"]) for row in rows]


def select_pass(results: dict, seed: int | None) -> list[dict]:
    passes = results["passes"]
    if seed is None:
        if len(passes) != 1:
            raise SystemExit(f"{len(passes)} measured passes: choose one with --seed")
        return passes[0]["results"]
    matching = [measured for measured in passes if measured["seed"] == seed]
    if not matching:
        raise SystemExit(f"no pass with seed {seed}")
    return matching[0]["results"]


def build_samples(records: list[dict], prompts: list[tuple[str, str]], model: str) -> list[dict]:
    if len(records) != len(prompts):
        raise SystemExit(f"{len(records)} results against {len(prompts)} prompts: not the same dataset")
    if any(record["prompt"] != prompt for (_, prompt), record in zip(prompts, records)):
        raise SystemExit("results and prompts disagree on prompt text: they are not aligned")
    return [{"model": model, "prompt_id": prompt_id, "prompt": prompt, "text": record["continuation"], "generated_tokens": record["tokens"],
             "seconds": record["ms"] / 1000, "nfe": record["calls"]} for (prompt_id, prompt), record in zip(prompts, records)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, type=Path, help="a plm.generate results file")
    parser.add_argument("--prompts", required=True, type=Path, help="data/fineweb-edu/prompts.jsonl")
    parser.add_argument("--model", required=True, help="the player name; also the output file's stem")
    parser.add_argument("--seed", type=int, default=None, help="which measured pass; needed only when the file has several")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    samples = build_samples(select_pass(json.loads(args.results.read_text()), args.seed), load_prompts(args.prompts), args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples))
    print(f"{len(samples)} samples -> {args.out}")


if __name__ == "__main__":
    main()
