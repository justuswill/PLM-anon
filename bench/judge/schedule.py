"""Which two models meet on which prompt, for the LLM-judge Elo.

    python -m bench.judge.schedule --samples SAMPLES --per-player 400 --seed 0 --out OUT/schedule.jsonl

A new schedule is built in rounds. Each round shuffles the field and pairs it off, so after R
rounds every model has met R opponents drawn uniformly from the whole field, which keeps the
comparison graph connected. Each meeting gets a prompt drawn at random, never repeating a
(pair, prompt) combination. With an odd field one model sits each round out, in turn.

When --out already exists it is kept, so the pairs already judged stay in the fit. Models in
the folder that the schedule has never seen get --per-player new meetings each, against
opponents drawn uniformly from the whole field, numbered after the existing pairs. Models the
schedule names but the folder no longer has are left in the file and skipped by the judge and
the fit. Delete the file to draw a fresh schedule.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from bench.judge.common import read_jsonl, read_samples, write_jsonl


def draw_prompt(rng: random.Random, prompts: list[str], used: set, a: str, b: str) -> str:
    key = tuple(sorted((a, b)))
    for _ in range(10_000):
        prompt = rng.choice(prompts)
        if (key, prompt) not in used:
            used.add((key, prompt))
            return prompt
    raise SystemExit(f"{a} and {b} have already met on every prompt; lower the meetings per model")


def fresh(models: list[str], prompts: list[str], per_player: int, rng: random.Random,
          used: set) -> list[tuple]:
    meetings = []
    for round_number in range(per_player):
        field = list(models)
        if len(field) % 2:
            field.remove(models[round_number % len(models)])  # sits this round out, in turn
        rng.shuffle(field)
        meetings += [(a, b, draw_prompt(rng, prompts, used, a, b)) for a, b in zip(field[::2], field[1::2])]
    return meetings


def extend(new: list[str], models: list[str], prompts: list[str], per_player: int, rng: random.Random,
           used: set) -> list[tuple]:
    meetings = []
    for model in new:
        others = [other for other in models if other != model]
        for _ in range(per_player):
            opponent = rng.choice(others)
            a, b = (model, opponent) if rng.random() < 0.5 else (opponent, model)
            meetings.append((a, b, draw_prompt(rng, prompts, used, a, b)))
    return meetings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, required=True, help="one <model>.jsonl per model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-player", type=int, default=400, help="meetings per model")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    samples, prompts = read_samples(args.samples)
    models = sorted(samples)
    existing = read_jsonl(args.out) if args.out.exists() else []
    stray = {row["prompt_id"] for row in existing} - set(prompts)
    if stray:
        raise SystemExit(f"{args.out} uses prompts the samples do not have, e.g. {sorted(stray)[:3]}; the "
                         "samples answer a different prompt set, so delete the schedule to draw a new one")
    seen = {name for row in existing for name in row["pair"]}
    new = [model for model in models if model not in seen]
    if existing and not new:
        print(f"keeping {args.out}: {len(existing)} pairs, every model already scheduled")
        return

    # A seed per extension, so adding the same models to the same schedule twice draws the same pairs.
    rng = random.Random(f"{args.seed}:{len(existing)}")
    used = {(tuple(sorted(row["pair"])), row["prompt_id"]) for row in existing}
    meetings = extend(new, models, prompts, args.per_player, rng, used) if existing else \
        fresh(models, prompts, args.per_player, rng, used)
    rng.shuffle(meetings)  # so any prefix of the schedule is a uniform sample of it
    start = max((row["pair_id"] for row in existing), default=-1) + 1
    rows = existing + [{"pair_id": start + k, "prompt_id": prompt, "pair": [a, b]}
                       for k, (a, b, prompt) in enumerate(meetings)]
    write_jsonl(args.out, rows)

    counts = Counter(name for row in rows for name in row["pair"] if name in samples)
    meta = {"models": models, "prompts": len(prompts), "pairs": len(rows), "per_player": args.per_player,
            "seed": args.seed,
            "meetings_per_model": {"min": min(counts.values()), "max": max(counts.values())}}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    what = f"added {len(meetings)} pairs for {len(new)} new models" if existing else \
        f"drew {len(meetings)} pairs"
    print(f"{what} -> {args.out}: {len(rows)} pairs over {len(models)} models and {len(prompts)} prompts, "
          f"meetings per model {meta['meetings_per_model']['min']}..{meta['meetings_per_model']['max']}")


if __name__ == "__main__":
    main()
