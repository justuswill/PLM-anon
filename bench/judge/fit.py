"""Bradley-Terry Elo from the judge's pairwise verdicts, with bootstrap intervals.

    python -m bench.judge.fit --judgments 'OUT/judgments/*.jsonl' --anchor ar-1.3b --out OUT/elo.json

A pair's two orderings become one outcome - a win only when both name the same winner,
otherwise a tie - the outcomes are fitted by maximum likelihood, and 95% intervals come from
resampling whole prompts. Writes --out, and the same table readable as <out>.md:

  anchor, bootstrap, pairs, prompts      what the fit used
  checks                                 swap consistency (both orderings agree), the judge's
                                         pull toward the first continuation (0.5 is none), the
                                         tie rate, and split-half Spearman: how well fits on two
                                         disjoint halves of the prompts agree on the ranking
  models                                 rank, model, elo, ci_low, ci_high, meetings, wins,
                                         losses, ties; best first
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

from bench.judge.common import (
    bootstrap_elo, fit_strengths, pair_outcomes, read_jsonl, spearman, to_elo, win_matrix, write_text,
)


def split_half(outcomes: list[dict], players: list[str], anchor: int, ridge: float, splits: int,
               seed: int) -> float:
    rng = np.random.default_rng(seed)
    prompts = sorted({o["prompt_id"] for o in outcomes})
    correlations = []
    for _ in range(splits):
        half = set(rng.permutation(prompts)[: len(prompts) // 2].tolist())
        parts = ([o for o in outcomes if o["prompt_id"] in half],
                 [o for o in outcomes if o["prompt_id"] not in half])
        elos = [to_elo(fit_strengths(win_matrix(part, players), ridge), anchor) for part in parts]
        correlations.append(spearman(*elos))
    return float(np.mean(correlations))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judgments", required=True, help="glob of bench.judge.judge outputs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--anchor", default="",
                        help="the model placed at 1000; default the first alphabetically")
    parser.add_argument("--bootstrap", type=int, default=1000, help="prompt resamples for the intervals")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--splits", type=int, default=20, help="random half splits for the split-half check")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.judgments))
    if not paths:
        raise SystemExit(f"no judgment files match {args.judgments}")
    calls = [call for path in paths for call in read_jsonl(path)]
    outcomes, stats = pair_outcomes(calls)
    if not outcomes:
        raise SystemExit("no pair has been judged in both orderings")
    players = sorted({name for o in outcomes for name in o["players"]})
    anchor_name = args.anchor if args.anchor in players else players[0]
    if args.anchor and anchor_name != args.anchor:
        print(f"anchor {args.anchor} is not among the models; anchoring {anchor_name} instead")
    anchor = players.index(anchor_name)

    wins = win_matrix(outcomes, players)
    games = wins + wins.T
    beta = fit_strengths(wins, args.ridge)
    elo = to_elo(beta, anchor)
    draws = bootstrap_elo(outcomes, players, anchor, args.bootstrap, args.seed, args.ridge, start=beta)
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)

    record = {name: [0, 0, 0] for name in players}
    for o in outcomes:
        x, y = o["players"]
        if o["winner"] is None:
            record[x][2] += 1
            record[y][2] += 1
        else:
            record[o["winner"]][0] += 1
            record[y if o["winner"] == x else x][1] += 1

    decided = [call["label"] for call in calls if call["label"] in ("A", "B")]
    checks = {
        "swap_consistency": stats["swap_consistency"],
        "first_position_rate": decided.count("A") / len(decided) if decided else None,
        "tie_rate": stats["tie_rate"],
        "unreadable_calls": sum(call["label"] not in ("A", "B", "C") for call in calls),
        "incomplete_pairs": stats["incomplete_pairs"],
        "split_half_spearman": split_half(outcomes, players, anchor, args.ridge, args.splits, args.seed),
    }
    models = [{"rank": rank, "model": players[k], "elo": float(elo[k]), "ci_low": float(low[k]),
               "ci_high": float(high[k]), "meetings": int(games[k].sum()), "wins": record[players[k]][0],
               "losses": record[players[k]][1], "ties": record[players[k]][2]}
              for rank, k in enumerate(np.argsort(-elo).tolist(), start=1)]
    result = {"anchor": anchor_name, "bootstrap": args.bootstrap, "pairs": stats["pairs"],
              "prompts": len({o["prompt_id"] for o in outcomes}), "checks": checks, "models": models}
    write_text(args.out, json.dumps(result, indent=1) + "\n")

    lines = [
        f"{stats['pairs']} pairs judged in both orderings over {result['prompts']} prompts. "
        f"`{anchor_name}` anchored at 1000; 95% intervals from {args.bootstrap} prompt resamples.", "",
        f"Swap consistency {checks['swap_consistency']:.1%} · "
        f"first-position rate {checks['first_position_rate']:.1%} (0.5 is unbiased) · "
        f"tied pairs {checks['tie_rate']:.1%} · split-half Spearman {checks['split_half_spearman']:.3f}", "",
        "| # | model | Elo | 95% CI | meetings | W-L-T |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {m['rank']} | `{m['model']}` | {m['elo']:.0f} | {m['ci_low']:.0f}-{m['ci_high']:.0f} | "
              f"{m['meetings']} | {m['wins']}-{m['losses']}-{m['ties']} |" for m in models]
    write_text(args.out.with_suffix(".md"), "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
