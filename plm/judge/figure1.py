"""Figure 1 from a judged samples folder: each player's Elo against its single-stream throughput.

    python -m plm.judge.figure1 --samples results/pretrain/samples --elo results/pretrain/elo/elo.json \\
        --out results/pretrain/figure1.png

Throughput is end to end, prefill included: a player's generated tokens over its seconds, summed over the prompts of its
samples file. Prints the joined table as Markdown and, with --out, draws Elo (95% interval) against tokens per second
on a log axis, one colour per decoding and one marker per model size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm.judge.common import read_jsonl

DECODINGS = ("ar_topk", "ar_voronoi", "exact", "relaxed", "eagle3")


def throughput(samples_file: Path) -> float:
    rows = read_jsonl(samples_file)
    return sum(row["generated_tokens"] for row in rows) / sum(row["seconds"] for row in rows)


def decoding_of(player: str) -> str:
    return next((decoding for decoding in DECODINGS if player.endswith(f"_pre_{decoding}")), "other")


def size_of(player: str) -> str:
    return next((size for size in ("3b", "1.3b", "350m") if f"_{size}_" in player), "other")


def joined_rows(samples: Path, elo: dict) -> list[dict]:
    rows = []
    for model in elo["models"]:
        samples_file = samples / f"{model['model']}.jsonl"
        if samples_file.exists():
            rows.append({**model, "tokens_per_second": throughput(samples_file)})
    return rows


def draw(rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = dict(zip(DECODINGS + ("other",), plt.get_cmap("tab10").colors))
    markers = {"1.3b": "o", "3b": "s", "350m": "^", "other": "x"}
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for row in rows:
        decoding, size = decoding_of(row["model"]), size_of(row["model"])
        axis.errorbar(row["tokens_per_second"], row["elo"], yerr=[[row["elo"] - row["ci_low"]], [row["ci_high"] - row["elo"]]],
                      fmt=markers[size], color=colours[decoding], capsize=2, markersize=6)
    for decoding in sorted({decoding_of(row["model"]) for row in rows}, key=(DECODINGS + ("other",)).index):
        axis.plot([], [], "o", color=colours[decoding], label=decoding.replace("_", "-"))
    for size in sorted({size_of(row["model"]) for row in rows} - {"other"}, key=list(markers).index):
        axis.plot([], [], markers[size], color="gray", label=size.upper())
    axis.set_xscale("log")
    axis.set_xlabel("tokens / s (single stream, end to end)")
    axis.set_ylabel("Elo (LLM judge)")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200)
    print(f"figure -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", required=True, type=Path, help="the folder of <player>.jsonl files that was judged")
    parser.add_argument("--elo", required=True, type=Path, help="elo.json from scripts/judge_elo.sh")
    parser.add_argument("--out", type=Path, default=None, help="where to draw the figure (png or pdf); table only if omitted")
    args = parser.parse_args()

    rows = sorted(joined_rows(args.samples, json.loads(args.elo.read_text())), key=lambda row: row["tokens_per_second"])
    print("| player | Elo | 95% interval | tok/s |")
    print("|---|---|---|---|")
    for row in rows:
        print(f"| {row['model']} | {row['elo']:.0f} | [{row['ci_low']:.0f}, {row['ci_high']:.0f}] | {row['tokens_per_second']:.0f} |")
    if args.out is not None:
        draw(rows, args.out)


if __name__ == "__main__":
    main()
