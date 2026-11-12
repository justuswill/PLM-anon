"""The Figure 1 table from a judged samples folder: each player's Elo next to its single-stream throughput.

    python -m bench.judge.table --samples results/pretrain/samples --elo results/pretrain/elo/elo.json \\
        --csv results/pretrain/figure1.csv

Throughput is end to end, prefill included: a player's generated tokens over its seconds, summed over the prompts of its
samples file. Prints the table as Markdown and, with --csv, also writes it as CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from bench.judge.common import read_jsonl


def throughput(samples_file: Path) -> float:
    rows = read_jsonl(samples_file)
    return sum(row["generated_tokens"] for row in rows) / sum(row["seconds"] for row in rows)


def joined_rows(samples: Path, elo: dict) -> list[dict]:
    rows = []
    for model in elo["models"]:
        samples_file = samples / f"{model['model']}.jsonl"
        if samples_file.exists():
            rows.append({**model, "tokens_per_second": throughput(samples_file)})
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["player", "elo", "ci_low", "ci_high", "tokens_per_second"])
        for row in rows:
            writer.writerow([row["model"], f"{row['elo']:.1f}", f"{row['ci_low']:.1f}", f"{row['ci_high']:.1f}", f"{row['tokens_per_second']:.1f}"])
    print(f"table -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", required=True, type=Path, help="the folder of <player>.jsonl files that was judged")
    parser.add_argument("--elo", required=True, type=Path, help="elo.json from scripts/judge_elo.sh")
    parser.add_argument("--csv", type=Path, default=None, help="also write the table to this CSV file")
    args = parser.parse_args()

    rows = sorted(joined_rows(args.samples, json.loads(args.elo.read_text())), key=lambda row: row["tokens_per_second"])
    print("| player | Elo | 95% interval | tok/s |")
    print("|---|---|---|---|")
    for row in rows:
        print(f"| {row['model']} | {row['elo']:.0f} | [{row['ci_low']:.0f}, {row['ci_high']:.0f}] | {row['tokens_per_second']:.0f} |")
    if args.csv is not None:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
