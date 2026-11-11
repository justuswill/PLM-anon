"""Print the runs under a results directory: mean ± 95% CI over seeds.

    python -m plm.eval.table results/posttrain   # Table 11 rows: score, questions/s, tokens/s and tokens/call per benchmark
    python -m plm.eval.table results/pretrain    # tokens/s and tokens/call on the Figure 1 prompts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from plm.eval.stats import spread

BENCHMARKS = ("gsm8k", "math500", "humaneval", "ifeval")
COLUMNS = (("score", 1), ("questions_per_second", 2), ("tokens_per_second", 0), ("tokens_per_call", 2))


def cells(report: dict | None) -> list[str]:
    if report is None:
        return ["-"] * len(COLUMNS)
    return [f"{report['summary'][metric]['mean']:.{digits}f} ± {report['summary'][metric]['ci95']:.{digits}f}" for metric, digits in COLUMNS]


def print_posttrain(root: Path) -> None:
    print("| run | " + " | ".join(f"{name} score | Q/s | tok/s | tok/call" for name in BENCHMARKS) + " |")
    print("|---|" + "---|" * (len(COLUMNS) * len(BENCHMARKS)))
    for run in sorted(path for path in root.iterdir() if any(path.glob("*.score.json"))):
        row = []
        for name in BENCHMARKS:
            score = run / f"{name}.score.json"
            row += cells(json.loads(score.read_text()) if score.exists() else None)
        print(f"| {run.name} | " + " | ".join(row) + " |")


def print_pretrain(root: Path) -> None:
    print("| run | tok/s | tok/call | ms/call |")   # tok/s end to end, prefill included
    print("|---|---|---|---|")
    for results in sorted(root.glob("*/fineweb-edu.json")):
        passes = [measured["aggregate"] for measured in json.loads(results.read_text())["passes"]]
        cells = [spread([aggregate[metric] for aggregate in passes]) for metric in ("e2e_tokens_per_second", "tokens_per_call", "ms_per_call")]
        print(f"| {results.parent.name} | " + " | ".join(f"{cell['mean']:.{digits}f} ± {cell['ci95']:.{digits}f}" for cell, digits in zip(cells, (0, 2, 2))) + " |")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/posttrain")
    print_pretrain(root) if any(root.glob("*/fineweb-edu.json")) else print_posttrain(root)


if __name__ == "__main__":
    main()
