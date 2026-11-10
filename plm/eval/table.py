"""Print the graded runs under a results directory as Table 11 rows: mean ± 95% CI over seeds of score, questions/s,
tokens/s and tokens/call per benchmark.

    python -m plm.eval.table results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BENCHMARKS = ("gsm8k", "math500", "humaneval", "ifeval")
COLUMNS = (("score", 1), ("questions_per_second", 2), ("tokens_per_second", 0), ("tokens_per_call", 2))


def cells(report: dict | None) -> list[str]:
    if report is None:
        return ["-"] * len(COLUMNS)
    return [f"{report['summary'][metric]['mean']:.{digits}f} ± {report['summary'][metric]['ci95']:.{digits}f}" for metric, digits in COLUMNS]


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    print("| run | " + " | ".join(f"{name} score | Q/s | tok/s | tok/call" for name in BENCHMARKS) + " |")
    print("|---|" + "---|" * (len(COLUMNS) * len(BENCHMARKS)))
    for run in sorted(path for path in root.iterdir() if path.is_dir()):
        row = []
        for name in BENCHMARKS:
            score = run / f"{name}.score.json"
            row += cells(json.loads(score.read_text()) if score.exists() else None)
        print(f"| {run.name} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
