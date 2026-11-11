"""Grade a ``plm.generate`` results file per seed and report the Table 11 columns: score, questions/s, tokens/s and tokens/call.

    python -m plm.eval.score --benchmark gsm8k --results results/plm-1.3B-beta1-exact/gsm8k.json --tasks prompts/gsm8k/tasks.json

Each continuation is cut at the end-of-text token and at the benchmark's stop strings, then stripped, before grading.
Every metric is reported per seed and as the mean over seeds, with the sample standard deviation and a two-sided 95%
Student-t confidence interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plm.eval.benchmarks import BENCHMARKS, Task, mean_score
from plm.eval.stats import spread

EOS = "<|endoftext|>"
METRICS = ("score", "questions_per_second", "tokens_per_second", "tokens_per_call")


def answer_text(continuation: str, stop: tuple[str, ...]) -> str:
    text = continuation.split(EOS, 1)[0]
    cuts = [text.index(marker) for marker in stop if marker in text]
    return (text[: min(cuts)] if cuts else text).strip()


def grade_pass(benchmark: str, tasks: list[Task], measured: dict) -> dict:
    records = measured["results"]
    grades = [BENCHMARKS[benchmark].grade(task, answer_text(record["continuation"], task.stop)) for task, record in zip(tasks, records)]
    seconds = sum(record["ms"] for record in records) / 1000
    return {"seed": measured["seed"], "score": 100 * mean_score(grades), "questions_per_second": len(records) / seconds,
            "tokens_per_second": sum(record["tokens"] for record in records) / seconds, "tokens_per_call": measured["aggregate"]["tokens_per_call"],
            "grades": [{"task_id": task.task_id, "score": grade.score, **grade.detail} for task, grade in zip(tasks, grades)]}



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, choices=tuple(BENCHMARKS))
    parser.add_argument("--results", required=True, help="a plm.generate results file")
    parser.add_argument("--tasks", required=True, help="build_prompts.py's tasks.json for this benchmark")
    parser.add_argument("--out", default="", help="where to write the graded summary; default <results>.score.json")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())
    manifest = json.loads(Path(args.tasks).read_text())
    tasks = BENCHMARKS[args.benchmark].tasks()
    if [task.task_id for task in tasks] != manifest["task_ids"]:
        raise SystemExit("the benchmark's tasks no longer match tasks.json: rebuild the prompts")
    seeds = [grade_pass(args.benchmark, tasks, measured) for measured in results["passes"]]
    summary = {metric: spread([graded[metric] for graded in seeds]) for metric in METRICS}
    report = {"benchmark": args.benchmark, "metric": BENCHMARKS[args.benchmark].metric, "tasks": len(seeds[0]["grades"]),
              "config": results["config"], "summary": summary, "seeds": seeds}
    out = Path(args.out or f"{args.results.removesuffix('.json')}.score.json")
    out.write_text(json.dumps(report, indent=1))
    print(f"  {args.benchmark}: {report['metric']} " + "  ".join(f"{metric} {summary[metric]['mean']:.2f} ± {summary[metric]['ci95']:.2f}" for metric in METRICS)
          + f"  ({len(seeds)} seeds, {report['tasks']} tasks) -> {out}")


if __name__ == "__main__":
    main()
