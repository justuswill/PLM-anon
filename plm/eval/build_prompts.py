"""Write one benchmark's prompts in the plain template the models were post-trained on, plus its task manifest.

    python -m plm.eval.build_prompts --benchmark gsm8k --tokenizer checkpoints/posttrain/plm-1.3B-beta1 --out-dir prompts/gsm8k

Writes ``prompts.json`` (the list ``plm.generate --dataset`` takes) and ``tasks.json`` (task ids and the answer budget).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from plm.eval.benchmarks import BENCHMARKS, Task

HEADROOM = 64


def plain_prompt(task: Task) -> str:
    return f"User: {task.question}\n\nAssistant:"


def check_budget(tasks: list[Task], prompts: list[str], tokenizer_path: str, max_model_len: int) -> int:
    """Generation takes one budget per run, so refuse a benchmark whose budget would be clipped for any prompt."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    longest = max(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts)
    budget = tasks[0].budget
    if longest + max(budget, HEADROOM) > max_model_len:
        raise SystemExit(f"longest prompt is {longest} tokens; {budget} more exceeds --max-model-len {max_model_len}")
    return budget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, choices=tuple(BENCHMARKS))
    parser.add_argument("--tokenizer", required=True, help="any downloaded model directory (all share the GPT-2 tokenizer)")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    tasks = BENCHMARKS[args.benchmark].tasks()
    prompts = [plain_prompt(task) for task in tasks]
    budget = check_budget(tasks, prompts, args.tokenizer, args.max_model_len)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompts.json").write_text(json.dumps(prompts, indent=1))
    (out_dir / "tasks.json").write_text(json.dumps({"benchmark": args.benchmark, "budget": budget, "task_ids": [task.task_id for task in tasks]}, indent=1))
    print(f"  {len(tasks)} tasks, budget {budget} -> {out_dir}")


if __name__ == "__main__":
    main()
