"""GSM8K, MATH-500, HumanEval and IFEval: each benchmark's tasks (prompt, answer budget, stop strings, reference) and grader."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Callable

from datasets import load_dataset

from plm.eval import grading

GSM8K_INSTRUCTION = ("Solve the following grade school math problem. Reason step by step, then write the final numeric answer on its own "
                     'last line in the form "The answer is <number>."')
MATH500_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
HUMANEVAL_INSTRUCTION = ("Read the following function signature and docstring, and fully implement the function described. "
                         "Reply with a single Python code block containing the complete function and nothing else.")


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    budget: int
    reference: Any
    stop: tuple[str, ...] = ()


@dataclass(frozen=True)
class Grade:
    score: float
    detail: dict = field(default_factory=dict)


def gsm8k_tasks() -> list[Task]:
    rows = load_dataset("openai/gsm8k", "main", split="test")
    return [Task(f"gsm8k/{index}", f"{GSM8K_INSTRUCTION}\n\n{row['question']}", 512, row["answer"].rsplit("####", 1)[-1].strip().replace(",", ""),
                 stop=("Question:",)) for index, row in enumerate(rows)]


def math500_tasks() -> list[Task]:
    rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [Task(row["unique_id"], f"{row['problem']}\n\n{MATH500_INSTRUCTION}", 2048, row["answer"]) for row in rows]


def humaneval_tasks() -> list[Task]:
    rows = load_dataset("openai/openai_humaneval", split="test")
    return [Task(row["task_id"], f"{HUMANEVAL_INSTRUCTION}\n\n```python\n{row['prompt']}```", 768,
                 {"prompt": row["prompt"], "test": row["test"], "entry_point": row["entry_point"]}) for row in rows]


def ifeval_tasks() -> list[Task]:
    rows = load_dataset("google/IFEval", split="train")
    return [Task(f"ifeval/{row['key']}", row["prompt"], 1280,
                 {"prompt": row["prompt"], "instruction_ids": row["instruction_id_list"], "arguments": row["kwargs"]}) for row in rows]


def grade_gsm8k(task: Task, text: str) -> Grade:
    predicted = grading.last_number(text)
    return Grade(float(grading.numbers_equal(predicted, task.reference)), {"predicted": predicted, "gold": task.reference})


def grade_math500(task: Task, text: str) -> Grade:
    predicted = grading.extract_answer(text)
    return Grade(float(grading.answers_equal(predicted, task.reference)), {"predicted": predicted, "gold": task.reference})


def grade_humaneval(task: Task, text: str) -> Grade:
    # The original prompt goes ahead of the model's code so helpers it defines stay in scope; the model's own definition comes later and wins.
    entry_point = task.reference["entry_point"]
    solution = grading.extract_definition(text, f"def {entry_point}")
    program = "\n\n".join([task.reference["prompt"], solution, task.reference["test"], f"check({entry_point})\n"])
    result = grading.run_python(program)
    return Grade(float(result.ok), {"output": result.output})


def grade_ifeval(task: Task, text: str) -> Grade:
    reference = task.reference
    strict = grading.strict_results(reference["instruction_ids"], reference["arguments"], reference["prompt"], text)
    return Grade(float(all(strict)), {"strict": strict})


@dataclass(frozen=True)
class Benchmark:
    tasks: Callable[[], list[Task]]
    grade: Callable[[Task, str], Grade]
    metric: str


BENCHMARKS = {
    "gsm8k": Benchmark(gsm8k_tasks, grade_gsm8k, "accuracy"),
    "math500": Benchmark(math500_tasks, grade_math500, "accuracy"),
    "humaneval": Benchmark(humaneval_tasks, grade_humaneval, "pass@1"),
    "ifeval": Benchmark(ifeval_tasks, grade_ifeval, "prompt-level strict accuracy"),
}


def mean_score(grades: list[Grade]) -> float:
    return fmean(grade.score for grade in grades)
