"""Answer extraction and checking: the last number (GSM8K), symbolic equivalence of the boxed answer (MATH-500),
sandboxed execution of the reference tests (HumanEval) and Google's IFEval instruction checkers."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
BOXED = re.compile(r"\\boxed\{")
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
IMPORT = re.compile(r"^(?:import|from)\s+\S+.*$", re.MULTILINE)
WHITESPACE = re.compile(r"\s+")
LATEX_WRAPPERS = ((r"\\left", ""), (r"\\right", ""), (r"\\!", ""), (r"\\,", ""), (r"\\ ", " "), (r"\\text\{([^}]*)\}", r"\1"),
                  (r"\\mbox\{([^}]*)\}", r"\1"), (r"\\dfrac", r"\\frac"), (r"\\tfrac", r"\\frac"))
LATEX_TRAILING = "$%.\\ "


# ------------------------------------------------------------------ GSM8K

def normalize_number(text: str) -> str:
    cleaned = text.replace(",", "").replace("$", "").rstrip(".")
    try:
        value = float(cleaned)
    except ValueError:
        return cleaned
    return str(int(value)) if value.is_integer() else str(value)


def last_number(text: str) -> str | None:
    matches = NUMBER.findall(text)
    return normalize_number(matches[-1]) if matches else None


def numbers_equal(predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(gold)) < 1e-6
    except ValueError:
        return predicted == gold


# ------------------------------------------------------------------ MATH-500

def last_boxed(text: str) -> str | None:
    starts = [match.end() for match in BOXED.finditer(text)]
    if not starts:
        return None
    depth = 1
    for index in range(starts[-1], len(text)):
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return text[starts[-1]:index]
    return None


def extract_answer(text: str) -> str | None:
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed
    return text.strip().rsplit("\n", 1)[-1] or None


def normalize_latex(text: str) -> str:
    for pattern, replacement in LATEX_WRAPPERS:
        text = re.sub(pattern, replacement, text)
    text = WHITESPACE.sub("", text).replace("dollars", "").replace("degrees", "")
    return text.strip(LATEX_TRAILING)


def answers_equal(predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False
    if normalize_latex(predicted) == normalize_latex(gold):
        return True
    from math_verify import parse, verify

    return bool(verify(parse(f"${gold}$"), parse(f"${predicted}$")))


# ------------------------------------------------------------------ HumanEval

def extract_definition(completion: str, defines: str) -> str:
    """The code block that defines the answer, with imports the model put in other blocks hoisted into it."""
    blocks = [block.strip() for block in FENCE.findall(completion)]
    if not blocks:
        return completion
    defining = [block for block in blocks if defines in block]
    chosen = defining[0] if defining else max(blocks, key=len)
    elsewhere = [line for block in blocks for line in IMPORT.findall(block)]
    missing = [line for line in dict.fromkeys(elsewhere) if line not in chosen]
    return "\n".join([*missing, chosen])


@dataclass(frozen=True)
class ExecutionResult:
    ok: bool
    output: str


SANDBOX_READ_ONLY_ROOTS = ("/usr", "/bin", "/lib", "/lib64", "/etc/alternatives")
SANDBOX_WORKDIR = "/work"
SANDBOX_SCRIPT = "program.py"


def rlimit_prologue(cpu_seconds: int, memory_bytes: int) -> str:
    """Hard rlimits locked in before any candidate code runs; they survive execve."""
    limits = {"RLIMIT_CPU": cpu_seconds, "RLIMIT_AS": memory_bytes, "RLIMIT_FSIZE": 16 << 20, "RLIMIT_NPROC": 64}
    settings = ", ".join(f"(_r.{name}, {value})" for name, value in limits.items())
    return f"import resource as _r\nfor _n, _v in ({settings},): _r.setrlimit(_n, (_v, _v))\ndel _r\n"


def interpreter_roots() -> list[str]:
    """Every installation the interpreter's symlink chain passes through: a virtualenv's python links into its base install, often via further links."""
    roots, path = [sys.prefix, sys.base_prefix], Path(sys.executable)
    while path.is_symlink():
        target = Path(os.readlink(path))
        path = target if target.is_absolute() else path.parent / target
        roots.append(str(path.parent.parent))
    return list(dict.fromkeys(roots))


def sandbox_argv(workdir: Path) -> tuple[list[str], str]:
    """Bubblewrap without network or a writable filesystem when the host has it, otherwise a plain subprocess under the rlimits."""
    bubblewrap = shutil.which("bwrap")
    if bubblewrap is None:
        return [sys.executable, "-I", "-B", str(workdir / SANDBOX_SCRIPT)], str(workdir)
    roots = [root for root in SANDBOX_READ_ONLY_ROOTS if Path(root).exists()] + interpreter_roots()
    binds = [argument for root in roots for argument in ("--ro-bind", root, root)]
    argv = [bubblewrap, "--unshare-all", "--die-with-parent", "--new-session", "--tmpfs", "/tmp", *binds,
            "--bind", str(workdir), SANDBOX_WORKDIR, "--proc", "/proc", "--dev", "/dev", "--chdir", SANDBOX_WORKDIR,
            sys.executable, "-I", "-B", f"{SANDBOX_WORKDIR}/{SANDBOX_SCRIPT}"]
    return argv, SANDBOX_WORKDIR


def run_python(source: str, timeout: float = 15.0, memory_bytes: int = 2 << 30) -> ExecutionResult:
    with tempfile.TemporaryDirectory(prefix="plm-eval-") as directory:
        workdir = Path(directory)
        (workdir / SANDBOX_SCRIPT).write_text(rlimit_prologue(int(timeout) + 1, memory_bytes) + source)
        argv, home = sandbox_argv(workdir)
        environment = {"PATH": "/usr/bin:/bin", "HOME": home, "TMPDIR": "/tmp", "OMP_NUM_THREADS": "1"}
        process = subprocess.Popen(argv, cwd=workdir, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            output, _ = process.communicate(timeout=timeout)
            return ExecutionResult(process.returncode == 0, output.decode(errors="replace")[-2000:])
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.communicate()
            return ExecutionResult(False, "timeout")


# ------------------------------------------------------------------ IFEval

@cache
def instruction_registry() -> Mapping[str, Any]:
    from instruction_following_eval.instructions_registry import INSTRUCTION_DICT
    from langdetect import DetectorFactory

    # langdetect samples internally; unseeded, the same response can pass a language check on one run and fail it on the next.
    DetectorFactory.seed = 0
    return INSTRUCTION_DICT


def follows(instruction_id: str, arguments: Mapping[str, Any], prompt: str, response: str) -> bool:
    checker = instruction_registry()[instruction_id](instruction_id)
    checker.build_description(**{key: value for key, value in arguments.items() if value is not None})
    if "prompt" in (checker.get_instruction_args() or {}):
        checker.build_description(prompt=prompt)
    return bool(response.strip()) and checker.check_following(response)


def strict_results(instruction_ids: Sequence[str], arguments: Sequence[Mapping[str, Any]], prompt: str, response: str) -> list[bool]:
    return [follows(instruction_id, argument, prompt, response) for instruction_id, argument in zip(instruction_ids, arguments)]
