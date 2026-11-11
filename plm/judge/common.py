"""What the LLM-judge Elo scripts share: reading samples and judgments, and the Bradley-Terry fit.

Numpy only, so scheduling and fitting run anywhere, the login node included.

Scale: Bradley-Terry strengths beta, with P(i beats j) = sigmoid(beta_i - beta_j), reported
Elo-style as 1000 + (400 / ln 10) * (beta - beta_anchor). The anchor sits at 1000 and a
100-point gap means the stronger model wins 64% of the time.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ELO_PER_LOGIT = 400 / math.log(10)
ANCHOR_ELO = 1000.0


def read_jsonl(path: str | Path) -> list[dict]:
    """Rows of a JSON-lines file. A half-written last line - a worker still appending, or a job
    cut off mid-write - is skipped with a warning; anywhere else it is an error."""
    with open(path) as handle:
        lines = handle.read().splitlines()
    rows = []
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if number != len(lines) - 1:
                raise
            print(f"warning: skipped a partial last line in {path}", file=sys.stderr)
    return rows


def write_text(path: Path, text: str) -> None:
    """Write through a temporary file and rename it into place. Rewriting a file in place on /users has left the old
    file's tail after shorter new content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def write_jsonl(path: Path, rows) -> None:
    write_text(path, "".join(json.dumps(row) + "\n" for row in rows))


def read_samples(folder: Path) -> tuple[dict[str, dict[str, dict]], list[str]]:
    """model -> prompt_id -> row, for every <model>.jsonl in the folder, and the prompt ids.

    Every model must answer the same prompts with the same opening, and every row needs
    prompt_id, prompt and text. An empty text is allowed and judged as it is: a model that
    sometimes stops at once should pay for it.
    """
    files = sorted(folder.glob("*.jsonl"))
    if len(files) < 2:
        raise SystemExit(f"{folder} needs at least two <model>.jsonl files, found {len(files)}")
    samples: dict[str, dict[str, dict]] = {}
    for path in files:
        rows = {}
        for number, row in enumerate(read_jsonl(path), start=1):
            missing = {"prompt_id", "prompt", "text"} - set(row)
            if missing:
                raise SystemExit(f"{path.name} line {number} has no {sorted(missing)}")
            if not isinstance(row["text"], str):
                raise SystemExit(f"{path.name} line {number}: text is not a string")
            if str(row["prompt_id"]) in rows:
                raise SystemExit(f"{path.name} answers prompt {row['prompt_id']} twice")
            rows[str(row["prompt_id"])] = row
        samples[path.stem] = rows
    models = sorted(samples)
    prompts = sorted(samples[models[0]])
    for model in models[1:]:
        if sorted(samples[model]) != prompts:
            gap = set(prompts) ^ set(samples[model])
            raise SystemExit(f"{model} and {models[0]} answer different prompts, e.g. {sorted(gap)[:3]}")
        for prompt_id in prompts:
            if samples[model][prompt_id]["prompt"] != samples[models[0]][prompt_id]["prompt"]:
                raise SystemExit(f"{model} and {models[0]} have different openings for prompt {prompt_id}")
    return samples, prompts


def winner(call: dict) -> str | None:
    """The model one call preferred, or None when it called the pair even (or was unreadable)."""
    return {"A": call["first"], "B": call["second"]}.get(call["label"])


def pair_outcomes(calls: list[dict]) -> tuple[list[dict], dict]:
    """One outcome per pair, from the two orderings it was judged in.

    A pair is a win only when both orderings name the same winner. Any disagreement -
    including one ordering calling it even - is a tie, so a judge that simply favours
    whichever continuation it reads first cannot manufacture wins. Counting a split as a tie
    does not bias the score when the judge never calls a tie itself: favoured with probability
    p in each ordering, a pair wins both p^2 of the time and splits 2p(1-p), a score of p.
    A pair with only one ordering judged is left out and counted.
    """
    orderings: dict[int, dict[str, dict]] = {}
    for call in calls:
        orderings.setdefault(call["pair_id"], {})[call["order"]] = call
    outcomes, incomplete = [], 0
    for pair_id in sorted(orderings):
        both = orderings[pair_id]
        if set(both) != {"ab", "ba"}:
            incomplete += 1
            continue
        forward, backward = winner(both["ab"]), winner(both["ba"])
        outcomes.append({
            "pair_id": pair_id,
            "prompt_id": both["ab"]["prompt_id"],
            "players": list(both["ab"]["pair"]),
            "winner": forward if forward == backward else None,
            "consistent": forward == backward,
        })
    count = len(outcomes)
    stats = {
        "pairs": count,
        "incomplete_pairs": incomplete,
        "swap_consistency": sum(o["consistent"] for o in outcomes) / count if count else None,
        "tie_rate": sum(o["winner"] is None for o in outcomes) / count if count else None,
    }
    return outcomes, stats


def add_outcome(wins: np.ndarray, outcome: dict, index: dict[str, int]) -> None:
    """Credit one outcome to wins[i, j] (times i beat j), a tie half to each side."""
    i, j = (index[name] for name in outcome["players"])
    if outcome["winner"] is None:
        wins[i, j] += 0.5
        wins[j, i] += 0.5
    elif outcome["winner"] == outcome["players"][0]:
        wins[i, j] += 1.0
    else:
        wins[j, i] += 1.0


def win_matrix(outcomes: list[dict], players: list[str]) -> np.ndarray:
    index = {name: k for k, name in enumerate(players)}
    wins = np.zeros((len(players), len(players)))
    for outcome in outcomes:
        add_outcome(wins, outcome, index)
    return wins


def fit_strengths(wins: np.ndarray, ridge: float = 1e-3, start: np.ndarray | None = None,
                  iterations: int = 200, tolerance: float = 1e-10) -> np.ndarray:
    """Bradley-Terry strengths maximising the likelihood of wins[i, j] = times i beat j.

    Newton's method with a backtracking line search. The small ridge gives a model that
    never lost a finite strength and pins the otherwise free overall level; at these sample
    sizes it moves a well-measured strength by far less than its bootstrap interval.
    """
    n = wins.shape[0]
    games = wins + wins.T
    beta = np.zeros(n) if start is None else np.array(start, dtype=float)

    def objective(b: np.ndarray) -> float:
        gaps = b[:, None] - b[None, :]
        return float(-(wins * np.logaddexp(0.0, -gaps)).sum() - 0.5 * ridge * (b @ b))

    value = objective(beta)
    for _ in range(iterations):
        gaps = beta[:, None] - beta[None, :]
        p = 0.5 * (1.0 + np.tanh(0.5 * gaps))  # sigmoid, without overflow
        gradient = (wins - games * p).sum(axis=1) - ridge * beta
        curvature = games * p * (1.0 - p)
        hessian = curvature - np.diag(curvature.sum(axis=1)) - ridge * np.eye(n)
        step = np.linalg.solve(hessian, gradient)
        scale = 1.0
        while True:
            candidate = beta - scale * step
            new_value = objective(candidate)
            if new_value >= value - 1e-12 or scale < 1e-8:
                break
            scale *= 0.5
        moved = float(np.max(np.abs(candidate - beta)))
        beta, value = candidate, new_value
        if moved < tolerance:
            break
    return beta


def to_elo(beta: np.ndarray, anchor: int) -> np.ndarray:
    return ANCHOR_ELO + ELO_PER_LOGIT * (beta - beta[anchor])


def bootstrap_elo(outcomes: list[dict], players: list[str], anchor: int, replicates: int = 1000,
                  seed: int = 0, ridge: float = 1e-3, start: np.ndarray | None = None) -> np.ndarray:
    """Elo under each resample of the prompts, shape [replicates, players].

    Whole prompts are resampled, not single judgments: two pairs judged on the same opening
    share whatever makes that opening easy or hard, so they are not independent draws.
    """
    index = {name: k for k, name in enumerate(players)}
    prompts = sorted({o["prompt_id"] for o in outcomes})
    position = {prompt: k for k, prompt in enumerate(prompts)}
    per_prompt = np.zeros((len(prompts), len(players), len(players)))
    for outcome in outcomes:
        add_outcome(per_prompt[position[outcome["prompt_id"]]], outcome, index)
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, len(players)))
    for replicate in range(replicates):
        counts = np.bincount(rng.integers(0, len(prompts), len(prompts)), minlength=len(prompts))
        wins = np.tensordot(counts, per_prompt, axes=1)
        draws[replicate] = to_elo(fit_strengths(wins, ridge, start=start), anchor)
    return draws


def ranks(values) -> np.ndarray:
    """Ranks from 0, tied values sharing their average rank."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values))
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or values[order[end]] != values[order[start]]:
            result[order[start:end]] = (start + end - 1) / 2
            start = end
    return result


def spearman(x, y) -> float:
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])
