#!/bin/bash
# Submit every Table 11 run; extra arguments go to each sbatch call, e.g. bash runs/submit_all.sh -p <partition> -A <account>.
# Run from the repository root. Prompts are built first so that concurrent jobs do not race to write them.
set -euo pipefail

CHECKPOINTS=${CHECKPOINTS:-checkpoints}
mkdir -p results/logs
for benchmark in gsm8k math500 humaneval ifeval; do
    [ -f "prompts/$benchmark/tasks.json" ] || uv run --no-sync python -m plm.eval.build_prompts --benchmark "$benchmark" --tokenizer "$CHECKPOINTS/posttrain/plm-1.3B-beta1" --out-dir "prompts/$benchmark"
done
for run in runs/*.sbatch; do
    sbatch "$@" "$run"
done
