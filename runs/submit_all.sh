#!/bin/bash
# Submit every run of one setting; extra arguments go to each sbatch call, e.g. bash runs/submit_all.sh posttrain -p <partition> -A <account>.
#   posttrain: Table 11 and the distilled model (84 jobs).  pretrain: the pretrained 1.3B and 3B models on the Figure 1 prompts (20 jobs).
# Run from the repository root. Benchmark prompts are built first so that concurrent jobs do not race to write them.
set -euo pipefail

SETTING=${1:?usage: bash runs/submit_all.sh {posttrain|pretrain} [sbatch arguments]}
shift
CHECKPOINTS=${CHECKPOINTS:-checkpoints}
mkdir -p results/logs
if [ "$SETTING" = posttrain ]; then
    for benchmark in gsm8k math500 humaneval ifeval; do
        [ -f "prompts/$benchmark/tasks.json" ] || uv run --no-sync python -m plm.eval.build_prompts --benchmark "$benchmark" --tokenizer "$CHECKPOINTS/posttrain/plm-1.3B-beta1" --out-dir "prompts/$benchmark"
    done
fi
for run in "runs/$SETTING"/*.sbatch; do
    sbatch "$@" "$run"
done
