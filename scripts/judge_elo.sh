#!/bin/bash
# LLM-judge Elo for a folder of samples files (Figures 1 and 7): every player's continuations are judged against other
# players' continuations of the same openings, in both orders, and a Bradley-Terry fit turns the verdicts into Elo.
#
#   bash scripts/judge_elo.sh results/pretrain/samples results/pretrain/elo          # here, one judge worker per GPU
#   sbatch --gres=gpu:4 scripts/judge_elo.sh results/pretrain/samples results/pretrain/elo
#   bash scripts/judge_elo.sh results/pretrain/samples results/pretrain/elo-qwen Qwen/Qwen2.5-7B-Instruct
#
# The optional third argument is the judge: any Hugging Face chat model id (or local directory) whose tokenizer encodes
# A, B and C as single tokens; default microsoft/phi-4. Judge calls are cached per judge model, so switching judges
# reuses nothing but never mixes verdicts; give each judge its own output folder.
#
# Input: one <player>.jsonl per player (plm.eval.judge_samples writes them), every file over the same prompt_ids with the
# same prompt text. Output in the second folder: elo.json and elo.md (rank, Elo, 95% interval, meetings, wins, losses,
# ties, and the judge's checks), schedule.jsonl, judgments/ and logs/.
#
# Each player meets PER_PLAYER opponents drawn at random, each on a random prompt, never the same two on one prompt
# twice. Both continuations are cut to a common length in GPT-2 tokens (each capped at CAP, then both to the shorter,
# never below FLOOR); the judge reads the opening and the two and answers A, B or C (about equal), from its letter
# probabilities. Every pair is judged in both orders and counts as a win only when both orders agree. ANCHOR sits at
# Elo 1000; 95% intervals come from BOOTSTRAP resamples of whole prompts. Every judge call is cached in CACHE under a
# hash of the judge model and the exact text it reads, so adding a player judges only that player's pairs.
# The judge model is downloaded to HF_HOME on first use (phi-4 is about 30 GB). ENGINE=mock checks everything without a GPU.
#SBATCH --job-name=judge-elo
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=results/logs/%x-%j.log
#SBATCH --error=results/logs/%x-%j.log
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

USAGE="usage: bash scripts/judge_elo.sh <samples folder> <output folder> [judge model, default microsoft/phi-4]"
SAMPLES=${1:?$USAGE}
OUT=${2:?$USAGE}
JUDGE_MODEL=${3:-microsoft/phi-4}
CACHE=${CACHE:-results/judge-cache}
ENGINE=${ENGINE:-vllm}                      # vllm | hf (transformers) | mock (no model)
PER_PLAYER=${PER_PLAYER:-400}
SEED=${SEED:-0}
ANCHOR=${ANCHOR:-vllm_ar_1.3b_pre_ar_topk}  # the first player alphabetically if absent
BOOTSTRAP=${BOOTSTRAP:-1000}
CAP=${CAP:-128}
FLOOR=${FLOOR:-32}
GPUS=${GPUS:-$( [ "$ENGINE" = mock ] && echo 1 || nvidia-smi -L | wc -l )}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error
PY=(uv run --no-sync python -m)

mkdir -p "$OUT" "$CACHE" || exit 1
"${PY[@]}" plm.judge.schedule --samples "$SAMPLES" --per-player "$PER_PLAYER" --seed "$SEED" --out "$OUT/schedule.jsonl" || exit 1

rm -rf "$OUT/judgments" "$OUT/logs"   # rebuilt from the cache and the schedule every run
mkdir -p "$OUT/judgments" "$OUT/logs"
for worker in $(seq 0 $((GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$worker "${PY[@]}" plm.judge.judge --samples "$SAMPLES" --schedule "$OUT/schedule.jsonl" --cache "$CACHE" \
        --engine "$ENGINE" --judge-model "$JUDGE_MODEL" --cap "$CAP" --floor "$FLOOR" --seed "$SEED" \
        --shard "$worker" --shards "$GPUS" --tag "${SLURM_JOB_ID:-local}-shard${worker}of${GPUS}" \
        --out "$OUT/judgments/shard${worker}of${GPUS}.jsonl" > "$OUT/logs/judge-worker$worker.log" 2>&1 &
done
wait
written=$(ls "$OUT"/judgments/shard*.jsonl 2>/dev/null | wc -l)
if [ "$written" -ne "$GPUS" ]; then
    echo "only $written of $GPUS judge workers finished; see $OUT/logs. Rerun to finish: the cache keeps their work"
    exit 1
fi
"${PY[@]}" plm.judge.fit --judgments "$OUT/judgments/shard*.jsonl" --anchor "$ANCHOR" --bootstrap "$BOOTSTRAP" --seed "$SEED" --out "$OUT/elo.json"
