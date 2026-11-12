# Parallel Language Models: inference

Anonymous code release for *Parallel Language Models via Variational Decoding*. This repository contains everything
needed to reproduce the post-training results (Table 11):
generation with the released checkpoints on vLLM, and grading on GSM8K, MATH-500, HumanEval and IFEval.

A PLM decodes in rounds. Each round runs one forward pass over `k` verify rows, which check the tokens proposed in
the previous round, and `k` proposal blocks of `k` tokens, one block for each possible number of accepted tokens.
`--match exact` accepts a proposed token only if it equals the verify row's own decode, so samples follow the model's
own distribution. `--match relaxed` also accepts tokens the verify row finds likely enough: inside its top-0.94 nucleus for post-trained models (`--setting posttrain`, the default), or with probability above 2e-3 for pretrained models (`--setting pretrain`). Rounds run as
captured CUDA graphs on vLLM's model, paged KV cache and FlexAttention kernels, one request at a time (batch size 1).

## Install

Needs Linux, an NVIDIA GPU with a driver for CUDA 12.8 or newer, and [uv](https://docs.astral.sh/uv/).

```
uv sync --extra eval          # torch 2.11 (cu128), vLLM 0.26.0, grading dependencies; pinned in uv.lock
bash scripts/setup_eval.sh    # Google's IFEval checkers and NLTK data, into third_party/
```

## Models

```
bash scripts/download.sh                            # the post-trained Table 11 models and the baselines, into checkpoints/posttrain/
bash scripts/download.sh posttrain/plm-1.3B-beta1   # or some of them
bash scripts/download.sh --pretrain                 # also the pretrained models
```

The checkpoints can also be browsed at https://anonymous-hf.com/a/gk540ibws5qh/.

Post-trained models (`posttrain/`), the ones this repository runs:

| directory | parameters | β | post-training |
|---|------------|---|---|
| `posttrain/ar-1.3B` | 1.3B       | – | autoregressive baseline, SFT + DPO |
| `posttrain/plm-1.3B-beta0.5` | 1.3B       | 0.5 | PLM, SFT + DPO |
| `posttrain/plm-1.3B-beta1` | 1.3B       | 1 | PLM, SFT + DPO |
| `posttrain/plm-1.3B-beta2` | 1.3B       | 2 | PLM, SFT + DPO |
| `posttrain/plm-3B-beta1` | 3B         | 1 | PLM, SFT + DPO |
| `posttrain/ptp-lora-1.3B` | +220M      | – | gated-LoRA parallel decoder distilled from `ar-1.3B` |
| `posttrain/eagle3-ar-1.3B` | +170M      | – | EAGLE-3 draft head for `ar-1.3B` |
| `posttrain/fdv2-1.3B` | 1.3B       | – | Fast-dLLM v2 block diffusion, converted from `ar-1.3B` |

Pretrained models (`pretrain/`). Training tokens are optimizer steps × global batch × 4096-token sequences.

| directory | parameters | β | training tokens | used in |
|---|---|---|---|---|
| `pretrain/ar-80M` | 80M | – | 3.4B | Figure 5 |
| `pretrain/plm-80M-beta0.5` | 80M | 0.5 | 3.4B | Figure 5 |
| `pretrain/plm-80M-beta1` | 80M | 1 | 3.4B | Figure 5 |
| `pretrain/plm-80M-beta2` | 80M | 2 | 3.4B | Figure 5 |
| `pretrain/ar-150M` | 150M | – | 6.9B | Figure 5 |
| `pretrain/plm-150M-beta0.5` | 150M | 0.5 | 6.9B | Figure 5 |
| `pretrain/plm-150M-beta1` | 150M | 1 | 6.9B | Figure 5 |
| `pretrain/plm-150M-beta2` | 150M | 2 | 6.9B | Figure 5 |
| `pretrain/ar-350M-23.6B` | 350M | – | 23.6B | Figure 5 |
| `pretrain/plm-350M-beta0.5-23.6B` | 350M | 0.5 | 23.6B | Figure 5 |
| `pretrain/plm-350M-beta1-23.6B` | 350M | 1 | 23.6B | Figure 5 |
| `pretrain/plm-350M-beta2-23.6B` | 350M | 2 | 23.6B | Figure 5 |
| `pretrain/ar-350M` | 350M | – | 25.0B | Table 7 |
| `pretrain/plm-350M-beta1` | 350M | 1 | 25.0B | Figure 1, Table 7 |
| `pretrain/ar-1.3B` | 1.3B | – | 100.0B | Figure 5, Table 7; base of the post-trained AR |
| `pretrain/plm-1.3B-beta0.5` | 1.3B | 0.5 | 100.0B | Figure 5; base of the post-trained β = 0.5 |
| `pretrain/plm-1.3B-beta1` | 1.3B | 1 | 100.0B | Figures 1 and 5, Table 7; base of the post-trained β = 1 |
| `pretrain/plm-1.3B-beta2` | 1.3B | 2 | 100.0B | Figure 5; base of the post-trained β = 2 |
| `pretrain/plm-3B-beta1` | 3B | 1 | 300.0B | Figures 1 and 5, Table 7; base of the post-trained 3B |

All models use the GPT-2 tokenizer. The post-trained models use the prompt template `User: {prompt}\n\nAssistant:`.

Pretrained models decode with `--setting pretrain`, which uses round width `k = 6` for both decodings and makes
`--match relaxed` accept a proposal when the model gives it probability above `2e-3`. Post-trained models use the
default, `--setting posttrain`. `--mode fast` uses GEMM tiles tuned to the target GPU (RTX 6000 Ada)
for both settings' round widths at every released hidden size.

```
uv run --no-sync python -m bench.generate --setting pretrain --model checkpoints/pretrain/plm-1.3B-beta1 --match relaxed \
    --dataset prompts.json --tokens 128 --seeds 0 1 2 --out results.json
uv run --no-sync python -m bench.generate --ar --ar-sampler voronoi --model checkpoints/pretrain/ar-1.3B \
    --ar-scheme checkpoints/pretrain/plm-1.3B-beta1 --dataset prompts.json --tokens 128 --out results.json
```

`--ar-sampler topk` is ordinary autoregressive sampling (temperature 1, top 128 tokens), while `--ar-sampler voronoi`
draws each token with the PLM's own sampling rule and seeded random values, so on a PLM it is theoretically
token-identical to `--match exact` for the same seed. A plain AR model has no such sampling table, so `--ar-scheme`
borrows it from the β = 1 PLM of the same size (all PLMs of one size share it).

The pretrained models' runs are in [Reproducing Figure 1](#reproducing-figure-1), the post-trained models' in
[Reproducing Table 11](#reproducing-table-11).

### PTP baseline

`posttrain/ptp-lora-1.3B` is a parallel (PTP) decoder distilled from `ar-1.3B`: the teacher's weights, unchanged, plus a gated
LoRA adapter (rank 256, `gated_lora.safetensors`) and a small auxiliary embedding. In each round the verify rows run the
teacher's own weights and only the proposal rows add the adapter, so exact decoding samples exactly the teacher's
distribution. Its random values blend the auxiliary embedding and decode by inverse CDF over the teacher's 128 likeliest
tokens at temperature 1, which is ordinary top-128 sampling. Its AR row, `ar-icdf` on `ar-1.3B`, applies the same
inverse-CDF rule to the same seeded random values one token at a time, so it is theoretically token-identical to the
distilled model's exact decoding for the same seed. It decodes with the post-training defaults (exact `k = 7`; relaxed
`k = 10`, top-0.94 nucleus):

```
uv run python -m bench.generate --model checkpoints/posttrain/ptp-lora-1.3B --mode fast --match exact --k 7 \
    --dataset prompts.json --tokens 512 --stop-at-eos --seeds 0 1 2 --out results.json
```

### EAGLE-3 baseline

`posttrain/eagle3-ar-1.3B` is an EAGLE-3 draft head for `ar-1.3B`, trained with the same data `ptp-lora-1.3B` is distilled from.
Both of its decodings are theoretically exact for the target's distribution (temperature 1 over the 128
likeliest tokens; just like PTP), so only speed changes.

- **Chain** (default): The head drafts `--eagle3-k` tokens per target call (default 4, the fastest), verified by rejection sampling. Official vLLM implementation.
- **Tree** (`--eagle3-tree`): EAGLE-2's dynamic draft tree, using their defaults of 60 nodes over 7 levels with 10 children a node,
  verified in one target call by multi-candidate speculative sampling. Experimental port of their code to vLLM.

```
uv run --no-sync python -m bench.generate --model checkpoints/posttrain/ar-1.3B --eagle3 checkpoints/posttrain/eagle3-ar-1.3B \
    --dataset prompts.json --tokens 512 --stop-at-eos --seeds 0 1 2 --out results.json          # add --eagle3-tree for the draft tree
```

### Fast-dLLM v2 baseline

`posttrain/fdv2-1.3B` is `ar-1.3B` converted to block diffusion with Fast-dLLM v2 (FDv2) and fine-tuned, so it is a different
model from `ar-1.3B` and no decoding of it reproduces `ar-1.3B`'s distribution. It is a plain Llama export. The baseline lives
in its own package, `fastdllmv2/`, which shares no code path with the PLM decoders: `bench.generate --fdv2` serves
it through `fastdllmv2/sampler.py`, the reference FDv2 sampler: 32-token canvases at 32-aligned positions with an exact block-causal
cache, sub-blocks of 8 in order, and in each pass every masked position of the active sub-block is sampled (temperature 1 over
the 128 likeliest tokens, as for the other decodings, or greedily with `--fdv2-temperature 0`) and those above
`--fdv2-threshold` are kept, plus the most confident one. A lower threshold unmasks more tokens per pass; at 1 it unmasks one
token a pass.
`tokens/call` counts the prefill as one call and every denoising and commit pass as one.

```
uv run --no-sync python -m bench.generate --model checkpoints/posttrain/fdv2-1.3B --fdv2 --fdv2-threshold 0.9 \
    --dataset prompts.json --tokens 512 --stop-at-eos --seeds 0 1 2 --out results.json
```

## Generating

```
uv run python -m bench.generate --model checkpoints/posttrain/plm-1.3B-beta1 --mode fast --match exact --k 7 \
    --dataset prompts.json --tokens 512 --stop-at-eos --seeds 0 1 2 --out results.json
```

`--dataset` is a JSON list of prompt strings. One process runs the dataset `1 + len(--seeds)` times: a warm-up pass,
which pays graph capture and compilation and is discarded, then one pass per seed. The results file records, for each
seed, every continuation with its timing and the number of tokens accepted per round.

- `--mode fast` sends the round's GEMMs through Triton tiles tuned on an **RTX 6000 Ada**, for the paper's round
  widths (`k = 7` and `k = 10` post-trained, `k = 6` pretrained); a width without tiles is an error. `--mode default`
  is stock vLLM with no patching and runs on any GPU vLLM supports. Both modes run the same algorithm, but the Triton GEMMs round differently from cuBLAS, so long generations
  can occasionally differ between the two. The paper's numbers use `fast`; on other GPUs, use `default`.
- `--match`, `--k` and `--ar` choose the decoding.

## Reproducing Figure 1

Figure 1 places every model by its quality, the Elo of a pairwise LLM-judge tournament on FineWeb-Edu continuations,
against its single-stream speed. This repository reproduces it for the pretrained 1.3B and 3B models, every decoding
of them; the paper's baselines (off-the-shelf autoregressive and diffusion language
models) and the 350M PLM are not included. Elo is relative to the field it is fitted on, so the values shift from the
paper's: here `vllm_ar_1.3b_pre_ar_topk` (`ar-1.3B`, top-k sampling) sits at 1000.

`runs/pretrain/` holds one frozen job per pretrained 1.3B and 3B model and decoding (ar-topk, ar-voronoi, exact and
relaxed, and `ar-1.3B`'s two AR rows), 18 in total, on the judge's 200 prompts (`data/fineweb-edu/`) under its contract:
128 tokens with no stop at EOS, temperature 1 over the 128 likeliest tokens. Each job writes
`results/pretrain/<model>-<decoding>/fineweb-edu.json` over a warm-up pass and seeds 0, 1 and 2, and turns seed 0 into
the judge's samples file.

```
bash scripts/download.sh --pretrain                                # the pretrained models
uv sync --extra eval
bash runs/submit_all.sh pretrain -p <partition>                   # 18 jobs, 3 to 15 minutes each on an RTX 6000 Ada
uv run --no-sync python -m bench.eval.table results/pretrain         # tokens/s and tokens/call, mean ± 95% CI over the three seeds
sbatch --gres=gpu:4 scripts/judge_elo.sh results/pretrain/samples results/pretrain/elo     # or bash ..., without Slurm
uv run --no-sync python -m bench.judge.figure1 --samples results/pretrain/samples \
    --elo results/pretrain/elo/elo.json --out results/pretrain/figure1.png
```

Each player is one samples file, `results/pretrain/samples/<player>.jsonl`, with seed 0's continuations of the 200
prompts. The judge is the optional third argument of `scripts/judge_elo.sh`, a Hugging Face model id (default
`microsoft/phi-4`, about 30 GB, downloaded on first use; any chat model whose tokenizer has A, B and C as single tokens).
It meets each player with 400 random opponents, each on a
random prompt, cuts both continuations to a common length (at most 128 GPT-2 tokens), and reads its probabilities for
A, B or C (about equal) after showing the opening and both continuations; every pair is judged in both orders and is a
win only when both agree. A Bradley-Terry fit turns the outcomes into Elo, with 95% intervals from 1000 resamples of
whole prompts, and `elo.md` also reports the judge's swap consistency, first-position bias, tie rate and split-half
rank agreement. Every judge call is cached (`results/judge-cache/`), so adding a player judges only its own pairs. The
other settings (meetings, anchor, length cap, engine) are environment variables at the top of `scripts/judge_elo.sh`; `ENGINE=mock` runs the whole pipeline without a GPU. `bench.judge.figure1` prints the joined
table (Elo and end-to-end tokens per second) and draws the figure.

## Reproducing Table 11

Every run of Table 11 has its own frozen job in `runs/posttrain/`: one per model, decoding and benchmark, with all flags pinned.
There are 72 in total: ar-topk, ar-voronoi, exact and relaxed for each PLM, and ar-topk and ar-voronoi for `ar-1.3B`,
each on four benchmarks, plus 12 for the distilled model (`ptp-lora-1.3B` exact and relaxed, and `ar-1.3B` ar-icdf), 8 for
`ar-1.3B` with its EAGLE-3 head (`eagle3`, the chain, and `eagle3-tree`) and 24 for the FDv2 baseline: `fdv2-1.3B` at thresholds 1, 0.95, 0.7 and 0.3 (`t1` … `t0.3`, temperature 1 over the 128 likeliest tokens) and
greedy at threshold 1 (`greedy-t1`), with `ar-1.3B`'s greedy decode (`ar-greedy`) to compare it with. Each job generates over a warm-up pass and seeds 0, 1 and 2, grades the run, and writes
`results/posttrain/<model>-<decoding>/<benchmark>.json` and `.score.json`.

```
bash runs/submit_all.sh posttrain -p <partition>                  # builds the prompts, then sbatch runs/posttrain/*.sbatch with your arguments
bash runs/posttrain/plm-1.3B-beta1-exact-gsm8k.sbatch             # or run one job directly, without Slurm
uv run --no-sync python -m bench.eval.table results/posttrain       # Table 11, mean ± 95% CI over the three seeds
```

The job headers are site-neutral (one GPU, 6 hours, logs in `results/logs/`). `CHECKPOINTS` overrides the model
directory (default `checkpoints/`; the jobs read `$CHECKPOINTS/posttrain/<name>`). `scripts/make_runs.py` regenerates the jobs
of both settings.

The table prints every metric as the mean over seeds ± a two-sided 95% Student-t interval (n = 3). The `.score.json`
files also record the standard deviation, the per-seed values and the per-question grades. To grade a single run by
hand:

```
uv run --no-sync python -m bench.eval.build_prompts --benchmark gsm8k --tokenizer checkpoints/posttrain/plm-1.3B-beta1 --out-dir prompts/gsm8k
uv run --no-sync python -m bench.generate --model checkpoints/posttrain/plm-1.3B-beta1 --mode fast --match relaxed --k 10 \
    --dataset prompts/gsm8k/prompts.json --tokens 512 --max-model-len 4096 --stop-at-eos --seeds 0 1 2 --out gsm8k.json
uv run --no-sync python -m bench.eval.score --benchmark gsm8k --results gsm8k.json --tasks prompts/gsm8k/tasks.json
```

Round width: `--k 7` for exact decoding, and for relaxed decoding `--k 10` on the 1.3B models and `--k 7` on the 3B.
All decodings sample at temperature 1.

Speed is single-stream wall-clock time, prefill included, and depends on the GPU.
The exact tokens also depend on the GPU architecture, since kernel selection and autotuning differ between GPUs.

## Layout

| path | contents |
|---|---|
| `bench/generate.py`, `bench/harness.py`, `bench/engine.py` | the CLI: generation over a dataset, with timing; the measurement harness and engine settings the baselines' CLIs share |
| `bench/ar.py` | autoregressive decoding on stock vLLM (`--ar`) |
| `plm/rounds.py` | `--setting` / `--mode` / `--match` / `--k`, and how the round decoder is built |
| `bench/eval/` | prompt building, grading and the results table |
| `runs/posttrain/`, `runs/pretrain/`, `scripts/make_runs.py` | one frozen job per Table 11 and per Figure 1 run, and their generator |
| `data/fineweb-edu/`, `plm/eval/judge_samples.py` | the judge's 200 FineWeb-Edu prompts, and the converter to its samples format |
| `bench/judge/`, `scripts/judge_elo.sh` | the LLM-judge tournament: schedule, judge, Elo fit, and the Figure 1 table and plot |
| `plm/decoder.py` | the round decoder: captured prefill, round-0 and steady-round graphs |
| `plm/mask.py` | the round's attention mask and layout |
| `plm/auxiliary.py`, `plm/iterate.py` | the Voronoi rule's random values, partition table and assignment engine (Triton) |
| `plm/gated_lora.py` | the distilled model's gated LoRA adapter, applied to proposal rows only |
| `plm/fast_gemm.py`, `plm/gemm_tables.py`, `plm/gpu_tuning/` | `--mode fast` |
| `plm/voronoi_ar.py` | a model's own decode rule (Voronoi or inverse CDF) as a vLLM logits processor, for the AR rows |
| `eagle3/` | the EAGLE-3 baseline (`--eagle3`): `generate.py` (its flags and decodings), `chain.py` (vLLM speculative decoding, `draft_topk.py` its `--eagle3-draft topk` patch), `tree.py` and `tree_head.py` (the draft tree) |
| `fastdllmv2/` | the Fast-dLLM v2 baseline (`--fdv2`): `generate.py` (its flags and decoding), `sampler.py` (block-diffusion sampler and model state) |
| `plm/model.py`, `plm/config.py` | the vLLM model classes, registered through a vLLM plugin entry point |
