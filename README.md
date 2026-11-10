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
bash scripts/download.sh                            # the five post-trained Table 11 models and the distilled model, into checkpoints/posttrain/
bash scripts/download.sh posttrain/plm-1.3B-beta1   # or some of them
bash scripts/download.sh --pretrain                 # also the pretrained models
```

The checkpoints can also be browsed at https://anonymous-hf.com/a/gk540ibws5qh/.

Post-trained models (`posttrain/`), the ones this repository runs:

| directory | parameters | β | post-training | Table 11 row |
|---|---|---|---|---|
| `posttrain/ar-1.3B` | 1.3B | – | autoregressive baseline, SFT + DPO | 1.3B AR |
| `posttrain/plm-1.3B-beta0.5` | 1.3B | 0.5 | PLM, SFT + DPO | 1.3B β = 0.5 |
| `posttrain/plm-1.3B-beta1` | 1.3B | 1 | PLM, SFT + DPO | 1.3B β = 1 |
| `posttrain/plm-1.3B-beta2` | 1.3B | 2 | PLM, SFT + DPO | 1.3B β = 2 |
| `posttrain/plm-3B-beta1` | 3B | 1 | PLM, SFT + DPO | 3B β = 1 |
| `posttrain/ptp-lora-1.3B` | 1.3B | – | gated-LoRA parallel decoder distilled from `ar-1.3B` |  |

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
uv run --no-sync python -m plm.generate --setting pretrain --model checkpoints/pretrain/plm-1.3B-beta1 --match relaxed \
    --dataset prompts.json --tokens 128 --seeds 0 1 2 --out results.json
uv run --no-sync python -m plm.generate --ar --ar-sampler voronoi --model checkpoints/pretrain/ar-1.3B \
    --ar-voronoi-scheme checkpoints/pretrain/plm-1.3B-beta1 --dataset prompts.json --tokens 128 --out results.json
```

`--ar-sampler topk` is ordinary autoregressive sampling (temperature 1, top 128 tokens), while `--ar-sampler voronoi` draws each token with the PLM's own sampling rule and seeded random values, so on a PLM it is theoretically token-identical to `--match exact` for the same seed. A plain AR model has no such sampling table, so `--ar-voronoi-scheme` borrows it from the β = 1 PLM of the same size (all PLMs of one size share it).

The quality numbers of Figure 1 and Table 7 come from a pairwise LLM-judge tournament on FineWeb-Edu continuations,
which this repository does not include. It reproduces the generations and their speed.

### The distilled model

`posttrain/ptp-lora-1.3B` is a parallel (PTP) decoder distilled from `ar-1.3B`: the teacher's weights, unchanged, plus a gated
LoRA adapter (rank 256, `gated_lora.safetensors`) and a small auxiliary embedding. In each round the verify rows run the
teacher's own weights and only the proposal rows add the adapter, so exact decoding samples exactly the teacher's
distribution. Its random values blend the auxiliary embedding and decode by inverse CDF over the teacher's 128 likeliest
tokens at temperature 1, which is ordinary top-128 sampling. Its AR row, `ar-1.3B` with
`--ar-sampler voronoi --ar-voronoi-scheme checkpoints/posttrain/ptp-lora-1.3B` (`ar-lora` in `runs/`), applies the same
inverse-CDF rule to the same seeded random values one token at a time, so it is theoretically token-identical to the
distilled model's exact decoding for the same seed. It decodes with the post-training defaults (exact `k = 7`; relaxed
`k = 10`, top-0.94 nucleus):

```
uv run python -m plm.generate --model checkpoints/posttrain/ptp-lora-1.3B --mode fast --match exact --k 7 \
    --dataset prompts.json --tokens 512 --stop-at-eos --seeds 0 1 2 --out results.json
```

## Generating

```
uv run python -m plm.generate --model checkpoints/posttrain/plm-1.3B-beta1 --mode fast --match exact --k 7 \
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

## Reproducing Table 11

Every run of Table 11 has its own frozen job in `runs/`: one per model, decoding and benchmark, with all flags pinned.
There are 72 in total: ar-topk, ar-voronoi, exact and relaxed for each PLM, and ar-topk and ar-voronoi for `ar-1.3B`,
each on four benchmarks, plus 12 for the distilled model (`ptp-lora-1.3B` exact and relaxed, and `ar-1.3B` ar-lora). Each job generates over a warm-up pass and seeds 0, 1 and 2, grades the run, and writes
`results/<model>-<decoding>/<benchmark>.json` and `.score.json`.

```
bash runs/submit_all.sh -p <partition>                  # builds the prompts, then sbatch runs/*.sbatch with your arguments
bash runs/plm-1.3B-beta1-exact-gsm8k.sbatch             # or run one job directly, without Slurm
uv run --no-sync python -m plm.eval.table results       # Table 11, mean ± 95% CI over the three seeds
```

The job headers are site-neutral (one GPU, 6 hours, logs in `results/logs/`). `CHECKPOINTS` overrides the model
directory (default `checkpoints/`; the jobs read `$CHECKPOINTS/posttrain/<name>`). `scripts/make_runs.py` regenerates the jobs.

The table prints every metric as the mean over seeds ± a two-sided 95% Student-t interval (n = 3). The `.score.json`
files also record the standard deviation, the per-seed values and the per-question grades. To grade a single run by
hand:

```
uv run --no-sync python -m plm.eval.build_prompts --benchmark gsm8k --tokenizer checkpoints/posttrain/plm-1.3B-beta1 --out-dir prompts/gsm8k
uv run --no-sync python -m plm.generate --model checkpoints/posttrain/plm-1.3B-beta1 --mode fast --match relaxed --k 10 \
    --dataset prompts/gsm8k/prompts.json --tokens 512 --max-model-len 4096 --stop-at-eos --seeds 0 1 2 --out gsm8k.json
uv run --no-sync python -m plm.eval.score --benchmark gsm8k --results gsm8k.json --tasks prompts/gsm8k/tasks.json
```

Round width: `--k 7` for exact decoding, and for relaxed decoding `--k 10` on the 1.3B models and `--k 7` on the 3B.
All decodings sample at temperature 1.

Speed is single-stream wall-clock time, prefill included, and depends on the GPU.
The exact tokens also depend on the GPU architecture, since kernel selection and autotuning differ between GPUs.

## Layout

| path | contents |
|---|---|
| `plm/generate.py` | the CLI: generation over a dataset, with timing |
| `plm/config.py` | `--mode` / `--match` / `--k`, and how the engines are built |
| `plm/eval/` | prompt building, grading and the results table |
| `runs/`, `scripts/make_runs.py` | one frozen job per Table 11 run, and their generator |
| `vllm_plm/decoder.py` | the round decoder: captured prefill, round-0 and steady-round graphs |
| `vllm_plm/mask.py` | the round's attention mask and layout |
| `vllm_plm/auxiliary.py`, `vllm_plm/iterate.py` | the Voronoi rule's random values, partition table and assignment engine (Triton) |
| `vllm_plm/gated_lora.py` | the distilled model's gated LoRA adapter, applied to proposal rows only |
| `vllm_plm/fast_gemm.py`, `vllm_plm/gemm_tables.py`, `vllm_plm/gpu_tuning/` | `--mode fast` |
| `vllm_plm/voronoi_ar.py` | the PLM decode rule as a vLLM logits processor, for the AR baseline |
| `vllm_plm/model.py`, `vllm_plm/config.py` | the vLLM model classes, registered through a vLLM plugin entry point |
