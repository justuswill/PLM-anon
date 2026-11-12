"""Triton GEMM for the decode-width shapes of the round, swapped into vLLM's unquantized linear path (``--mode fast``).

C[M,N] = A[M,K] @ W[N,K]^T with fp32 accumulation. Split-K writes fp32 partials in a fixed order and a second kernel
sums them, so results are deterministic run to run. ``install`` monkeypatches
``vllm.model_executor.layers.utils.default_unquantized_gemm`` process-globally. Tile configs are keyed on (M, N, K) and
were tuned on an RTX 6000 Ada; shapes without a config fall back to ``F.linear``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(a_ptr, w_ptr, c_ptr, p_ptr, M, N, K,
                stride_am, stride_ak, stride_wn, stride_wk, stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
                INPUT_PRECISION: tl.constexpr = "tf32"):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_m = tl.cdiv(M, BLOCK_M)
    # M-tile fastest so CTAs sharing a weight block run together and the block comes from HBM once
    pid_m = pid % num_m
    pid_n = pid // num_m
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    k_per_split = K // SPLIT_K
    k0 = pid_k * k_per_split
    a_ptrs = a_ptr + rm[:, None] * stride_am + (k0 + rk)[None, :] * stride_ak
    w_ptrs = w_ptr + rn[None, :] * stride_wn + (k0 + rk)[:, None] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_ok = rm[:, None] < M
    n_ok = rn[None, :] < N
    for _ in range(0, tl.cdiv(k_per_split, BLOCK_K)):
        a = tl.load(a_ptrs, mask=m_ok, other=0.0)
        w = tl.load(w_ptrs, mask=n_ok, other=0.0)
        acc = tl.dot(a, w, acc, input_precision=INPUT_PRECISION)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    mask = m_ok & n_ok
    if SPLIT_K == 1:
        tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc.to(c_ptr.dtype.element_ty), mask=mask)
    else:
        tl.store(p_ptr + pid_k * M * N + rm[:, None] * N + rn[None, :], acc, mask=mask)


@triton.jit
def splitk_reduce_kernel(p_ptr, c_ptr, M, N, stride_cm, stride_cn, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(p_ptr + s * M * N + offs, mask=mask, other=0.0)
    rm = offs // N
    rn = offs % N
    tl.store(c_ptr + rm * stride_cm + rn * stride_cn, acc.to(c_ptr.dtype.element_ty), mask=mask)


Config = dict[str, int]  # BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages

HALF_DTYPES = (torch.bfloat16, torch.float16)

CONFIGS: dict[tuple[int, int, int], Config] = {}

_partials: dict[tuple[int, int, int], torch.Tensor] = {}


def _partial_buffer(M: int, N: int, split_k: int, device: torch.device) -> torch.Tensor:
    key = (M, N, device.index)
    buf = _partials.get(key)
    if buf is None or buf.shape[0] < split_k:
        buf = torch.empty(split_k, M, N, device=device, dtype=torch.float32)
        _partials[key] = buf
    return buf


def triton_linear(a: torch.Tensor, w: torch.Tensor, cfg: Config, out: torch.Tensor | None = None) -> torch.Tensor:
    M, K = a.shape
    N = w.shape[0]
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=a.dtype)
    split_k = cfg["SPLIT_K"]
    p = _partial_buffer(M, N, split_k, a.device) if split_k > 1 else out
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]), split_k)
    gemm_kernel[grid](a, w, out, p, M, N, K, a.stride(0), a.stride(1), w.stride(0), w.stride(1), out.stride(0), out.stride(1),
                      BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"], SPLIT_K=split_k,
                      INPUT_PRECISION="ieee" if a.dtype == torch.float32 else "tf32",
                      num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    if split_k > 1:
        block = 1024
        splitk_reduce_kernel[(triton.cdiv(M * N, block),)](p, out, M, N, out.stride(0), out.stride(1), SPLIT_K=split_k, BLOCK=block, num_warps=4)
    return out


def install(configs: dict[tuple[int, int, int], Config]) -> None:
    """Route vLLM's unquantized GEMMs with a tuned (M, N, K) through the Triton kernel."""
    import vllm.model_executor.layers.utils as vllm_utils

    CONFIGS.update(configs)

    def fast_gemm(layer, x, weight, bias=None):
        if bias is None and x.dim() == 2 and x.dtype in HALF_DTYPES:
            cfg = CONFIGS.get((x.shape[0], weight.shape[0], weight.shape[1]))
            if cfg is not None:
                return triton_linear(x, weight, cfg)
        return F.linear(x, weight, bias)

    vllm_utils.default_unquantized_gemm = fast_gemm
