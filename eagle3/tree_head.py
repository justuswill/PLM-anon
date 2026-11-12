"""The EAGLE-3 draft head as fixed-shape PyTorch, so each call can be captured in a CUDA graph.

Same math as vLLM's Eagle3LlamaForCausalLM: a draft entry j pairs the target's aux hidden state of position j (the layers
in the head's ``eagle_config``, through ``fc``) or a draft output with the embedding of token j+1,
runs one decoder layer whose q/k/v read ``[input_layernorm(emb), hidden_norm(hidden)]``, and predicts token j+2 with
``lm_head(norm(.))``. RoPE position = entry index; a tree level i below the root entry sits at the next positions, as in
training-time test.

One K/V buffer holds the committed entries (``stable``, indexed by position) followed by a scratch region for the tree levels
being expanded. Every call attends over the whole buffer under a boolean mask, so shapes never change with the sequence.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class DraftHead:
    @staticmethod
    def aux_layers(head_dir: str) -> tuple[int, ...]:
        config = json.loads((Path(head_dir) / "config.json").read_text())
        return tuple(config["eagle_config"]["eagle_aux_hidden_state_layer_ids"])

    def __init__(self, head_dir: str, embed_weight: torch.Tensor, max_entries: int, scratch: int, device: torch.device) -> None:
        config = json.loads((Path(head_dir) / "config.json").read_text())
        weights = {name: tensor.to(device=device, dtype=torch.bfloat16) for name, tensor in load_file(str(Path(head_dir) / "model.safetensors")).items()}
        # A 50257-wide lm_head runs an unaligned (align1) GEMM several times slower; pad to a multiple of 64 and mask the padding.
        head = weights["lm_head.weight"]
        padded = (head.shape[0] + 63) // 64 * 64
        weights["lm_head.weight"] = torch.cat([head, head.new_zeros(padded - head.shape[0], head.shape[1])])
        self.w = weights
        self.eps = config["rms_norm_eps"]
        self.hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = config.get("head_dim", self.hidden // self.heads)
        self.groups = self.heads // self.kv_heads
        self.vocab = config["draft_vocab_size"]
        theta = (config.get("rope_parameters") or {}).get("rope_theta", config.get("rope_theta", 10000.0))
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
        self.embed_weight = embed_weight
        self.max_entries, self.scratch = max_entries, scratch
        length = max_entries + scratch
        self.k = torch.zeros(length, self.kv_heads, self.head_dim, dtype=torch.bfloat16, device=device)
        self.v = torch.zeros_like(self.k)
        self.key_index = torch.arange(length, device=device)
        self.device = device

    def rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        freqs = positions[:, None].float() * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos()[:, None].to(x.dtype), emb.sin()[:, None].to(x.dtype)
        return x * cos + rotate_half(x) * sin

    def fc(self, aux: torch.Tensor) -> torch.Tensor:
        return F.linear(aux, self.w["fc.weight"])

    def layer(self, tokens: torch.Tensor, hidden: torch.Tensor, positions: torch.Tensor, write_slots: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        """One midlayer pass over ``n`` rows; writes their K/V at ``write_slots`` first, then attends under ``visible`` [n, L]."""
        w, n = self.w, tokens.shape[0]
        emb = F.embedding(tokens, self.embed_weight).to(hidden.dtype)
        x = torch.cat((rms_norm(emb, w["midlayer.input_layernorm.weight"], self.eps), rms_norm(hidden, w["midlayer.hidden_norm.weight"], self.eps)), dim=-1)
        q = F.linear(x, w["midlayer.self_attn.q_proj.weight"]).view(n, self.heads, self.head_dim)
        k = F.linear(x, w["midlayer.self_attn.k_proj.weight"]).view(n, self.kv_heads, self.head_dim)
        v = F.linear(x, w["midlayer.self_attn.v_proj.weight"]).view(n, self.kv_heads, self.head_dim)
        q, k = self.rope(q, positions), self.rope(k, positions)
        self.k.index_copy_(0, write_slots, k)
        self.v.index_copy_(0, write_slots, v)
        # GQA without copying K/V: fold each KV head's query group into the query length (q head h reads kv head h // groups).
        # Plain batched matmuls: with a few dozen queries, fused attention kernels launch a handful of blocks over thousands of keys.
        qg = q.view(n, self.kv_heads, self.groups, self.head_dim).permute(1, 2, 0, 3).reshape(self.kv_heads, self.groups * n, self.head_dim)
        mask = visible[None].expand(self.groups, -1, -1).reshape(1, self.groups * n, -1)
        scores = torch.bmm(qg, self.k.permute(1, 2, 0)).float() * self.head_dim ** -0.5
        weights_ = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1).to(qg.dtype)
        out = torch.bmm(weights_, self.v.transpose(0, 1))
        out = out.view(self.kv_heads, self.groups, n, self.head_dim).permute(2, 0, 1, 3).reshape(n, self.hidden)
        h = hidden + F.linear(out, w["midlayer.self_attn.o_proj.weight"])
        y = rms_norm(h, w["midlayer.post_attention_layernorm.weight"], self.eps)
        return h + F.linear(F.silu(F.linear(y, w["midlayer.mlp.gate_proj.weight"])) * F.linear(y, w["midlayer.mlp.up_proj.weight"]), w["midlayer.mlp.down_proj.weight"])

    def logprobs(self, h: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(F.linear(rms_norm(h, self.w["norm.weight"], self.eps), self.w["lm_head.weight"])[:, : self.vocab].float(), dim=-1)
