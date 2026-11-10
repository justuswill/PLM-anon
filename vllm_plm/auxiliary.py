"""The auxiliary schemes of the released checkpoints: the logit-Voronoi scheme with the ``iterate`` backend (every PLM), and
the linear-interpolation scheme of the gated-LoRA distilled model.

Parameter and buffer names match the exported checkpoints, so ``auxiliary_scheme.*`` tensors load straight in.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .iterate import Voronoi

MAX_ITERATE_TOP_M = 128


class LogitVoronoiAuxiliary(nn.Module):
    def __init__(self, auxiliary_vocab_size: int, hidden_size: int, text_vocab_size: int, *, top_m: int = 128, backend: str = "iterate",
                 geometry: str = "additive", iterate_shortlist: int = 8, iterate_rounds: int = 2, iterate_sweeps: int = 12,
                 dtype: torch.dtype | None = None, **unused_fields) -> None:
        super().__init__()
        if backend != "iterate" or geometry != "additive":
            raise ValueError(f"only the additive iterate backend is supported, got {backend!r} / {geometry!r}")
        if top_m > MAX_ITERATE_TOP_M or top_m & (top_m - 1) or iterate_shortlist & (iterate_shortlist - 1):
            raise ValueError(f"iterate needs top_m a power of two <= {MAX_ITERATE_TOP_M} and a power-of-two shortlist")
        self.auxiliary_vocab_size = int(auxiliary_vocab_size)
        self.hidden_size = int(hidden_size)
        self.text_vocab_size = int(text_vocab_size)
        self.top_m = int(top_m)
        self.iterate_shortlist = int(iterate_shortlist)
        self.iterate_rounds = int(iterate_rounds)
        self.iterate_sweeps = int(iterate_sweeps)
        self.register_buffer("auxiliary_vocabulary", torch.zeros(self.auxiliary_vocab_size, self.hidden_size, dtype=dtype), persistent=True)
        self.mask_auxiliary = nn.Parameter(torch.zeros(self.hidden_size, dtype=dtype))
        self._engine = None
        self._cached_key = None

    @property
    def masked_value(self) -> int:
        return self.auxiliary_vocab_size

    @property
    def placeholder_token_id(self) -> int:
        return self.text_vocab_size

    @torch.no_grad()
    def sample_prior(self, shape: tuple[int, ...], *, device: torch.device | str = "cpu", generator: torch.Generator | None = None) -> Tensor:
        return torch.randint(self.auxiliary_vocab_size, tuple(shape), device=device, generator=generator, dtype=torch.long)

    def embed(self, input_ids: Tensor, values: Tensor, aux_mask: Tensor) -> Tensor:
        sphere_values = values.reshape(-1).long().clamp(max=self.auxiliary_vocab_size - 1)
        vectors = self.auxiliary_vocabulary.index_select(0, sphere_values).view(*values.shape, -1)
        masked = (values == self.masked_value).unsqueeze(-1)
        return torch.where(masked, self.mask_auxiliary.to(dtype=vectors.dtype), vectors)

    def _engine_for(self, output_weight: Tensor) -> Voronoi:
        # Keyed on the projection's identity and version, so the V x K table is built once rather than every round.
        key = (output_weight.data_ptr(), output_weight._version, tuple(output_weight.shape))
        if self._cached_key != key:
            vocabulary = self.auxiliary_vocabulary.to(device=output_weight.device, dtype=output_weight.dtype)
            self._engine = None
            scores = torch.nn.functional.linear(vocabulary, output_weight)
            self._engine = Voronoi.from_table(scores.t().contiguous().neg())
            self._cached_key = key
        return self._engine

    def decode_unmasked(self, logits: Tensor, hidden_states: Tensor, output_weight: Tensor, values: Tensor) -> Tensor:
        """The token each position's auxiliary cell holds; the caller guarantees no value is the masked sentinel."""
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_values = values.reshape(-1).long().clamp(max=self.auxiliary_vocab_size - 1)
        engine = self._engine_for(output_weight)
        assignment = engine.assign(flat_logits, n_targets=self.top_m, k=min(self.iterate_shortlist, self.top_m),
                                   rounds=self.iterate_rounds, iterations=self.iterate_sweeps)
        slots = assignment.sigma.gather(1, flat_values[:, None])
        return assignment.tokens.gather(1, slots).squeeze(1).view(values.shape)


class LinearInterpolationAuxiliary(nn.Module):
    """A value u in [0, 1) blends learned vectors, and decodes by inverse CDF over the verify row's top-k probabilities."""

    def __init__(self, hidden_size: int, text_vocab_size: int, *, num_embeddings: int = 3, top_k: int = 128) -> None:
        super().__init__()
        self.text_vocab_size = int(text_vocab_size)
        self.top_k = int(top_k)
        # float32 like the checkpoint: the blend is cast to the model's dtype only once it is made
        self.embedding = nn.Embedding(int(num_embeddings), int(hidden_size), dtype=torch.float32)

    @property
    def masked_value(self) -> None:
        return None

    @torch.no_grad()
    def sample_prior(self, shape: tuple[int, ...], *, device: torch.device | str = "cpu", generator: torch.Generator | None = None) -> Tensor:
        return torch.rand(tuple(shape), device=device, generator=generator, dtype=torch.float32)

    def embed(self, input_ids: Tensor, values: Tensor, aux_mask: Tensor) -> Tensor:
        scaled = values.to(torch.float32) * (self.embedding.num_embeddings - 1)
        lower_f = torch.floor(scaled)
        lower = lower_f.long().clamp(0, self.embedding.num_embeddings - 1)
        upper = torch.clamp(lower + 1, max=self.embedding.num_embeddings - 1)
        frac = (scaled - lower_f).unsqueeze(-1)
        return self.embedding(lower) * (1.0 - frac) + self.embedding(upper) * frac

    def decode_unmasked(self, logits: Tensor, hidden_states: Tensor, output_weight: Tensor, values: Tensor) -> Tensor:
        """The bin holding u among the top-k probabilities at temperature 1, renormalised and ordered by token id."""
        probs = torch.softmax(logits.reshape(-1, logits.shape[-1]).float(), dim=-1)
        top_probs, top_indices = probs.topk(self.top_k, dim=-1)
        order = top_indices.argsort(dim=-1)
        probs_by_id = top_probs.gather(1, order)
        right_edges = (probs_by_id / probs_by_id.sum(dim=-1, keepdim=True)).cumsum(dim=-1)
        right_edges[:, -1] = 1.0
        bins = (right_edges > values.reshape(-1, 1).float()).to(torch.int8).argmax(dim=-1, keepdim=True)
        return top_indices.gather(1, order).gather(1, bins).squeeze(1).view(values.shape)


def build_auxiliary_scheme(config, dtype: torch.dtype | None = None) -> nn.Module:
    spec = dict(config.auxiliary_scheme)
    kind = spec.pop("kind")
    spec.pop("masked_value", None)
    if kind == "linear_interpolation":
        return LinearInterpolationAuxiliary(hidden_size=config.hidden_size, text_vocab_size=config.text_vocab_size, **spec)
    if kind != "logit_voronoi":
        raise ValueError(f"unsupported auxiliary scheme {kind!r}")
    return LogitVoronoiAuxiliary(auxiliary_vocab_size=spec.pop("auxiliary_vocab_size"), hidden_size=config.hidden_size,
                                 text_vocab_size=config.text_vocab_size, dtype=dtype, **spec)
