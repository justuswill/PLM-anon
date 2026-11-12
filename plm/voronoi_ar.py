"""Autoregressive decoding under a PLM's own sampling rule, as a vLLM logits processor (``--ar-sampler voronoi`` and ``icdf``).

An exact parallel round emits, at output position t, ``auxiliary_scheme.decode_unmasked`` of the causal row's logits with
``aux_stream[t]``: one prior draw per position from the request's seed, the same stream for every prompt. This processor
applies the same rule to one row per step on stock vLLM autoregression, so an AR model samples exactly as a PLM's exact
decode does. The seed is the request's ``SamplingParams.extra_args["aux_seed"]`` (default 0).

The scheme comes from ``PLM_AR_SCHEME_DIR``, a PLM export of the same hidden size whose config carries
``auxiliary_scheme``; the ``lm_head`` is the served model's own. With the distilled model's linear-interpolation scheme
this is top-128 sampling at temperature 1 by inverse CDF on that model's seeded random values, its exact decode run
autoregressively. Serves one request at a time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor

from .auxiliary import build_auxiliary_scheme

AUX_STREAM_LENGTH = 8192
EAGER_STEPS_BEFORE_CAPTURE = 2


def read_tensor(export: Path, name: str) -> torch.Tensor:
    for shard in sorted(export.glob("*.safetensors")):
        with safe_open(str(shard), "pt") as handle:
            if name in handle.keys():
                return handle.get_tensor(name)
    raise KeyError(f"{name} is not in {export}")


class VoronoiARLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool) -> None:
        scheme_dir = Path(os.environ["PLM_AR_SCHEME_DIR"])
        self.dtype = vllm_config.model_config.dtype
        scheme_config = SimpleNamespace(**json.loads((scheme_dir / "config.json").read_text()))
        self.scheme = build_auxiliary_scheme(scheme_config, dtype=self.dtype).to(device)
        if scheme_config.auxiliary_scheme["kind"] == "logit_voronoi":   # linear interpolation decodes from the logits alone
            self.scheme.auxiliary_vocabulary.copy_(read_tensor(scheme_dir, "auxiliary_scheme.auxiliary_vocabulary"))
        self.vocab = self.scheme.text_vocab_size
        head = read_tensor(Path(vllm_config.model_config.model), "lm_head.weight")
        self.lm_head_weight = head[: self.vocab].to(device, self.dtype)
        self.device = device
        self.stream = self.scheme.sample_prior((AUX_STREAM_LENGTH,), device=device)   # integer cells or uniform floats, per scheme
        self.seed: int | None = None
        self._draw_stream(0)
        self.output_ids: dict[int, list[int]] = {}
        # The one-row decode is captured once the engine's Triton autotune has run; a replay is then a handful of launches.
        self.eager_steps = 0
        self.graph = None
        self.row_in = torch.zeros(1, self.vocab, dtype=self.dtype, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.row_out = None

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.output_ids.pop(index, None)
        for index, params, _prompt_ids, output_ids in batch_update.added:
            self.output_ids[index] = output_ids
            self._draw_stream(int((params.extra_args or {}).get("aux_seed", 0)))

    def _draw_stream(self, seed: int) -> None:
        """Redrawn in place, so a captured decode keeps reading the live stream; the same draw a parallel round makes from this seed."""
        if seed == self.seed:
            return
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self.stream.copy_(self.scheme.sample_prior((AUX_STREAM_LENGTH,), device=self.device, generator=generator))
        self.seed = seed

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        rows = [index for index in self.output_ids if index < logits.shape[0]]
        if not rows:
            return logits
        if len(rows) > 1:
            raise NotImplementedError("the Voronoi AR sampler serves one request at a time")
        row = rows[0]
        self.position.fill_(len(self.output_ids[row]))
        self.row_in.copy_(logits[row : row + 1, : self.vocab])
        if self.graph is None and self.eager_steps >= EAGER_STEPS_BEFORE_CAPTURE:
            self.capture(logits.shape[1])
        if self.graph is None:
            self.eager_steps += 1
            logits[row : row + 1].copy_(self.forced_row(logits.shape[1]))
            return logits
        self.graph.replay()
        logits[row : row + 1].copy_(self.row_out)
        return logits

    def forced_row(self, width: int) -> torch.Tensor:
        """The row with every logit but the decoded token's set to -inf, so greedy sampling reads the token off."""
        # The sampler hands over float32 copies of the model's logits; casting back is exact.
        token = self.scheme.decode_unmasked(self.row_in.to(self.dtype), None, self.lm_head_weight, self.stream.index_select(0, self.position))
        row = torch.full((1, width), float("-inf"), dtype=torch.float32, device=self.row_in.device)
        return row.scatter_(1, token[:, None], 0.0)

    def capture(self, width: int) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.forced_row(width)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.row_out = self.forced_row(width)
        self.graph = graph
