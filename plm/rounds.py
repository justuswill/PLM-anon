"""The PLM decoding configuration behind ``--setting``, ``--mode``, ``--match`` and ``--k``, and the round decoder it builds.

``--mode fast`` routes the round's five GEMM shapes through Triton tiles tuned on an RTX 6000 Ada (a process-global
patch of vLLM's linear path). ``--mode default`` is stock vLLM with no patching and runs on any GPU vLLM supports.
``--match exact`` accepts a proposal only if it equals the verify row's own decode, so the output distribution is the
model's own. ``--match relaxed`` also accepts proposals that pass a looser rule on the verify row, chosen per
``--setting``: the top-0.94 nucleus for post-trained models, probability above 2e-3 for pretrained ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PREFILL_BUCKETS = tuple(range(64, 641, 64))


@dataclass(frozen=True)
class Setting:
    default_k: dict[str, int]       # --match -> round width k
    relaxed_rule: tuple[str, float]  # (accept rule, threshold) for --match relaxed


# Fitted on this repository's GPU: k for tokens per second, the relaxed rule for the fastest rule within 1% MAUVE of exact.
SETTINGS = {
    "posttrain": Setting(default_k={"exact": 7, "relaxed": 10}, relaxed_rule=("nucleus", 0.94)),
    "pretrain": Setting(default_k={"exact": 6, "relaxed": 6}, relaxed_rule=("floor", 2e-3)),
}


def is_plm_export(model: str) -> bool:
    config = Path(model) / "config.json"
    return config.exists() and "PLMForCausalLM" in json.loads(config.read_text()).get("architectures", [])


def scheme_kind(scheme: str) -> str | None:
    """The auxiliary scheme an export decodes with: logit_voronoi for a PLM, linear_interpolation for the distilled model, None for plain AR."""
    config = json.loads((Path(scheme) / "config.json").read_text())
    return config.get("auxiliary_scheme", {}).get("kind")


@dataclass
class RoundConfig:
    model: str
    setting: str = "posttrain"
    mode: str = "fast"
    match: str = "exact"
    k: int | None = None
    dtype: str = "float16"
    max_model_len: int = 4096

    def __post_init__(self) -> None:
        if self.k is None:
            self.k = SETTINGS[self.setting].default_k[self.match]

    @property
    def accept(self) -> tuple[str, float | None]:
        """Exact match is the nucleus test switched off, in either setting."""
        return SETTINGS[self.setting].relaxed_rule if self.match == "relaxed" else ("nucleus", None)

    @property
    def width(self) -> int:
        return self.k + self.k * self.k

    def describe(self) -> dict:
        rule, threshold = self.accept
        return {"setting": self.setting, "mode": self.mode, "match": self.match, "k": self.k, "width": self.width, "dtype": self.dtype,
                "model": self.model, "accept": f"{rule} {threshold}" if threshold is not None else "exact"}

    def build(self, gpu_memory_utilization: float, kv_cache_bytes: int, block_size: int):
        from plm import gemm_tables
        from plm.decoder import RoundDecoder, build_llm

        llm = build_llm(self.model, dtype=self.dtype, max_model_len=self.max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                        kv_cache_memory_bytes=kv_cache_bytes, block_size=block_size)
        if self.mode == "fast":
            gemm_tables.install(llm.llm_engine.model_config.hf_config, self.width)
        decoder = RoundDecoder(llm, self.k, *self.accept, PREFILL_BUCKETS)
        decoder.prepare()
        return decoder
