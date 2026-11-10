"""The config class vLLM builds from a PLM export's ``config.json``: Llama fields plus the auxiliary-scheme metadata."""

from transformers.models.llama.configuration_llama import LlamaConfig


class PLMConfig(LlamaConfig):
    model_type = "plm"


class PTPConfig(PLMConfig):
    """The gated-LoRA distilled model keeps the PTP naming of its export; it loads through the PLM classes."""

    model_type = "ptp"
