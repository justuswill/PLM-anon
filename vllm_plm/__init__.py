"""vLLM plugin registering the PLM model classes.

Nothing here may import torch, vLLM or transformers at module scope: vLLM loads plugins in every process it spawns, and
initializing CUDA before the fork raises. The models are registered by ``"module:Class"`` string so they are imported only
when one is built.
"""


def register() -> None:
    """Entry point for the ``vllm.general_plugins`` group. Must be re-entrant."""
    from transformers import AutoConfig
    from vllm import ModelRegistry

    from .config import PLMConfig, PTPConfig

    AutoConfig.register("plm", PLMConfig, exist_ok=True)
    AutoConfig.register("ptp", PTPConfig, exist_ok=True)
    # PTPForCausalLM is the gated-LoRA distilled model's architecture name; it is the same class.
    for architecture, model_class in (("PLMForCausalLM", "PLMForCausalLM"), ("PLMBackboneForCausalLM", "PLMBackboneForCausalLM"), ("PTPForCausalLM", "PLMForCausalLM")):
        if architecture not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(architecture, f"vllm_plm.model:{model_class}")
