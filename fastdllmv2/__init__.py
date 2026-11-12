"""The Fast-dLLM v2 baseline: ``fdv2-1.3B``, ar-1.3B converted to block diffusion, decoded by its reference threshold
sampler on vLLM. Nothing here is shared with the PLM decoder in ``plm``.

``register`` is this package's own ``vllm.general_plugins`` entry point; like every vLLM plugin it must not import torch
or vLLM at module scope.
"""


def register() -> None:
    """Makes the FDv2 model class visible to vLLM's engine in every process it spawns. Re-entrant."""
    from vllm import ModelRegistry

    if "Fdv2ForBlockDiffusion" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("Fdv2ForBlockDiffusion", "fastdllmv2.sampler:Fdv2ForBlockDiffusion")
