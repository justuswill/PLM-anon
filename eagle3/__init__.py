"""The EAGLE-3 baseline: ``ar-1.3B`` with a draft head, as vLLM's own speculative decoding (``chain``) or as EAGLE-2's
dynamic draft tree (``tree``). Nothing here is shared with the PLM decoder in ``plm``.

``register`` is this package's own ``vllm.general_plugins`` entry point; like every vLLM plugin it must not import torch
or vLLM at module scope.
"""


def register() -> None:
    """Patches vLLM's draft sampling in every vLLM process when ``--eagle3-draft topk`` asked for it (see ``draft_topk``). Re-entrant."""
    import os

    if os.environ.get("EAGLE3_DRAFT_TOP_K"):
        from . import draft_topk

        draft_topk.install()
