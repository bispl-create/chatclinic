"""Compatibility shim for AYQ standalone plugin runtime.

This plugin runs AYQ inference with precomputed class embeddings from
`text_embeddings/single_class_embeddings_openclip`. In this path,
`open_clip` is imported by some detector modules but is not used directly.

If on-the-fly text encoding is needed in the future, install
`open_clip_torch` in the AYQ Python environment and remove this shim.
"""

from __future__ import annotations


def _not_available(*args, **kwargs):
    raise RuntimeError(
        "open_clip_torch is not installed in the active AYQ environment. "
        "This standalone plugin path uses precomputed class embeddings only."
    )


create_model_and_transforms = _not_available
get_tokenizer = _not_available
