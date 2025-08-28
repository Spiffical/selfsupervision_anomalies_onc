"""Lightweight shim to expose causal_conv1d symbols from our vendor fallback.

This lets code do `from causal_conv1d import causal_conv1d_fn` without pulling
in the CUDA package. It's just enough for imports to work in CPU-only envs.
"""

from vendor.causal_conv1d import CausalConv1d, causal_conv1d_fn  # re-export


def causal_conv1d_update(*_args, **_kwargs):
    """Placeholder for streaming update API.

    Not implemented in the CPU fallback. It exists only so imports don't fail.
    """
    raise NotImplementedError(
        "causal_conv1d_update is not available in the CPU fallback shim"
    )


__all__ = [
    "CausalConv1d",
    "causal_conv1d_fn",
    "causal_conv1d_update",
]




