"""Lightweight shim to expose causal_conv1d symbols from our vendor fallback.

This lets code do `from causal_conv1d import causal_conv1d_fn` without pulling
in the CUDA package. It's just enough for imports to work in CPU-only envs.

Note:
- This shim intentionally does NOT provide CUDA kernels (no `causal_conv1d_cuda`).
- If you are on a GPU runtime and need CUDA acceleration (e.g., for mamba-ssm),
  ensure the real `causal-conv1d` package is installed and that your Python
  import order prefers site-packages over this repo, or remove this repo path
  from `sys.path` before importing.
"""

from __future__ import annotations

import os
import warnings

from vendor.causal_conv1d import CausalConv1d, causal_conv1d_fn  # re-export

# Emit a helpful warning in CUDA environments unless explicitly silenced.
if not os.environ.get("SSAMBA_SILENCE_SHIM_WARNING"):
    try:
        import torch  # noqa: F401
        if getattr(torch.cuda, "is_available", lambda: False)():
            warnings.warn(
                "Importing causal_conv1d shim (CPU fallback). On a GPU runtime this may "
                "prevent access to CUDA kernels from the real causal-conv1d package and "
                "cause mamba-ssm to error. Prefer installing causal-conv1d and ensure "
                "site-packages takes precedence over the repo on sys.path.",
                RuntimeWarning,
            )
    except Exception:
        # If torch is not available, stay silent.
        pass


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



