from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """
    Causal 1D convolution using grouped conv and left padding.
    x: (batch, channels, length)
    weight: (channels, kernel_size)
    bias: (channels,) optional
    activation: "silu"/"swish" or None
    """
    batch_size, num_channels, seq_len = x.shape
    kernel_size = weight.shape[-1]

    y = F.conv1d(
        x,
        weight.unsqueeze(1),
        bias=bias,
        padding=kernel_size - 1,
        groups=num_channels,
    )
    y = y[..., :seq_len]

    if activation in ("silu", "swish"):
        y = y * torch.sigmoid(y)
    return y


class CausalConv1d(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        activation: Optional[str] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(dim, kernel_size))
        self.bias = nn.Parameter(torch.empty(dim)) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = dim * kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return causal_conv1d_fn(x, self.weight, self.bias, activation=self.activation)




