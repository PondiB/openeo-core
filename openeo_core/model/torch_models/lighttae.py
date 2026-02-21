"""Lightweight Temporal Self-Attention Encoder (LightTAE).

Based on Garnot & Landrieu (2020): "Lightweight Temporal Self-attention
for Classifying Satellite Images Time Series."
https://doi.org/10.1007/978-3-030-65742-0_12

The encoder maps a 1-D feature sequence through:
  1. A linear input projection,
  2. Learned positional encoding,
  3. A stack of lightweight multi-head self-attention layers,
  4. Global mean pooling → MLP classification head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LightAttentionLayer(nn.Module):
    """Single lightweight multi-head self-attention layer.

    Uses a simplified attention where queries and keys share the same
    projection (reducing parameters), followed by layer normalisation and
    a small feed-forward network.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.qk_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qk = self.qk_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(qk, qk.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x


class LightTAE(nn.Module):
    """Lightweight Temporal Attention Encoder for time series classification.

    Parameters
    ----------
    n_features : int
        Length of the input time series (number of temporal features).
    n_classes : int
        Number of output classes.
    d_model : int
        Internal embedding dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of stacked attention layers.
    dropout : float
        Dropout probability throughout the network.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(1, d_model)

        self.pos_enc = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)

        self.layers = nn.ModuleList(
            [_LightAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, n_features)``. Each feature value is treated
            as one element in a temporal sequence.
        """
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # (B, T, 1)
        x = self.input_proj(x)  # (B, T, d_model)
        x = x + self.pos_enc[:, : x.size(1), :]

        for layer in self.layers:
            x = layer(x)

        x = x.mean(dim=1)  # global mean pooling → (B, d_model)
        return self.head(x)
