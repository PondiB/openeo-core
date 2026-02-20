"""Temporal Convolutional Neural Network (TempCNN).

Based on Pelletier et al. (2019): "Temporal Convolutional Neural Network
for the Classification of Satellite Image Time Series."
https://doi.org/10.3390/rs11050523

The network applies a stack of 1-D convolutions along the temporal axis,
followed by a fully connected classification head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TempCNN(nn.Module):
    """Temporal CNN for time series classification.

    Parameters
    ----------
    n_features : int
        Length of the input time series (number of temporal features).
    n_classes : int
        Number of output classes.
    cnn_layers : list[int]
        Number of filters in each convolutional layer.
    cnn_kernels : list[int]
        Kernel size for each convolutional layer.
    cnn_dropout_rates : list[float]
        Dropout rate after each convolutional layer.
    dense_layer_nodes : int
        Number of units in the fully-connected hidden layer.
    dense_layer_dropout_rate : float
        Dropout rate for the dense hidden layer.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        cnn_layers: list[int] | None = None,
        cnn_kernels: list[int] | None = None,
        cnn_dropout_rates: list[float] | None = None,
        dense_layer_nodes: int = 256,
        dense_layer_dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        if cnn_layers is None:
            cnn_layers = [256, 256, 256]
        if cnn_kernels is None:
            cnn_kernels = [7, 7, 7]
        if cnn_dropout_rates is None:
            cnn_dropout_rates = [0.2, 0.2, 0.2]

        n_conv = len(cnn_layers)
        if len(cnn_kernels) != n_conv or len(cnn_dropout_rates) != n_conv:
            raise ValueError(
                "cnn_layers, cnn_kernels, and cnn_dropout_rates must have the same length"
            )

        blocks: list[nn.Module] = []
        in_channels = 1
        for i in range(n_conv):
            blocks.append(
                nn.Conv1d(
                    in_channels,
                    cnn_layers[i],
                    kernel_size=cnn_kernels[i],
                    padding=cnn_kernels[i] // 2,
                )
            )
            blocks.append(nn.BatchNorm1d(cnn_layers[i]))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.Dropout(cnn_dropout_rates[i]))
            in_channels = cnn_layers[i]

        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(cnn_layers[-1], dense_layer_nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(dense_layer_dropout_rate),
            nn.Linear(dense_layer_nodes, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, n_features)``. Reshaped internally to
            ``(batch, 1, n_features)`` for the 1-D convolution.
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, 1, T)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # (B, C)
        return self.head(x)
