"""PyTorch backend – Phase 2 / roadmap stub.

Provides helpers for wrapping ``torch.nn.Module`` instances into
:class:`~openeo_core.model.mlm.MLModel` objects, and a batched
predict function.

This module is **internal**.  Users should use :func:`Model.torch` or
the direct ``MLModel`` construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from openeo_core.model.mlm import (
    MLModel,
    ModelInput,
    ModelOutput,
    InputStructure,
    ResultStructure,
)


def build_torch_model(
    *,
    module: Any,
    device: str = "cpu",
    batch_size: int = 1024,
    task: str = "regression",
) -> MLModel:
    """Wrap a ``torch.nn.Module`` in an :class:`MLModel`.

    .. note:: Phase 2 / roadmap.
    """
    import torch

    framework_version: str | None = None
    try:
        framework_version = torch.__version__
    except Exception:
        pass

    return MLModel(
        name="PyTorch Model",
        architecture="Custom",
        tasks=[task],
        framework="PyTorch",
        framework_version=framework_version,
        hyperparameters={"batch_size": batch_size, "device": device},
        estimator=_TorchEstimatorWrapper(module=module, device=device, batch_size=batch_size),
        backend="torch",
    )


class _TorchEstimatorWrapper:
    """Thin wrapper that gives a torch Module an sklearn-compatible
    ``fit`` / ``predict`` interface so it can be used by
    :func:`~openeo_core.model.base.ml_fit` and
    :func:`~openeo_core.model.base.ml_predict`.
    """

    def __init__(self, *, module: Any, device: str = "cpu", batch_size: int = 1024):
        self._module = module
        self._device = device
        self._batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        """Minimal training loop."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        epochs = kwargs.pop("epochs", 10)
        lr = kwargs.pop("lr", 1e-3)

        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self._device)

        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self._batch_size, shuffle=True)

        self._module.to(self._device)
        self._module.train()
        optimizer = torch.optim.Adam(self._module.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self._module(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Batched inference."""
        import torch

        self._module.to(self._device)
        self._module.eval()

        all_preds: list[np.ndarray] = []
        n = X.shape[0]
        for i in range(0, n, self._batch_size):
            batch = torch.tensor(X[i: i + self._batch_size], dtype=torch.float32).to(self._device)
            with torch.no_grad():
                out = self._module(batch).cpu().numpy().squeeze(-1)
            all_preds.append(out)

        return np.concatenate(all_preds, axis=0)
