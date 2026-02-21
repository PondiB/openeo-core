"""PyTorch backend – estimator wrapper and model builders.

Provides :class:`TorchClassifierWrapper`, a thin adapter that gives a
``torch.nn.Module`` an sklearn-compatible ``fit`` / ``predict`` interface
so it integrates with :func:`~openeo_core.model.base.ml_fit` and
:func:`~openeo_core.model.base.ml_predict`.

Also contains builder functions that construct the specific module
architectures (TempCNN, LightTAE) from openEO process parameters.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Optimizer mapping – openEO spec name → (module_path, class_name)
# -----------------------------------------------------------------------

_BUILTIN_OPTIMIZERS: dict[str, str] = {
    "adam": "Adam",
    "nadam": "NAdam",
    "radam": "RAdam",
}

_TORCH_OPT_OPTIMIZERS: dict[str, str] = {
    "adabound": "AdaBound",
    "adabelief": "AdaBelief",
    "madagrad": "MADGRAD",
    "qhadam": "QHAdam",
    "swats": "SWATS",
    "yogi": "Yogi",
}


def _build_optimizer(
    name: str,
    params: Any,
    lr: float,
    *,
    weight_decay: float = 0.0,
    eps: float = 1e-8,
) -> Any:
    """Instantiate an optimizer by its openEO spec name."""
    import torch

    common: dict[str, Any] = {"lr": lr}

    if name in _BUILTIN_OPTIMIZERS:
        cls = getattr(torch.optim, _BUILTIN_OPTIMIZERS[name])
        if name in ("adam", "nadam", "radam"):
            common["eps"] = eps
            common["weight_decay"] = weight_decay
        return cls(params, **common)

    if name in _TORCH_OPT_OPTIMIZERS:
        import torch_optimizer

        cls = getattr(torch_optimizer, _TORCH_OPT_OPTIMIZERS[name])
        if name in ("adabound", "adabelief", "qhadam", "yogi"):
            common["eps"] = eps
            common["weight_decay"] = weight_decay
        elif name == "madagrad":
            common["weight_decay"] = weight_decay
        elif name == "swats":
            common["eps"] = eps
            common["weight_decay"] = weight_decay
        return cls(params, **common)

    raise ValueError(
        f"Unknown optimizer {name!r}. Supported: "
        f"{sorted(list(_BUILTIN_OPTIMIZERS) + list(_TORCH_OPT_OPTIMIZERS))}"
    )


# -----------------------------------------------------------------------
# TorchClassifierWrapper – sklearn-compatible fit/predict for torch
# -----------------------------------------------------------------------


class TorchClassifierWrapper:
    """Wraps a ``torch.nn.Module`` factory with an sklearn-compatible API.

    Rather than storing a pre-built module (which would need to know
    ``n_features`` and ``n_classes`` at init time), this wrapper stores
    the *factory callable* and its keyword arguments.  The actual module
    is constructed lazily inside :meth:`fit`, once the data shape and
    label set are known.

    Parameters
    ----------
    module_factory : callable
        A callable ``(n_features, n_classes, **kwargs) -> nn.Module``.
    module_kwargs : dict
        Extra keyword arguments forwarded to the factory.
    epochs : int
        Number of training epochs.
    batch_size : int
        Training mini-batch size.
    optimizer_name : str
        Optimizer name (openEO spec).
    learning_rate : float
        Base learning rate.
    eps : float
        Epsilon for the optimizer.
    weight_decay : float
        L2 regularisation.
    lr_decay_epochs : int
        Epochs between LR decay steps (``StepLR``).
    lr_decay_rate : float
        Multiplicative factor for LR decay.
    seed : int | None
        Random seed.
    device : str
        Torch device string.
    """

    def __init__(
        self,
        *,
        module_factory: Any,
        module_kwargs: dict[str, Any] | None = None,
        epochs: int = 100,
        batch_size: int = 64,
        optimizer_name: str = "adam",
        learning_rate: float = 1e-3,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        lr_decay_epochs: int = 50,
        lr_decay_rate: float = 1.0,
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        self._module_factory = module_factory
        self._module_kwargs = dict(module_kwargs or {})
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.eps = eps
        self.weight_decay = weight_decay
        self.lr_decay_epochs = lr_decay_epochs
        self.lr_decay_rate = lr_decay_rate
        self.seed = seed
        self.device = device

        self._module: Any = None
        self._classes: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the classifier."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self._classes = np.unique(y)
        n_classes = len(self._classes)
        n_features = X.shape[1]

        label_map = {c: i for i, c in enumerate(self._classes)}
        y_mapped = np.array([label_map[v] for v in y], dtype=np.int64)

        self._module = self._module_factory(
            n_features=n_features,
            n_classes=n_classes,
            **self._module_kwargs,
        )
        self._module.to(self.device)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_mapped, dtype=torch.long)

        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        optimizer = _build_optimizer(
            self.optimizer_name,
            self._module.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=self.eps,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.lr_decay_epochs,
            gamma=self.lr_decay_rate,
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        self._module.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self._module(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                logger.info(
                    "epoch %d/%d  loss=%.4f  lr=%.2e",
                    epoch + 1,
                    self.epochs,
                    epoch_loss / len(loader),
                    scheduler.get_last_lr()[0],
                )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels (mapped back to original values)."""
        import torch

        if self._module is None or self._classes is None:
            raise RuntimeError("Model has not been fitted.")

        self._module.to(self.device)
        self._module.eval()

        all_preds: list[np.ndarray] = []
        n = X.shape[0]
        for i in range(0, n, self.batch_size):
            batch = torch.tensor(
                X[i : i + self.batch_size], dtype=torch.float32
            ).to(self.device)
            with torch.no_grad():
                logits = self._module(batch)
                indices = np.asarray(logits.argmax(dim=-1).cpu().tolist())
            all_preds.append(indices)

        indices = np.concatenate(all_preds, axis=0)
        return self._classes[indices]

    def __deepcopy__(self, memo: dict) -> "TorchClassifierWrapper":
        """Support ``copy.deepcopy`` for the :func:`ml_fit` clone step."""
        new = TorchClassifierWrapper(
            module_factory=self._module_factory,
            module_kwargs=copy.deepcopy(self._module_kwargs, memo),
            epochs=self.epochs,
            batch_size=self.batch_size,
            optimizer_name=self.optimizer_name,
            learning_rate=self.learning_rate,
            eps=self.eps,
            weight_decay=self.weight_decay,
            lr_decay_epochs=self.lr_decay_epochs,
            lr_decay_rate=self.lr_decay_rate,
            seed=self.seed,
            device=self.device,
        )
        if self._module is not None:
            import torch
            import io

            buf = io.BytesIO()
            torch.save(self._module.state_dict(), buf)
            buf.seek(0)

            new._module = self._module_factory(
                n_features=next(iter(self._module.parameters())).shape[-1]
                if any(True for _ in self._module.parameters())
                else 1,
                n_classes=len(self._classes) if self._classes is not None else 2,
                **self._module_kwargs,
            )
            new._module.load_state_dict(torch.load(buf, weights_only=True))
        new._classes = self._classes.copy() if self._classes is not None else None
        return new


# -----------------------------------------------------------------------
# Builder functions (called from base.py factory functions)
# -----------------------------------------------------------------------


def build_tempcnn_estimator(
    *,
    cnn_layers: list[int] | None = None,
    cnn_kernels: list[int] | None = None,
    cnn_dropout_rates: list[float] | None = None,
    dense_layer_nodes: int = 256,
    dense_layer_dropout_rate: float = 0.5,
    epochs: int = 100,
    batch_size: int = 64,
    optimizer: str = "adam",
    learning_rate: float = 0.001,
    seed: int | None = None,
) -> TorchClassifierWrapper:
    """Build a :class:`TorchClassifierWrapper` for TempCNN."""
    from openeo_core.model.torch_models.tempcnn import TempCNN

    module_kwargs: dict[str, Any] = {
        "cnn_layers": cnn_layers,
        "cnn_kernels": cnn_kernels,
        "cnn_dropout_rates": cnn_dropout_rates,
        "dense_layer_nodes": dense_layer_nodes,
        "dense_layer_dropout_rate": dense_layer_dropout_rate,
    }
    return TorchClassifierWrapper(
        module_factory=TempCNN,
        module_kwargs=module_kwargs,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_name=optimizer,
        learning_rate=learning_rate,
        seed=seed,
    )


def build_lighttae_estimator(
    *,
    epochs: int = 150,
    batch_size: int = 128,
    optimizer: str = "adam",
    learning_rate: float = 0.0005,
    epsilon: float = 1e-8,
    weight_decay: float = 0.0007,
    lr_decay_epochs: int = 50,
    lr_decay_rate: float = 1.0,
    seed: int | None = None,
) -> TorchClassifierWrapper:
    """Build a :class:`TorchClassifierWrapper` for LightTAE."""
    from openeo_core.model.torch_models.lighttae import LightTAE

    return TorchClassifierWrapper(
        module_factory=LightTAE,
        module_kwargs={},
        epochs=epochs,
        batch_size=batch_size,
        optimizer_name=optimizer,
        learning_rate=learning_rate,
        eps=epsilon,
        weight_decay=weight_decay,
        lr_decay_epochs=lr_decay_epochs,
        lr_decay_rate=lr_decay_rate,
        seed=seed,
    )
