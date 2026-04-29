"""Shared scalar helper utilities."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def unwrap_scalar(operand: Any) -> Any:
    """Map numpy scalar / 0-d array to Python scalars for type checks."""
    if isinstance(operand, np.ndarray):
        if operand.ndim == 0:
            return operand.item()
        raise TypeError("scalar operands must be 0-dimensional")
    if isinstance(operand, np.generic):
        return operand.item()
    return operand


def normalize_array_like(data: Any, *, process_name: str) -> Sequence[Any]:
    """Validate 1-D array-like input and normalize numpy arrays to Python lists."""
    if isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise TypeError(f"`{process_name}` data must be a one-dimensional array")
        return data.tolist()
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        raise TypeError(f"`{process_name}` data must be an array-like sequence")
    return data
