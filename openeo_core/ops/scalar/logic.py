"""Logical scalar processes."""

from __future__ import annotations

from typing import Any

from openeo_core.ops.scalar._common import unwrap_scalar


def _as_bool_or_null(operand: Any) -> bool | None:
    operand = unwrap_scalar(operand)
    if operand is None or isinstance(operand, bool):
        return operand
    raise TypeError(
        "`or` operands must be boolean or null (openEO booleans-only), "
        f"got {type(operand)!r}"
    )


def or_(x: Any, y: Any) -> bool | None:
    """Logical OR implementing the ``or`` openEO process."""
    xa = _as_bool_or_null(x)
    ya = _as_bool_or_null(y)

    if xa is True:
        return True
    if xa is False:
        if ya is True:
            return True
        if ya is False:
            return False
        return None
    if ya is True:
        return True
    return None
