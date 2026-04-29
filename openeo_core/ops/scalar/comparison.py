"""Comparison scalar processes."""

from __future__ import annotations

from typing import Any, Literal, overload

from openeo_core.ops.scalar._common import unwrap_scalar


def _type_tag(operand: Any) -> Literal["null", "boolean", "number", "string"] | None:
    if operand is None:
        return None
    if isinstance(operand, bool):
        return "boolean"
    if isinstance(operand, (int, float)):
        return "number"
    if isinstance(operand, str):
        return "string"
    raise TypeError(
        f"eq operands must be number, boolean, string, or null, got {type(operand)!r}"
    )


def _eq_numbers(a: Any, b: Any, delta: Any) -> bool:
    xa, xb = float(a), float(b)
    if delta is None:
        return xa == xb
    d = float(delta)
    if d > 0:
        return abs(xa - xb) <= d
    return xa == xb


def _eq_strings(sa: str, sb: str, case_sensitive: bool) -> bool:
    if case_sensitive:
        return sa == sb
    return sa.casefold() == sb.casefold()


@overload
def eq(
    x: None,
    y: Any,
    *,
    delta: float | int | None = ...,
    case_sensitive: bool = ...,
) -> None: ...


@overload
def eq(
    x: Any,
    y: None,
    *,
    delta: float | int | None = ...,
    case_sensitive: bool = ...,
) -> None: ...


@overload
def eq(
    x: float | int | bool | str,
    y: float | int | bool | str,
    *,
    delta: float | int | None = ...,
    case_sensitive: bool = ...,
) -> bool | None: ...


def eq(
    x: Any,
    y: Any,
    *,
    delta: float | int | None = None,
    case_sensitive: bool = True,
) -> bool | None:
    """Equal-to comparison implementing the ``eq`` openEO process."""
    x = unwrap_scalar(x)
    y = unwrap_scalar(y)

    if x is None or y is None:
        return None

    tx = _type_tag(x)
    ty = _type_tag(y)
    assert tx is not None and ty is not None

    if tx != ty:
        return False
    if tx == "number":
        return _eq_numbers(x, y, delta)
    if tx == "boolean":
        return bool(x) == bool(y)
    if tx == "string":
        return _eq_strings(str(x), str(y), case_sensitive)
    raise RuntimeError("unreachable")
