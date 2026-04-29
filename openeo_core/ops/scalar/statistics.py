"""Statistical scalar processes."""

from __future__ import annotations

from statistics import median as _py_median
from typing import Any

from openeo_core.ops.scalar._common import normalize_array_like, unwrap_scalar


def _as_number_or_null(value: Any) -> float | int | None:
    value = unwrap_scalar(value)
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("`median` data items must be number or null, not boolean")
    if isinstance(value, (int, float)):
        return value
    raise TypeError(f"`median` data items must be number or null, got {type(value)!r}")


def median(data: Any, *, ignore_nodata: bool = True) -> float | int | None:
    """Statistical median implementing the ``median`` openEO process."""
    seq = normalize_array_like(data, process_name="median")
    values = [_as_number_or_null(v) for v in seq]
    if not ignore_nodata and any(v is None for v in values):
        return None

    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return _py_median(filtered)
