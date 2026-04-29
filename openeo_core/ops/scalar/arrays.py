"""Array-oriented scalar processes."""

from __future__ import annotations

from typing import Any

from openeo_core.exceptions import (
    ArrayElementNotAvailable,
    ArrayElementParameterConflict,
    ArrayElementParameterMissing,
    ArrayNotLabeled,
)
from openeo_core.ops.scalar._common import normalize_array_like

_MSG_ARRAY_EL_MISSING = (
    "The process `array_element` requires either the `index` or `labels` parameter to be set."
)
_MSG_ARRAY_EL_CONFLICT = (
    "The process `array_element` only allows that either the `index` or the `labels` parameter is set."
)
_MSG_ARRAY_EL_NOT_AVAILABLE = "The array has no element with the specified index or label."
_MSG_ARRAY_NOT_LABELED = (
    "The array is not a labeled array, but the `label` parameter is set. Use the `index` instead."
)


def array_element(
    data: Any,
    *,
    index: int | None = None,
    label: str | float | int | None = None,
    return_nodata: bool = False,
) -> Any:
    """Get one element from an array implementing ``array_element``."""
    if index is None and label is None:
        raise ArrayElementParameterMissing(_MSG_ARRAY_EL_MISSING)
    if index is not None and label is not None:
        raise ArrayElementParameterConflict(_MSG_ARRAY_EL_CONFLICT)

    seq = normalize_array_like(data, process_name="array_element")

    if label is not None:
        raise ArrayNotLabeled(_MSG_ARRAY_NOT_LABELED)

    assert index is not None
    if index < 0 or index >= len(seq):
        if return_nodata:
            return None
        raise ArrayElementNotAvailable(_MSG_ARRAY_EL_NOT_AVAILABLE)
    return seq[index]
