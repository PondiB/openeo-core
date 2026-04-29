"""openEO spec exceptions."""

from openeo_core.exceptions.ndvi import (
    BandExists,
    DimensionAmbiguous,
    NirBandAmbiguous,
    RedBandAmbiguous,
)
from openeo_core.exceptions.general import (
    ArrayElementNotAvailable,
    ArrayElementParameterConflict,
    ArrayElementParameterMissing,
    ArrayNotLabeled,
    DimensionLabelCountMismatch,
    DimensionNotAvailable,
    IncompatibleDataCubes,
    KernelDimensionsUneven,
    UnitMismatch,
)

__all__ = [
    "ArrayElementNotAvailable",
    "ArrayElementParameterConflict",
    "ArrayElementParameterMissing",
    "ArrayNotLabeled",
    "BandExists",
    "DimensionAmbiguous",
    "DimensionLabelCountMismatch",
    "DimensionNotAvailable",
    "IncompatibleDataCubes",
    "KernelDimensionsUneven",
    "NirBandAmbiguous",
    "RedBandAmbiguous",
    "UnitMismatch",
]
