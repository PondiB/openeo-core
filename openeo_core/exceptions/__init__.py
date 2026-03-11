"""openEO spec exceptions."""

from openeo_core.exceptions.ndvi import (
    BandExists,
    DimensionAmbiguous,
    NirBandAmbiguous,
    RedBandAmbiguous,
)
from openeo_core.exceptions.general import (
    DimensionNotAvailable,
    IncompatibleDataCubes,
    KernelDimensionsUneven,
    UnitMismatch,
)

__all__ = [
    "BandExists",
    "DimensionAmbiguous",
    "DimensionNotAvailable",
    "IncompatibleDataCubes",
    "KernelDimensionsUneven",
    "NirBandAmbiguous",
    "RedBandAmbiguous",
    "UnitMismatch",
]
