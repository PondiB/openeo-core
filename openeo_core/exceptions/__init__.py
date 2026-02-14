"""openEO spec exceptions."""

from openeo_core.exceptions.ndvi import (
    BandExists,
    DimensionAmbiguous,
    NirBandAmbiguous,
    RedBandAmbiguous,
)
from openeo_core.exceptions.general import (
    DimensionNotAvailable,
    KernelDimensionsUneven,
    UnitMismatch,
)

__all__ = [
    "BandExists",
    "DimensionAmbiguous",
    "DimensionNotAvailable",
    "KernelDimensionsUneven",
    "NirBandAmbiguous",
    "RedBandAmbiguous",
    "UnitMismatch",
]
