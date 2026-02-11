"""openEO spec exceptions."""

from openeo_core.exceptions.ndvi import (
    BandExists,
    DimensionAmbiguous,
    NirBandAmbiguous,
    RedBandAmbiguous,
)

__all__ = [
    "BandExists",
    "DimensionAmbiguous",
    "NirBandAmbiguous",
    "RedBandAmbiguous",
]
