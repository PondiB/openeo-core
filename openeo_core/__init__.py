"""openeo-core – standalone Python library for openEO processes, data cubes, and ML models."""

from openeo_core.datacube import DataCube
from openeo_core.exceptions import (
    BandExists,
    DimensionAmbiguous,
    DimensionNotAvailable,
    KernelDimensionsUneven,
    NirBandAmbiguous,
    RedBandAmbiguous,
    UnitMismatch,
)
from openeo_core.types import Cube, RasterCube, VectorCube

__all__ = [
    "DataCube",
    "Cube",
    "RasterCube",
    "VectorCube",
    # openEO spec exceptions
    "BandExists",
    "DimensionAmbiguous",
    "DimensionNotAvailable",
    "KernelDimensionsUneven",
    "NirBandAmbiguous",
    "RedBandAmbiguous",
    "UnitMismatch",
]

__version__ = "0.1.0"
